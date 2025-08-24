from dataclasses import dataclass
from pathlib import Path
import logging
import warnings

import numpy as np
import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
logger = logging.getLogger(__name__)


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
        use_mixed_precision: bool = True,
        compile_models: bool = True,
        optimize_memory: bool = True,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device != "cpu"
        self.compile_models = compile_models and device != "cpu"
        self.optimize_memory = optimize_memory
        
        # Set autocast dtype for mixed precision
        self.autocast_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        
        # Apply optimizations
        self._apply_optimizations()
        
        logger.info(f"ChatterboxTTS initialized with optimizations: "
                   f"mixed_precision={self.use_mixed_precision}, "
                   f"compile={self.compile_models}, "
                   f"memory_opt={self.optimize_memory}")

    def _apply_optimizations(self):
        """Apply various optimizations for speed and memory efficiency"""
        try:
            # Enable mixed precision for models
            if self.use_mixed_precision:
                self._enable_mixed_precision()
            
            # Compile models for speed
            if self.compile_models:
                self._compile_models()
                
            # Memory optimizations
            if self.optimize_memory:
                self._optimize_memory()
                
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")

    def _enable_mixed_precision(self):
        """Enable mixed precision inference"""
        try:
            # Convert models to half precision where beneficial
            if hasattr(self.s3gen, 'flow') and hasattr(self.s3gen.flow, 'decoder'):
                # Keep flow matching in FP32 for stability, but enable autocast
                pass
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.warning(f"Mixed precision setup failed: {e}")

    def _compile_models(self):
        """Compile models with torch.compile for speed"""
        try:
            # Compile S3Gen inference for speed
            if hasattr(self.s3gen, 'inference'):
                self.s3gen.inference = torch.compile(
                    self.s3gen.inference,
                    mode="reduce-overhead",
                    fullgraph=False
                )
            
            # Compile HiFiGAN inference
            if hasattr(self.s3gen, 'mel2wav') and hasattr(self.s3gen.mel2wav, 'inference'):
                self.s3gen.mel2wav.inference = torch.compile(
                    self.s3gen.mel2wav.inference,
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                
            logger.info("Models compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")

    def _optimize_memory(self):
        """Apply memory optimizations"""
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory format for better performance
            if self.device.startswith('cuda'):
                torch.backends.cudnn.benchmark = True
                
            logger.info("Memory optimizations applied")
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        optimize_quality=True,
    ):
        """
        Generate speech from text with optimizations for speed and quality.
        
        Args:
            text: Input text to synthesize
            audio_prompt_path: Path to reference audio file
            exaggeration: Emotion intensity (0.25-2.0, default 0.5)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for T3 generation
            max_new_tokens: Maximum tokens to generate
            optimize_quality: Apply quality optimizations
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if abs(exaggeration - self.conds.t3.emotion_adv[0, 0, 0].item()) > 1e-6:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device, dtype=self.autocast_dtype),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Use mixed precision for inference
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.split(':')[0], dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                try:
                    # T3 inference with optimizations
                    speech_tokens = self.t3.inference(
                        t3_cond=self.conds.t3,
                        text_tokens=text_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        cfg_weight=cfg_weight,
                    )
                    
                    # Extract only the conditional batch
                    if speech_tokens.dim() > 1:
                        speech_tokens = speech_tokens[0]

                    # Clean up tokens - remove invalid tokens and apply vocab limit
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]
                    speech_tokens = speech_tokens.to(self.device)

                    # S3Gen inference with quality optimizations
                    if optimize_quality:
                        # Apply quality improvements
                        wav, _ = self._generate_with_quality_optimization(speech_tokens)
                    else:
                        wav, _ = self.s3gen.inference(
                            speech_tokens=speech_tokens,
                            ref_dict=self.conds.gen,
                        )
                    
                    # Convert to numpy and apply watermark
                    wav = wav.squeeze(0).detach().cpu().numpy()
                    
                    # Apply quality post-processing
                    if optimize_quality:
                        wav = self._post_process_audio(wav)
                    
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    # Fallback to non-optimized generation
                    return self._fallback_generate(text, audio_prompt_path, exaggeration, cfg_weight, temperature)
                    
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _generate_with_quality_optimization(self, speech_tokens):
        """Generate audio with quality optimizations to reduce glitches"""
        # Use a smaller cache to reduce artifacts
        cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.autocast_dtype)
        
        wav, sources = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            cache_source=cache_source,
            finalize=True,
        )
        return wav, sources

    def _post_process_audio(self, wav):
        """Apply post-processing to improve audio quality"""
        # Apply gentle high-pass filter to remove DC offset
        if len(wav) > 1024:  # Only for longer audio
            from scipy import signal
            try:
                # High-pass filter at 80Hz to remove rumble
                sos = signal.butter(2, 80, btype='high', fs=self.sr, output='sos')
                wav = signal.sosfilt(sos, wav)
                
                # Gentle normalization to prevent clipping
                max_val = np.abs(wav).max()
                if max_val > 0.95:
                    wav = wav * (0.95 / max_val)
                    
            except ImportError:
                logger.warning("scipy not available for audio post-processing")
                
        return wav

    def _fallback_generate(self, text, audio_prompt_path, exaggeration, cfg_weight, temperature):
        """Fallback generation without optimizations"""
        logger.info("Using fallback generation")
        
        # Simple generation without optimizations
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
            
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            
            if speech_tokens.dim() > 1:
                speech_tokens = speech_tokens[0]
                
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)
            
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
