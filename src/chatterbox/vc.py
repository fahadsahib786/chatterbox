from pathlib import Path
import logging
import numpy as np

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"
logger = logging.getLogger(__name__)


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
        use_mixed_precision: bool = True,
        compile_models: bool = True,
        optimize_quality: bool = True,
    ):
        self.sr = S3GEN_SR
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device != "cpu"
        self.compile_models = compile_models and device != "cpu"
        self.optimize_quality = optimize_quality
        
        # Validate device and available features before applying optimizations.
        # If a CUDA device was requested but no CUDA runtime is available, gracefully fall back to CPU.
        if "cuda" in self.device and not torch.cuda.is_available():
            logger.warning(f"Requested CUDA device {self.device} but CUDA is unavailable. Falling back to CPU.")
            self.device = "cpu"
        # Similarly, fall back if an MPS device was requested and is not available.
        if "mps" in self.device and hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
            logger.warning("MPS device requested but not available; falling back to CPU.")
            self.device = "cpu"
        # Disable model compilation if torch.compile is not present (e.g. on older PyTorch versions).
        if self.compile_models and not hasattr(torch, "compile"):
            logger.warning("torch.compile not available in this PyTorch build; disabling model compilation.")
            self.compile_models = False
        # Mixed precision requires a CUDA or MPS device; disable it on CPU.
        if self.use_mixed_precision and self.device == "cpu":
            logger.warning("Mixed precision requires a CUDA or MPS device; disabling mixed precision.")
            self.use_mixed_precision = False

        # Set autocast dtype for mixed precision (float16) or full precision (float32)
        self.autocast_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        
        self.s3gen = s3gen
        self.watermarker = perth.PerthImplicitWatermarker()
        
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }
        
        # Apply optimizations
        self._apply_optimizations()
        
        logger.info(f"ChatterboxVC initialized with optimizations: "
                   f"mixed_precision={self.use_mixed_precision}, "
                   f"compile={self.compile_models}, "
                   f"quality_opt={self.optimize_quality}")

    def _apply_optimizations(self):
        """Apply various optimizations for speed and memory efficiency"""
        try:
            # Compile models for speed
            if self.compile_models:
                self._compile_models()
                
            # Memory optimizations
            if self.use_mixed_precision:
                self._enable_mixed_precision()
                
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")

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
            logger.info("VC models compiled successfully")
        except Exception as e:
            logger.warning(f"VC model compilation failed: {e}")

    def _enable_mixed_precision(self):
        """Enable mixed precision optimizations"""
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory format for better performance
            if self.device.startswith('cuda'):
                torch.backends.cudnn.benchmark = True
                
            logger.info("VC mixed precision enabled")
        except Exception as e:
            logger.warning(f"VC mixed precision setup failed: {e}")

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
        optimize_quality=None,
    ):
        """
        Generate voice conversion with optimizations for speed and quality.
        
        Args:
            audio: Input audio file path or tensor
            target_voice_path: Path to target voice reference
            optimize_quality: Override quality optimization setting
        """
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            # Provide a clearer error message when no voice reference has been set.
            if self.ref_dict is None:
                raise RuntimeError(
                    "No target voice reference provided. Call `set_target_voice` with a reference sample "
                    "or specify `target_voice_path` before calling generate()."
                )

        # Use instance setting if not overridden
        if optimize_quality is None:
            optimize_quality = self.optimize_quality

        # Use mixed precision for inference
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.split(':')[0], dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                try:
                    # Load and preprocess audio with quality improvements
                    if isinstance(audio, str):
                        audio_16, _ = librosa.load(audio, sr=S3_SR)
                        if optimize_quality:
                            audio_16 = self._preprocess_audio(audio_16)
                    elif isinstance(audio, np.ndarray):
                        audio_16 = audio
                    elif torch.is_tensor(audio):
                        audio_16 = audio.detach().cpu().numpy()
                    else:
                        raise TypeError("Unsupported audio type; expected file path, numpy array, or torch tensor.")
                    
                    audio_16 = torch.from_numpy(np.asarray(audio_16, dtype=np.float32)).to(self.device)[None, ]

                    # Tokenize audio
                    s3_tokens, _ = self.s3gen.tokenizer(audio_16)
                    
                    # Generate with quality optimizations
                    if optimize_quality:
                        wav, _ = self._generate_with_quality_optimization(s3_tokens)
                    else:
                        wav, _ = self.s3gen.inference(
                            speech_tokens=s3_tokens,
                            ref_dict=self.ref_dict,
                        )
                    
                    # Convert to numpy and apply post-processing
                    wav = wav.squeeze(0).detach().cpu().numpy()
                    
                    # Apply quality post-processing
                    if optimize_quality:
                        wav = self._post_process_audio(wav)
                    
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    
                except Exception as e:
                    logger.error(f"VC generation failed: {e}")
                    # Fallback to non-optimized generation
                    return self._fallback_generate(audio, target_voice_path)
                    
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _preprocess_audio(self, audio):
        """Apply pre-processing to improve input audio quality"""
        # Normalize audio level
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
            
        # Apply gentle high-pass filter to remove low-frequency noise
        if len(audio) > 1024:
            try:
                from scipy import signal
                # High-pass filter at 50Hz
                sos = signal.butter(2, 50, btype='high', fs=S3_SR, output='sos')
                audio = signal.sosfilt(sos, audio)
            except ImportError:
                logger.warning("scipy not available for audio preprocessing")
                
        return audio

    def _generate_with_quality_optimization(self, speech_tokens):
        """Generate audio with quality optimizations"""
        # Use optimized cache for better quality
        cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.autocast_dtype)
        
        wav, sources = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.ref_dict,
            cache_source=cache_source,
            finalize=True,
        )
        return wav, sources

    def _post_process_audio(self, wav):
        """Apply post-processing to improve output audio quality"""
        # Apply gentle normalization
        max_val = np.abs(wav).max()
        if max_val > 0.95:
            wav = wav * (0.95 / max_val)
            
        # Remove DC offset
        if len(wav) > 1024:
            wav = wav - np.mean(wav)
            
        return wav

    def _fallback_generate(self, audio, target_voice_path):
        """Fallback generation without optimizations"""
        logger.info("Using fallback VC generation")
        
        with torch.inference_mode():
            if isinstance(audio, str):
                audio_16, _ = librosa.load(audio, sr=S3_SR)
            else:
                audio_16 = audio
                
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]
            
            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
