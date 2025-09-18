"""Whisper adapter for speech-to-text functionality."""

from typing import Any, Optional

import numpy as np

from live_translation.core.config import settings
from live_translation.core.exceptions import ModelLoadError, TranslationError
from live_translation.translation.engine import BaseSpeechToTextEngine
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperAdapter(BaseSpeechToTextEngine):
    """OpenAI Whisper adapter for speech-to-text with Apple Silicon optimizations."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        model_name = model_name or settings.whisper_model
        device = device or settings.device
        super().__init__(model_name, device)
        self._whisper = None

    @staticmethod
    def get_optimal_device() -> str:
        """Get the optimal device for the current system."""
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                return "cuda"  # NVIDIA GPU
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get system information for optimization reporting."""
        try:
            import platform

            import torch

            info = {
                "platform": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            }

            if torch.backends.mps.is_available():
                info["apple_silicon"] = True
                info["mps_available"] = True
            else:
                info["apple_silicon"] = False
                info["mps_available"] = False

            info["cuda_available"] = torch.cuda.is_available(
            ) if hasattr(torch, 'cuda') else False

            return info
        except ImportError:
            return {"error": "PyTorch not available for system info"}

    def _load_model(self) -> Any:
        """Load the Whisper model with Apple Silicon optimizations."""
        try:
            import torch
            import whisper

            logger.info(f"Loading Whisper model: {self.model_name}")

            # Apple Silicon optimizations with fallback
            device = self.device
            if torch.backends.mps.is_available() and self.device in ["auto", "mps"]:
                device = "mps"
                logger.info(
                    "🍎 Attempting to use Apple Silicon MPS acceleration")
                try:
                    # Try loading on MPS first
                    model = whisper.load_model(
                        self.model_name,
                        device=device,
                        download_root=str(settings.model_cache_dir),
                    )
                    logger.info("✅ MPS acceleration enabled successfully")
                    return model
                except Exception as mps_error:
                    logger.warning(
                        f"❌ MPS failed, falling back to CPU: {str(mps_error)}")
                    device = "cpu"
            elif self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # Download and load model with optimizations
            model = whisper.load_model(
                self.model_name,
                device=device,
                download_root=str(settings.model_cache_dir),
            )

            # Apply Apple Silicon specific optimizations
            if device == "mps":
                # Enable memory efficient attention for Apple Silicon
                if hasattr(model, 'encoder'):
                    model.encoder = model.encoder.to(torch.device('mps'))
                if hasattr(model, 'decoder'):
                    model.decoder = model.decoder.to(torch.device('mps'))

                logger.info("✅ Applied MPS optimizations for Apple Silicon")

            self._whisper = whisper
            self.device = device  # Update device to actual used device
            return model

        except ImportError as e:
            raise ModelLoadError(
                "Whisper library not installed. Install with: pip install openai-whisper",
                error_code="WHISPER_NOT_INSTALLED",
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model {self.model_name}: {str(e)}",
                error_code="WHISPER_LOAD_FAILED",
                details={"model_name": self.model_name, "device": self.device},
            ) from e

    def _transcribe_impl(
        self, audio_data: bytes, sample_rate: int, language: Optional[str]
    ) -> tuple[str, str]:
        """Implement Whisper transcription with Apple Silicon optimizations."""
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data, sample_rate)

            # Apple Silicon optimized transcription options
            options = {
                "language": language,
                "task": "transcribe",
                # MPS doesn't support fp16 well
                "fp16": False if self.device in ["cpu", "mps"] else True,
                "verbose": False,  # Reduce overhead
            }

            # Apple Silicon specific optimizations
            if self.device == "mps":
                # Use optimized parameters for Apple Silicon
                options.update({
                    "beam_size": 1,  # Faster inference on MPS
                    "best_of": 1,    # Reduce computation
                    "temperature": 0.0,  # Deterministic output
                })

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Transcribe with optimizations
            result = self._model.transcribe(audio_array, **options)

            text = result.get("text", "").strip()
            detected_language = result.get("language", language or "en")

            logger.debug(
                f"Whisper transcription: '{text[:100]}...' "
                f"(language: {detected_language})"
            )

            return text, detected_language

        except Exception as e:
            raise TranslationError(
                f"Whisper transcription failed: {str(e)}",
                error_code="WHISPER_TRANSCRIPTION_FAILED",
                details={
                    "sample_rate": sample_rate,
                    "audio_size": len(audio_data),
                    "language": language,
                },
            ) from e

    def _bytes_to_audio_array(self, audio_data: bytes, sample_rate: int) -> np.ndarray:
        """Convert audio bytes to numpy array format expected by Whisper."""
        try:
            # For WAV format, we need to handle the audio data properly
            # This is a simplified implementation - in practice, you might want
            # to use libraries like librosa or soundfile for more robust audio loading

            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 and normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0

            # Whisper expects 16kHz, so resample if necessary
            if sample_rate != 16000:
                audio_array = self._resample_audio(
                    audio_array, sample_rate, 16000)

            return audio_array

        except Exception as e:
            raise TranslationError(
                f"Failed to convert audio data: {str(e)}",
                error_code="AUDIO_CONVERSION_FAILED",
            ) from e

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast'
            )
        except ImportError:
            # Fallback: simple linear interpolation (not ideal but works)
            logger.warning("librosa not available, using simple resampling")
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> tuple[str, str]:
        """Transcribe audio file directly with Apple Silicon optimizations."""
        if not self._loaded:
            self.load()

        try:
            # Apple Silicon optimized options
            options = {
                "language": language,
                "task": "transcribe",
                # MPS optimization
                "fp16": False if self.device in ["cpu", "mps"] else True,
                "verbose": False,
            }

            # Apply Apple Silicon specific optimizations for file transcription
            if self.device == "mps":
                options.update({
                    "beam_size": 1,  # Faster on Apple Silicon
                    "best_of": 1,    # Reduce computation overhead
                    "temperature": 0.0,  # Deterministic results
                })

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            result = self._model.transcribe(file_path, **options)

            text = result.get("text", "").strip()
            detected_language = result.get("language", language or "en")

            return text, detected_language

        except Exception as e:
            raise TranslationError(
                f"File transcription failed: {str(e)}",
                error_code="FILE_TRANSCRIPTION_FAILED",
                details={"file_path": file_path, "language": language},
            ) from e


class WhisperStreamAdapter(WhisperAdapter):
    """Streaming version of Whisper adapter for real-time processing."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        chunk_length_s: int = 30,
    ) -> None:
        super().__init__(model_name, device)
        self.chunk_length_s = chunk_length_s
        self.audio_buffer = np.array([], dtype=np.float32)

    def add_audio_chunk(self, audio_data: bytes, sample_rate: int) -> Optional[tuple[str, str]]:
        """Add audio chunk to buffer and transcribe if ready."""
        # Convert and add to buffer
        audio_array = self._bytes_to_audio_array(audio_data, sample_rate)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])

        # Check if we have enough audio for transcription
        required_samples = self.chunk_length_s * 16000  # Assuming 16kHz target

        if len(self.audio_buffer) >= required_samples:
            # Take chunk and transcribe
            chunk = self.audio_buffer[:required_samples]
            result = self._transcribe_impl(chunk.tobytes(), 16000, None)

            # Keep remaining audio in buffer
            # 50% overlap
            self.audio_buffer = self.audio_buffer[required_samples // 2:]

            return result

        return None

    def flush_buffer(self) -> Optional[tuple[str, str]]:
        """Transcribe any remaining audio in buffer."""
        if len(self.audio_buffer) > 0:
            result = self._transcribe_impl(
                self.audio_buffer.tobytes(), 16000, None)
            self.audio_buffer = np.array([], dtype=np.float32)
            return result
        return None
