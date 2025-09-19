"""XTTS voice cloning implementation."""

from __future__ import annotations

import logging
from pathlib import Path
import time

from TTS.api import TTS
import numpy as np
import soundfile as sf
import torch

from live_translation.voice_cloning.models import (
    VoiceCloningError,
    VoiceCloningRequest,
    VoiceCloningResponse,
)

logger = logging.getLogger(__name__)


class XTTSVoiceCloner:
    """XTTS-based voice cloning engine."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
    ) -> None:
        """Initialize XTTS voice cloner.

        Args:
            model_name: TTS model name to use
            device: Device to use for inference ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = self._detect_device() if device == "auto" else device
        self._model: TTS | None = None
        self._loaded = False
        self._reference_audio: bytes | None = None
        self._reference_sample_rate: int = 22050

        logger.info(
            f"Initialized XTTS cloner with model: {model_name}, device: {self.device}"
        )

    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded and self._model is not None

    def load_model(self) -> None:
        """Load the TTS model."""
        if self.is_loaded():
            return

        try:
            logger.info(f"Loading TTS model: {self.model_name}")
            start_time = time.time()

            # Temporarily patch torch.load to use weights_only=False for XTTS model loading
            # This is safe since XTTS is a trusted model from Coqui TTS
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = patched_load

            try:
                self._model = TTS(model_name=self.model_name, progress_bar=False).to(
                    self.device
                )
                self._loaded = True
            finally:
                # Restore original torch.load
                torch.load = original_load

            load_time = time.time() - start_time
            logger.info(f"TTS model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise VoiceCloningError(
                f"Failed to load TTS model: {e}", "MODEL_LOAD_FAILED"
            ) from e

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for voice cloning."""
        # XTTS v2 supported languages
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "pl",  # Polish
            "tr",  # Turkish
            "ru",  # Russian
            "nl",  # Dutch
            "cs",  # Czech
            "ar",  # Arabic
            "zh",  # Chinese
            "ja",  # Japanese
            "hu",  # Hungarian
            "ko",  # Korean
        ]

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported for voice cloning."""
        return language in self.get_supported_languages()

    def load_reference_voice(self, audio_path: str) -> bool:
        """Load reference voice from audio file.

        Args:
            audio_path: Path to reference audio file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not Path(audio_path).exists():
                logger.warning(f"Reference audio file not found: {audio_path}")
                return False

            # Read audio file
            audio_data, sample_rate = self._read_audio_file(audio_path)
            self._reference_audio = audio_data
            self._reference_sample_rate = sample_rate

            logger.info(f"Loaded reference voice from: {audio_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load reference voice: {e}")
            return False

    def _read_audio_file(self, file_path: str) -> tuple[bytes, int]:
        """Read audio file and return audio data and sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            VoiceCloningError: If file cannot be read
        """
        try:
            audio_data, sample_rate = sf.read(file_path, dtype="float32")

            # Handle case where audio_data is already bytes (for testing)
            if isinstance(audio_data, bytes):
                return audio_data, sample_rate

            # Convert numpy array audio data to bytes for storage
            audio_bytes = self._audio_to_bytes(audio_data, sample_rate)
            return audio_bytes, sample_rate

        except Exception as e:
            logger.error(f"Failed to read audio file {file_path}: {e}")
            raise VoiceCloningError(
                "AUDIO_READ_FAILED", f"Failed to read audio file: {e}"
            ) from e

    def _audio_to_bytes(
        self, audio_array: np.ndarray, sample_rate: int = 22050
    ) -> bytes:
        """Convert audio array to bytes.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Audio data as bytes
        """
        # Convert to int16 and then to bytes
        audio_int16 = (audio_array * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def clone_voice(self, request: VoiceCloningRequest) -> VoiceCloningResponse:
        """Clone voice for the given request.

        Args:
            request: Voice cloning request

        Returns:
            Voice cloning response with generated audio

        Raises:
            VoiceCloningError: If voice cloning fails
        """
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not self.is_loaded():
                self.load_model()

            # Check if language is supported
            if not self.is_language_supported(request.target_language):
                raise VoiceCloningError(
                    "UNSUPPORTED_LANGUAGE",
                    f"Language '{request.target_language}' is not supported",
                )

            # Load reference voice if provided
            if request.reference_audio_path:
                if not Path(request.reference_audio_path).exists():
                    raise VoiceCloningError(
                        "REFERENCE_AUDIO_NOT_FOUND",
                        f"Reference audio file not found: {request.reference_audio_path}",
                    )

                if not self.load_reference_voice(request.reference_audio_path):
                    raise VoiceCloningError(
                        "REFERENCE_AUDIO_LOAD_FAILED",
                        f"Failed to load reference audio: {request.reference_audio_path}",
                    )

            # Generate speech
            logger.info(f"Generating speech for text: {request.text[:50]}...")

            # Ensure model is loaded
            if self._model is None:
                raise VoiceCloningError(
                    "MODEL_NOT_LOADED", "XTTS model is not loaded. Call load() first."
                )

            # Use TTS to generate speech
            if request.reference_audio_path:
                # Use reference audio file path for voice cloning with XTTS
                logger.info(
                    f"Using reference audio path: {request.reference_audio_path}"
                )
                logger.info(
                    f"Reference audio exists: {Path(request.reference_audio_path).exists()}"
                )
                audio = self._model.tts(
                    text=request.text,
                    speaker_wav=request.reference_audio_path,
                    language=request.target_language,
                )
            else:
                # Use default voice
                audio = self._model.tts(
                    text=request.text,
                    language=request.target_language,
                )

            # Convert to bytes
            audio_bytes = self._audio_to_bytes(
                np.array(audio, dtype=np.float32), request.sample_rate
            )

            # Calculate duration
            duration_seconds = len(audio) / request.sample_rate

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Voice cloning completed in {processing_time:.1f}ms, "
                f"duration: {duration_seconds:.1f}s"
            )

            return VoiceCloningResponse(
                audio_data=audio_bytes,
                sample_rate=request.sample_rate,
                duration_seconds=duration_seconds,
                text=request.text,
                processing_time_ms=processing_time,
                reference_audio_path=request.reference_audio_path,
                model_info={
                    "model_name": self.model_name,
                    "device": self.device,
                    "language": request.target_language,
                },
            )

        except VoiceCloningError:
            raise
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise VoiceCloningError(
                "VOICE_CLONING_FAILED", f"Voice cloning failed: {e}"
            ) from e
