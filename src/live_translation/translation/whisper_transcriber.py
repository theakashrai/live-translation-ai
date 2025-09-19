"""Direct Whisper transcriber for audio data processing."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
import whisper

from live_translation.audio.whisper_utils import WhisperAudioProcessor
from live_translation.core.config import settings
from live_translation.core.exceptions import ModelLoadError, TranslationError
from live_translation.utils.device import DeviceManager
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """Clean Whisper transcriber for direct audio data processing."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the Whisper transcriber.

        Args:
            model_name: Whisper model name
            device: Device to use
        """
        self.model_name = model_name or settings.whisper_model
        self.device = device or DeviceManager.get_optimal_device()
        self._model: Any | None = None
        self._device_manager = DeviceManager()
        self._audio_processor = WhisperAudioProcessor()

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> Document:
        """Transcribe audio data and return as Document.

        Args:
            audio_data: Audio data as bytes
            sample_rate: Sample rate of audio data
            language: Language code for transcription

        Returns:
            Document with transcription and metadata
        """
        if self._model is None:
            self._load_model()

        assert self._model is not None  # Type narrowing for mypy

        try:
            # Process audio data
            audio_array = self._audio_processor.load_from_bytes(audio_data, sample_rate)

            # Get transcription options
            options = self._device_manager.get_transcription_options(
                self.device, language
            )

            # Transcribe
            result = self._model.transcribe(audio_array, **options)

            text = result.get("text", "").strip()
            detected_language = result.get("language", language or "en")

            logger.debug(f"Transcribed: '{text[:50]}...' (lang: {detected_language})")

            return Document(
                page_content=text,
                metadata={
                    "language": detected_language,
                    "model": self.model_name,
                    "device": self.device,
                    "sample_rate": sample_rate,
                    "audio_size": len(audio_data),
                    "segments": len(result.get("segments", [])),
                },
            )

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

    def _load_model(self) -> None:
        """Load Whisper model with device fallback."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")

            device = self.device
            if device == "auto":
                device = DeviceManager.get_optimal_device()

            try:
                # Try primary device
                self._model = whisper.load_model(
                    self.model_name,
                    device=device,
                    download_root=str(settings.model_cache_dir),
                )
                self.device = device

            except Exception as e:
                if device != "cpu":
                    logger.warning(
                        f"Failed to load on {device}, falling back to CPU: {e}"
                    )
                    self._model = whisper.load_model(
                        self.model_name,
                        device="cpu",
                        download_root=str(settings.model_cache_dir),
                    )
                    self.device = "cpu"
                else:
                    raise

            logger.info(f"✅ Whisper model loaded successfully on {self.device}")

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model {self.model_name}: {str(e)}",
                error_code="WHISPER_LOAD_FAILED",
                details={"model_name": self.model_name, "device": self.device},
            ) from e
