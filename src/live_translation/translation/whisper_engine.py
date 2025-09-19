"""Whisper adapter implementing the SpeechToTextEngine protocol."""

from __future__ import annotations

import numpy as np

from live_translation.core.exceptions import TranslationError
from live_translation.translation.whisper_transcriber import WhisperTranscriber
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperAdapter:
    """Adapter that implements SpeechToTextEngine protocol for Whisper transcription."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the Whisper adapter.

        Args:
            model_name: Whisper model name
            device: Device to use
        """
        self._transcriber = WhisperTranscriber(model_name, device)
        self._loaded = False

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe audio to text following SpeechToTextEngine protocol.

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Expected language code (None for auto-detection)

        Returns:
            Tuple of (transcribed_text, detected_language)

        Raises:
            TranslationError: If transcription fails
        """
        try:
            # Use the underlying transcriber
            document = self._transcriber.transcribe(audio_data, sample_rate, language)

            # Mark as loaded after first successful transcription
            self._loaded = True

            # Extract text and language from document
            text = document.page_content
            detected_language = document.metadata.get("language", language or "en")

            return text, detected_language

        except Exception as e:
            raise TranslationError(
                f"Whisper transcription failed: {str(e)}",
                error_code="WHISPER_ADAPTER_TRANSCRIPTION_FAILED",
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready.

        Returns:
            True if the model is loaded and ready for use
        """
        return self._loaded

    def load(self) -> None:
        """Pre-load the model (optional - model loads on first transcribe call)."""
        try:
            # Force model loading by doing a dummy transcription with empty audio
            # 0.1 seconds of silence
            dummy_audio = np.zeros(1600, dtype=np.int16).tobytes()
            self.transcribe(dummy_audio, 16000)
            logger.info("✅ Whisper model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Model pre-loading failed, will load on first use: {e}")
            # Don't raise - model will load on first real transcription call
