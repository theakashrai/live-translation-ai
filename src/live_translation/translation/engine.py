"""Translation engine interface and base classes."""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Tuple

from live_translation.core.exceptions import ModelLoadError, TranslationError
from live_translation.core.models import (
    LanguageCode,
    TranslationRequest,
    TranslationResponse,
)
from live_translation.utils.logger import get_logger, log_performance

logger = get_logger(__name__)


class TranslationEngine(Protocol):
    """Protocol for translation engines.

    This protocol defines the interface that all translation engines must implement
    to be compatible with the translation pipeline.
    """

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        """Translate text from source to target language.

        Args:
            text: The text to translate
            source_lang: Source language code (default: "auto" for auto-detection)
            target_lang: Target language code (default: "en")

        Returns:
            The translated text

        Raises:
            TranslationError: If translation fails
        """
        ...

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text.

        Args:
            text: The text to analyze

        Returns:
            The detected language code

        Raises:
            LanguageDetectionError: If language detection fails
        """
        ...

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready.

        Returns:
            True if the model is loaded and ready for use
        """
        ...


class SpeechToTextEngine(Protocol):
    """Protocol for speech-to-text engines.

    This protocol defines the interface that all speech-to-text engines must implement
    to be compatible with the translation pipeline.
    """

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Expected language code (None for auto-detection)

        Returns:
            Tuple of (transcribed_text, detected_language)

        Raises:
            TranslationError: If transcription fails
        """
        ...

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready.

        Returns:
            True if the model is loaded and ready for use
        """
        ...


class BaseModelEngine(ABC):
    """Common base class for model-based engines."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._model: Any = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model."""

    def load(self) -> None:
        """Load the model if not already loaded."""
        if self._loaded:
            return

        start_time = time.time()
        try:
            logger.info(f"Loading model: {self.model_name}")
            self._model = self._load_model()
            self._loaded = True
            load_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Model {self.model_name} loaded in {load_time_ms:.2f}ms")
            log_performance("model_load", load_time_ms, model=self.model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model {self.model_name}: {str(e)}",
                error_code="MODEL_LOAD_FAILED",
                details={"model_name": self.model_name, "device": self.device},
            ) from e


class BaseTranslationEngine(BaseModelEngine):
    """Base class for translation engines."""

    @abstractmethod
    def _translate_impl(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Implement the actual translation logic."""

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "en"
    ) -> str:
        """Translate text from source to target language."""
        if not self._loaded:
            self.load()

        start_time = time.time()
        try:
            result = self._translate_impl(text, source_lang, target_lang)
            processing_time_ms = (time.time() - start_time) * 1000
            log_performance(
                "translation",
                processing_time_ms,
                source_lang=source_lang,
                target_lang=target_lang,
                text_length=len(text),
            )
            return result
        except Exception as e:
            raise TranslationError(
                f"Translation failed: {str(e)}",
                error_code="TRANSLATION_FAILED",
                details={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_length": len(text),
                },
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded


class BaseSpeechToTextEngine(BaseModelEngine):
    """Base class for speech-to-text engines."""

    @abstractmethod
    def _transcribe_impl(
        self, audio_data: bytes, sample_rate: int, language: Optional[str]
    ) -> tuple[str, str]:
        """Implement the actual transcription logic."""

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> tuple[str, str]:
        """Transcribe audio to text."""
        if not self._loaded:
            self.load()

        start_time = time.time()
        try:
            text, detected_lang = self._transcribe_impl(
                audio_data, sample_rate, language
            )
            processing_time_ms = (time.time() - start_time) * 1000
            log_performance(
                "transcription",
                processing_time_ms,
                sample_rate=sample_rate,
                audio_size=len(audio_data),
                detected_language=detected_lang,
            )
            return text, detected_lang
        except Exception as e:
            raise TranslationError(
                f"Transcription failed: {str(e)}",
                error_code="TRANSCRIPTION_FAILED",
                details={
                    "sample_rate": sample_rate,
                    "audio_size": len(audio_data),
                    "language": language,
                },
            ) from e

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded


class TranslationPipeline:
    """Complete translation pipeline combining speech-to-text and translation."""

    def __init__(
        self,
        stt_engine: SpeechToTextEngine,
        translation_engine: TranslationEngine,
    ) -> None:
        self.stt_engine = stt_engine
        self.translation_engine = translation_engine
        self.logger = get_logger(self.__class__.__name__)

    def process_request(self, request: TranslationRequest) -> TranslationResponse:
        """Process a complete translation request."""
        start_time = time.time()

        try:
            # Handle text input
            if request.text:
                original_text = request.text
                detected_language = self._detect_language(original_text)
            # Handle audio input
            elif request.audio_data:
                original_text, detected_language = self.stt_engine.transcribe(
                    request.audio_data,
                    request.sample_rate,
                    request.source_language
                    if request.source_language != LanguageCode.AUTO
                    else None,
                )
            else:
                raise TranslationError("No text or audio data provided")

            # Perform translation
            source_lang = (
                detected_language
                if request.source_language == LanguageCode.AUTO
                else request.source_language
            )

            translated_text = self.translation_engine.translate(
                original_text, source_lang, request.target_language
            )

            processing_time_ms = (time.time() - start_time) * 1000

            return TranslationResponse(
                original_text=original_text,
                translated_text=translated_text,
                detected_language=detected_language,
                confidence=0.95,  # TODO: Implement actual confidence calculation
                processing_time_ms=processing_time_ms,
                model_info={
                    "stt_model": getattr(self.stt_engine, "model_name", "unknown"),
                    "translation_model": getattr(
                        self.translation_engine, "model_name", "unknown"
                    ),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Translation pipeline failed in {processing_time_ms:.2f}ms: {str(e)}"
            )
            raise

    def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            # Use translation engine's detect_language if available
            if hasattr(self.translation_engine, "detect_language"):
                return self.translation_engine.detect_language(text)

            # Fallback to simple language detection
            from langdetect import detect

            return detect(text)
        except Exception:
            # Default fallback
            return "en"
