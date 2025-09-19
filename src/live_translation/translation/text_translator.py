"""Text translation using Hugging Face transformers."""

from __future__ import annotations

from typing import Any

from transformers.pipelines import pipeline

from live_translation.core.config import settings
from live_translation.core.exceptions import ModelLoadError, TranslationError
from live_translation.translation.engine import BaseTranslationEngine
from live_translation.utils.device import DeviceManager
from live_translation.utils.language_detector import LanguageDetector
from live_translation.utils.language_mapper import LanguageMapper
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleTranslator(BaseTranslationEngine):
    """Simple mock translator for testing and development."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        model_name = model_name or "mock-translator"
        device = device or "cpu"
        super().__init__(model_name, device)
        self._language_detector = LanguageDetector()

    def _load_model(self) -> Any:
        """Mock model loading - returns None."""
        return None

    def _translate_impl(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        """Mock translation implementation."""
        return f"[{source_language}→{target_language}] {text}"

    def detect_language(self, text: str) -> str:
        """Mock language detection using predictable rules for testing."""
        # Use predictable mock detection for testing
        if not text or not text.strip():
            return "en"

        text_lower = text.lower()
        # Simple heuristic for testing
        if any(word in text_lower for word in ["hola", "sí", "gracias", "spanish"]):
            return "es"
        elif any(word in text_lower for word in ["bonjour", "merci", "oui", "french"]):
            return "fr"
        elif any(word in text_lower for word in ["guten", "danke", "ja", "german"]):
            return "de"
        else:
            return "en"  # Default to English for testing consistency

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> tuple[str, str]:
        """Mock transcription implementation for STT engine compatibility."""
        # Mock transcription that returns predictable text based on audio length
        audio_length = len(audio_data)
        if audio_length < 1000:
            text = "Hello"
        elif audio_length < 5000:
            text = "Hello, world!"
        else:
            text = "Hello, world! How are you today?"

        detected_lang = language or "en"
        return text, detected_lang

    def is_loaded(self) -> bool:
        """Mock implementation - always returns True after load()."""
        return self._loaded


class TransformersTranslator(BaseTranslationEngine):
    """Translation using Hugging Face transformers library with simplified design."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        model_name = model_name or settings.translation_model
        device = device or DeviceManager.get_optimal_device()
        super().__init__(model_name, device)

        self.pipeline: Any = None
        self._language_detector = LanguageDetector()
        self._language_mapper = LanguageMapper(model_name)

    def _load_model(self) -> Any:
        """Load the translation model using transformers pipeline."""
        try:
            logger.info(f"Loading translation model: {self.model_name}")

            # Use transformers built-in device handling with fallback
            device = -1 if self.device == "cpu" else self.device

            # Create pipeline - let transformers handle device management and fallbacks
            self.pipeline = pipeline(
                "translation",
                model=self.model_name,
                device=device,
                torch_dtype="auto",  # Let transformers choose optimal dtype
                model_kwargs={
                    "cache_dir": str(settings.model_cache_dir),
                    "device_map": "auto" if device != -1 else None,
                },
                # Standard generation parameters
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

            # Store actual device used by pipeline
            actual_device = getattr(self.pipeline.model, "device", self.device)
            logger.info(f"✅ Model loaded on {actual_device}")

            return self.pipeline.model

        except ImportError as e:
            raise ModelLoadError(
                "Transformers library not installed. Install with: pip install transformers torch",
                error_code="TRANSFORMERS_NOT_INSTALLED",
            ) from e
        except Exception as e:
            raise TranslationError(
                f"Failed to load translation model {self.model_name}: {str(e)}",
                error_code="TRANSLATION_MODEL_LOAD_FAILED",
                details={"model_name": self.model_name, "device": self.device},
            ) from e

    def _translate_impl(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        """Implement translation using the loaded pipeline."""
        try:
            # Handle auto-detection
            if source_language == "auto":
                source_language = self._language_detector.detect(text)
                logger.debug(f"Auto-detected source language: {source_language}")

            # Map language codes for model compatibility
            src_code = self._language_mapper.map_to_model_code(source_language)
            tgt_code = self._language_mapper.map_to_model_code(target_language)

            # Use pipeline for translation - it handles all the complexity
            if self._language_mapper.is_nllb_model:
                # For NLLB models, use the specific source/target language format
                result = self.pipeline(
                    text,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                )
            else:
                # For generic models, let pipeline handle language codes
                result = self.pipeline(
                    text,
                    src_lang=source_language if source_language != "auto" else None,
                    tgt_lang=target_language,
                )

            # Extract translation text from result
            if isinstance(result, list) and len(result) > 0:
                translation = result[0].get("translation_text", "")
            else:
                translation = str(result)

            return translation.strip()

        except Exception as e:
            raise TranslationError(
                f"Translation failed: {str(e)}",
                error_code="TRANSLATION_FAILED",
                details={
                    "source_lang": source_language,
                    "target_lang": target_language,
                    "model_name": self.model_name,
                    "text_preview": text[:50] + "..." if len(text) > 50 else text,
                },
            ) from e

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        return self._language_detector.detect(text)

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported language codes and names."""
        return self._language_mapper.get_supported_languages()
