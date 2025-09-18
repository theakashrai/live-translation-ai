"""Text translation using Hugging Face transformers."""

from typing import Any, Dict, Optional

from live_translation.core.config import settings
from live_translation.core.exceptions import ModelLoadError, TranslationError
from live_translation.translation.engine import BaseTranslationEngine
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleTranslator(BaseTranslationEngine):
    """Simple mock translator for testing and development."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        model_name = model_name or "mock-translator"
        device = device or "cpu"
        super().__init__(model_name, device)

    def _load_model(self) -> Any:
        """Mock model loading - returns None."""
        return None

    def _translate_impl(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Mock translation implementation."""
        return f"[{source_lang}→{target_lang}] {text}"

    def detect_language(self, text: str) -> str:
        """Mock language detection."""
        # Simple heuristic for testing
        if any(char in text.lower() for char in ['hola', 'sí', 'gracias']):
            return "es"
        elif any(char in text.lower() for char in ['bonjour', 'merci', 'oui']):
            return "fr"
        elif any(char in text.lower() for char in ['guten', 'danke', 'ja']):
            return "de"
        else:
            return "en"

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
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
    """Translation using Hugging Face transformers library."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        model_name = model_name or settings.translation_model
        device = device or settings.device
        super().__init__(model_name, device)
        self.tokenizer: Any = None
        self.pipeline: Any = None
        self._mps_fallback_attempted = False  # Track fallback attempts

        # Language code mappings for NLLB
        self.lang_code_map = {
            "auto": None,  # Auto-detection handled separately
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ar": "arb_Arab",
            "hi": "hin_Deva",
        }

    def _load_model(self) -> Any:
        """Load the translation model and tokenizer."""
        try:
            import torch
            from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                      pipeline)

            logger.info(f"Loading translation model: {self.model_name}")

            # Determine device with MPS fallback
            device = -1  # Default to CPU
            if self.device == "cuda" and torch.cuda.is_available():
                device = 0  # Use first GPU
            elif self.device == "mps" and torch.backends.mps.is_available():
                try:
                    device = "mps"
                    logger.info(
                        "🍎 Attempting to use MPS for translation model")
                except Exception:
                    logger.warning("❌ MPS not fully supported, using CPU")
                    device = -1
            else:
                device = -1  # CPU

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(settings.model_cache_dir),
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=str(settings.model_cache_dir),
            )

            # Try creating pipeline with MPS, fallback to CPU if it fails
            try:
                self.pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                )
                if device == "mps":
                    logger.info(
                        "✅ MPS acceleration enabled for translation model")
            except Exception as mps_error:
                if device == "mps":
                    logger.warning(
                        f"❌ MPS pipeline failed, falling back to CPU: {str(mps_error)}")
                    device = -1
                    self.pipeline = pipeline(
                        "translation",
                        model=model,
                        tokenizer=self.tokenizer,
                        device=device,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                    )

            return model

        except ImportError as e:
            raise ModelLoadError(
                "Transformers library not installed. Install with: pip install transformers torch",
                error_code="TRANSFORMERS_NOT_INSTALLED",
            ) from e
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load translation model {self.model_name}: {str(e)}",
                error_code="TRANSLATION_MODEL_LOAD_FAILED",
                details={"model_name": self.model_name, "device": self.device},
            ) from e

    def _translate_impl(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Implement the actual translation logic."""
        try:
            # Map language codes to model-specific format
            src_code = self._map_language_code(source_lang)
            tgt_code = self._map_language_code(target_lang)

            # Handle NLLB model format
            if "nllb" in self.model_name.lower():
                return self._translate_nllb(text, src_code, tgt_code)
            else:
                # Generic transformer translation
                return self._translate_generic(text, source_lang, target_lang)

        except Exception as e:
            raise TranslationError(
                f"Translation implementation failed: {str(e)}",
                error_code="TRANSLATION_IMPL_FAILED",
                details={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "model_name": self.model_name,
                },
            ) from e

    def _translate_nllb(self, text: str, src_code: Optional[str], tgt_code: Optional[str]) -> str:
        """Translate using NLLB model format."""
        try:
            # NLLB expects specific format
            if src_code is None:
                # Auto-detect source language
                src_code = self._detect_language_nllb(text)

            if tgt_code is None:
                tgt_code = "eng_Latn"  # Default to English

            # Prepare input with language codes
            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True)

            # Get target language token ID - handle different tokenizer versions
            target_token_id = None
            if hasattr(self.tokenizer, 'lang_code_to_id') and self.tokenizer.lang_code_to_id:
                target_token_id = self.tokenizer.lang_code_to_id.get(tgt_code)
            elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                # Alternative approach for newer transformers versions
                target_token_id = self.tokenizer.convert_tokens_to_ids(
                    tgt_code)

            # Generate translation
            generation_kwargs = {
                **inputs,
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True,
            }

            if target_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = target_token_id

            translated_tokens = self._model.generate(**generation_kwargs)

            # Decode result
            result = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            return result.strip()

        except RuntimeError as e:
            if ("MPS" in str(e) or "Placeholder storage has not been allocated" in str(e)) and not self._mps_fallback_attempted:
                # MPS error - try to recreate pipeline with CPU fallback (only once)
                logger.warning(
                    f"❌ MPS inference failed, falling back to CPU: {str(e)[:100]}...")
                self._mps_fallback_attempted = True
                try:
                    # Move model and tokenizer to CPU explicitly
                    import torch
                    self._model = self._model.to('cpu')

                    # Recreate pipeline with CPU
                    from transformers import pipeline
                    self.pipeline = pipeline(
                        "translation",
                        model=self._model,
                        tokenizer=self.tokenizer,
                        device=-1,  # Force CPU
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                    )
                    logger.info(
                        "✅ Successfully switched to CPU for translation")

                    # Re-prepare inputs on CPU
                    self.tokenizer.src_lang = src_code
                    inputs_cpu = self.tokenizer(
                        text, return_tensors="pt", max_length=512, truncation=True)

                    # Move input tensors to CPU explicitly
                    inputs_cpu = {k: v.to('cpu') if hasattr(
                        v, 'to') else v for k, v in inputs_cpu.items()}

                    # Generate on CPU
                    generation_kwargs = {
                        **inputs_cpu,
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True,
                    }

                    if target_token_id is not None:
                        generation_kwargs["forced_bos_token_id"] = target_token_id

                    translated_tokens = self._model.generate(
                        **generation_kwargs)

                    # Decode result
                    result = self.tokenizer.batch_decode(
                        translated_tokens, skip_special_tokens=True
                    )[0]

                    return result.strip()

                except Exception as fallback_error:
                    raise TranslationError(
                        f"NLLB translation failed even with CPU fallback: {str(fallback_error)}") from fallback_error
            else:
                # Either not an MPS error, or we already attempted fallback
                error_msg = f"NLLB translation failed: {str(e)}"
                if self._mps_fallback_attempted:
                    error_msg += " (CPU fallback was already attempted)"
                raise TranslationError(error_msg) from e
        except Exception as e:
            raise TranslationError(f"NLLB translation failed: {str(e)}") from e

    def _translate_generic(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generic transformer translation."""
        try:
            # Use pipeline for generic models
            result = self.pipeline(
                text,
                src_lang=source_lang if source_lang != "auto" else None,
                tgt_lang=target_lang,
            )

            if isinstance(result, list) and len(result) > 0:
                return result[0].get("translation_text", "").strip()

            return str(result).strip()

        except Exception as e:
            raise TranslationError(
                f"Generic translation failed: {str(e)}") from e

    def _map_language_code(self, lang_code: str) -> Optional[str]:
        """Map general language code to model-specific code."""
        if "nllb" in self.model_name.lower():
            return self.lang_code_map.get(lang_code, lang_code)
        return lang_code

    def _detect_language_nllb(self, text: str) -> str:
        """Detect language for NLLB model."""
        try:
            # Simple language detection fallback
            from langdetect import detect
            detected = detect(text)
            return self.lang_code_map.get(detected, "eng_Latn")
        except Exception:
            # Default to English if detection fails
            return "eng_Latn"

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return "en"  # Default fallback

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names."""
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
        }
