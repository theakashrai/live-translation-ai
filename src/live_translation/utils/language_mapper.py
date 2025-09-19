"""Language code mapping utilities for translation models."""

from __future__ import annotations


class LanguageMapper:
    """Maps between different language code formats used by various translation models."""

    # Standard ISO 639-1 to human-readable names
    LANGUAGE_NAMES: dict[str, str] = {
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

    # NLLB-specific language codes
    NLLB_LANGUAGE_CODES: dict[str, str] = {
        "auto": "auto",  # Special case for auto-detection
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

    def __init__(self, model_name: str) -> None:
        """Initialize mapper for specific model.

        Args:
            model_name: Name of the translation model
        """
        self.model_name = model_name.lower()
        self.is_nllb_model = "nllb" in self.model_name

    def map_to_model_code(self, lang_code: str) -> str | None:
        """Map standard language code to model-specific format.

        Args:
            lang_code: Standard language code (e.g., 'en', 'es')

        Returns:
            Model-specific language code, or None for auto-detection
        """
        if lang_code == "auto":
            return None

        if self.is_nllb_model:
            return self.NLLB_LANGUAGE_CODES.get(lang_code, lang_code)

        return lang_code

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported language codes and their names.

        Returns:
            Dictionary mapping language codes to human-readable names
        """
        return self.LANGUAGE_NAMES.copy()

    def is_supported(self, lang_code: str) -> bool:
        """Check if language is supported.

        Args:
            lang_code: Language code to check

        Returns:
            True if language is supported
        """
        return lang_code in self.LANGUAGE_NAMES

    def get_default_target_code(self) -> str:
        """Get default target language code for the model.

        Returns:
            Default target language code
        """
        if self.is_nllb_model:
            return self.NLLB_LANGUAGE_CODES["en"]  # eng_Latn
        return "en"
