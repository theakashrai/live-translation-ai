"""Language detection utilities using standard libraries."""

from __future__ import annotations

from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class LanguageDetector:
    """Simple language detection using langdetect library."""

    def __init__(self, default_language: str = "en") -> None:
        """Initialize language detector.

        Args:
            default_language: Default language code to return on detection failure
        """
        self.default_language = default_language

    def detect(self, text: str) -> str:
        """Detect the language of the given text.

        Args:
            text: Text to analyze

        Returns:
            Two-letter language code (e.g., 'en', 'es', 'fr')
        """
        if not text or not text.strip():
            return self.default_language

        try:
            from langdetect import detect

            return detect(text.strip())
        except ImportError:
            logger.warning(
                "langdetect library not installed. Install with: pip install langdetect"
            )
            return self.default_language
        except Exception as e:
            logger.debug(f"Language detection failed for text '{text[:50]}...': {e}")
            return self.default_language

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """Detect language with confidence score.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or not text.strip():
            return self.default_language, 0.0

        try:
            from langdetect import detect_langs

            results = detect_langs(text.strip())
            if results:
                best_result = results[0]
                return best_result.lang, best_result.prob
        except ImportError:
            logger.warning(
                "langdetect library not installed. Install with: pip install langdetect"
            )
        except Exception as e:
            logger.debug(f"Language detection failed for text '{text[:50]}...': {e}")

        return self.default_language, 0.0
