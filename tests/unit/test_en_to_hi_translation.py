"""Test English to Hindi translation functionality."""

from unittest.mock import patch

import pytest

from live_translation.core.models import LanguageCode, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


class TestEnglishToHindiTranslation:
    """Test English to Hindi translation workflows."""

    @pytest.fixture
    def simple_translator(self) -> SimpleTranslator:
        """Create a simple translator for testing."""
        translator = SimpleTranslator()
        translator.load()  # Ensure it's loaded
        return translator

    @pytest.fixture
    def translation_pipeline(
        self, simple_translator: SimpleTranslator
    ) -> TranslationPipeline:
        """Create a translation pipeline with mock engines."""
        pipeline = TranslationPipeline(
            stt_engine=simple_translator,
            translation_engine=simple_translator,
        )
        return pipeline

    def test_simple_translator_en_to_hi(
        self, simple_translator: SimpleTranslator
    ) -> None:
        """Test SimpleTranslator English to Hindi translation."""
        # Test basic translation
        result = simple_translator.translate("Hello, world!", "en", "hi")

        # The SimpleTranslator returns a mock format: [source→target] text
        assert result == "[en→hi] Hello, world!"
        assert isinstance(result, str)
        assert len(result) > 0

    def test_simple_translator_common_phrases_en_to_hi(
        self, simple_translator: SimpleTranslator
    ) -> None:
        """Test SimpleTranslator with common English phrases to Hindi."""
        test_phrases = [
            "Hello",
            "How are you?",
            "Thank you",
            "Good morning",
            "Welcome",
            "Nice to meet you",
            "Have a good day",
        ]

        for phrase in test_phrases:
            result = simple_translator.translate(phrase, "en", "hi")
            expected = f"[en→hi] {phrase}"
            assert result == expected
            assert isinstance(result, str)

    def test_translation_request_en_to_hi(self) -> None:
        """Test creating a translation request for English to Hindi."""
        request = TranslationRequest(
            text="Hello, how are you today?",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.HINDI,
        )

        assert request.text == "Hello, how are you today?"
        assert request.source_language == "en"
        assert request.target_language == "hi"
        assert request.audio_data is None

    def test_translation_pipeline_text_en_to_hi(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test translation pipeline with English to Hindi text translation."""
        request = TranslationRequest(
            text="Hello, world! How are you?",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.HINDI,
        )

        # Mock language detection to return English
        with patch.object(translation_pipeline, "_detect_language", return_value="en"):
            response = translation_pipeline.process_request(request)

        # Verify response structure
        assert response.original_text == "Hello, world! How are you?"
        assert response.translated_text == "[en→hi] Hello, world! How are you?"
        assert response.detected_language == "en"
        assert 0.0 <= response.confidence <= 1.0
        assert response.processing_time_ms >= 0.0
        assert response.model_info is not None

    def test_translation_pipeline_audio_en_to_hi(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test translation pipeline with English audio to Hindi translation."""
        # Create audio data (mock) - based on SimpleTranslator.transcribe logic:
        # < 1000 bytes -> "Hello"
        # < 5000 bytes -> "Hello, world!"
        # >= 5000 bytes -> "Hello, world! How are you today?"
        audio_data = (
            b"fake_english_audio_data_for_testing" * 150
        )  # Make it larger (>5000 bytes)

        request = TranslationRequest(
            audio_data=audio_data,
            sample_rate=16000,
            source_language=LanguageCode.AUTO,  # Auto-detect
            target_language=LanguageCode.HINDI,
        )

        response = translation_pipeline.process_request(request)

        # The SimpleTranslator mock will transcribe audio >= 5000 bytes as the longer text
        expected_transcription = "Hello, world! How are you today?"
        expected_translation = "[en→hi] Hello, world! How are you today?"

        assert response.original_text == expected_transcription
        assert response.translated_text == expected_translation
        # SimpleTranslator returns "en" for transcribed text
        assert response.detected_language == "en"
        assert 0.0 <= response.confidence <= 1.0
        assert response.processing_time_ms >= 0.0

    def test_language_code_validation_hindi(self) -> None:
        """Test that Hindi language code is properly validated."""
        # Test with LanguageCode enum
        request1 = TranslationRequest(
            text="Test",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.HINDI,
        )
        assert request1.target_language == "hi"

        # Test with string code
        request2 = TranslationRequest(
            text="Test",
            source_language="en",
            target_language="hi",
        )
        assert request2.target_language == "hi"

    def test_multiple_en_to_hi_translations(
        self, simple_translator: SimpleTranslator
    ) -> None:
        """Test multiple consecutive English to Hindi translations."""
        test_cases = [
            ("Hello", "hi"),
            ("Good morning", "hi"),
            ("Thank you very much", "hi"),
            ("How are you doing today?", "hi"),
            ("Welcome to our application", "hi"),
        ]

        for text, target_lang in test_cases:
            result = simple_translator.translate(text, "en", target_lang)
            expected = f"[en→{target_lang}] {text}"
            assert result == expected

    def test_hindi_language_support(self, simple_translator: SimpleTranslator) -> None:
        """Test that Hindi is in the list of supported languages."""
        # Check if the translator supports Hindi through its language mapping
        # For SimpleTranslator, we can verify it handles the "hi" code
        result = simple_translator.translate("Test", "en", "hi")
        assert "hi" in result  # Should contain the target language code

        # Test that it doesn't raise an error for Hindi language code
        try:
            simple_translator.translate("Test phrase", "en", "hi")
        except Exception as e:
            pytest.fail(
                f"SimpleTranslator should support Hindi translation, but got error: {e}"
            )

    def test_auto_detect_to_hindi(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test auto-detection of English and translation to Hindi."""
        request = TranslationRequest(
            text="Hello, this is an English sentence.",
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.HINDI,
        )

        # Mock language detection to return English
        with patch.object(translation_pipeline, "_detect_language", return_value="en"):
            response = translation_pipeline.process_request(request)

        assert response.original_text == "Hello, this is an English sentence."
        assert response.translated_text == "[en→hi] Hello, this is an English sentence."
        assert response.detected_language == "en"

    def test_edge_cases_en_to_hi(self, simple_translator: SimpleTranslator) -> None:
        """Test edge cases for English to Hindi translation."""
        # Empty string
        result = simple_translator.translate("", "en", "hi")
        assert result == "[en→hi] "

        # Single word
        result = simple_translator.translate("Hello", "en", "hi")
        assert result == "[en→hi] Hello"

        # Numbers and special characters
        result = simple_translator.translate("Hello 123! @#$", "en", "hi")
        assert result == "[en→hi] Hello 123! @#$"

        # Very long text
        long_text = "This is a very long sentence for testing purposes. " * 10
        result = simple_translator.translate(long_text, "en", "hi")
        assert result == f"[en→hi] {long_text}"
        assert len(result) > len(long_text)  # Should include the prefix

    def test_translation_request_validation_with_hindi(self) -> None:
        """Test that TranslationRequest properly validates Hindi language code."""
        # Valid request
        request = TranslationRequest(
            text="Test translation to Hindi",
            source_language="en",
            target_language="hi",
        )
        assert request.source_language == "en"
        assert request.target_language == "hi"

        # Invalid target language should raise error
        with pytest.raises(ValueError, match="Unsupported language code"):
            TranslationRequest(
                text="Test",
                source_language="en",
                target_language="invalid_lang",
            )

    def test_response_structure_en_to_hi(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test that the response structure is correct for English to Hindi translation."""
        request = TranslationRequest(
            text="Testing response structure",
            source_language="en",
            target_language="hi",
        )

        with patch.object(translation_pipeline, "_detect_language", return_value="en"):
            response = translation_pipeline.process_request(request)

        # Check all required fields are present
        assert hasattr(response, "original_text")
        assert hasattr(response, "translated_text")
        assert hasattr(response, "detected_language")
        assert hasattr(response, "confidence")
        assert hasattr(response, "processing_time_ms")
        assert hasattr(response, "timestamp")
        assert hasattr(response, "model_info")

        # Check types and values
        assert isinstance(response.original_text, str)
        assert isinstance(response.translated_text, str)
        assert isinstance(response.detected_language, str)
        assert isinstance(response.confidence, float)
        assert isinstance(response.processing_time_ms, float)
        assert response.processing_time_ms >= 0.0
        assert 0.0 <= response.confidence <= 1.0
