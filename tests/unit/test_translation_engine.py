"""Unit tests for translation engines."""

from unittest.mock import Mock, patch

import pytest

from live_translation.core.exceptions import TranslationError
from live_translation.core.models import LanguageCode, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


class TestSimpleTranslator:
    """Test SimpleTranslator implementation."""

    def test_translator_initialization(self) -> None:
        """Test translator initializes correctly."""
        translator = SimpleTranslator()
        assert translator.model_name == "mock-translator"
        assert translator.device == "cpu"
        assert not translator.is_loaded()

    def test_load_model(self) -> None:
        """Test model loading."""
        translator = SimpleTranslator()
        translator.load()
        assert translator.is_loaded()

    def test_translate_text(self) -> None:
        """Test text translation."""
        translator = SimpleTranslator()
        result = translator.translate("Hello", "en", "es")

        assert result == "[en→es] Hello"

    def test_detect_language(self) -> None:
        """Test language detection."""
        translator = SimpleTranslator()
        result = translator.detect_language("Hello world")

        assert result == "en"  # Mock always returns English


class TestTranslationPipeline:
    """Test TranslationPipeline integration."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline initializes correctly."""
        stt_engine = Mock()
        translation_engine = Mock()

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        assert pipeline.stt_engine == stt_engine
        assert pipeline.translation_engine == translation_engine

    def test_process_text_request(self) -> None:
        """Test processing text translation request."""
        # Mock translation engine
        translation_engine = Mock()
        translation_engine.translate.return_value = "¡Hola, mundo!"

        # Mock STT engine (not used for text requests)
        stt_engine = Mock()

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        request = TranslationRequest(
            text="Hello, world!",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        # Mock language detection
        with patch.object(pipeline, '_detect_language', return_value="en"):
            response = pipeline.process_request(request)

        assert response.original_text == "Hello, world!"
        assert response.translated_text == "¡Hola, mundo!"
        assert response.detected_language == "en"
        assert response.processing_time_ms > 0

        # Verify translation engine was called
        translation_engine.translate.assert_called_once_with(
            "Hello, world!", "en", LanguageCode.SPANISH
        )

    def test_process_audio_request(self) -> None:
        """Test processing audio translation request."""
        # Mock STT engine
        stt_engine = Mock()
        stt_engine.transcribe.return_value = ("Hello, world!", "en")

        # Mock translation engine
        translation_engine = Mock()
        translation_engine.translate.return_value = "¡Hola, mundo!"

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        audio_data = b"fake_audio_data"
        request = TranslationRequest(
            audio_data=audio_data,
            sample_rate=16000,
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.SPANISH,
        )

        response = pipeline.process_request(request)

        assert response.original_text == "Hello, world!"
        assert response.translated_text == "¡Hola, mundo!"
        assert response.detected_language == "en"
        assert response.processing_time_ms > 0

        # Verify STT engine was called
        stt_engine.transcribe.assert_called_once_with(
            audio_data, 16000, None
        )

        # Verify translation engine was called
        translation_engine.translate.assert_called_once_with(
            "Hello, world!", "en", LanguageCode.SPANISH
        )

    def test_process_request_no_input_error(self) -> None:
        """Test that request without text or audio raises error."""
        stt_engine = Mock()
        translation_engine = Mock()

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        # This should raise a validation error during request creation
        with pytest.raises(ValueError) as exc_info:
            TranslationRequest(
                source_language=LanguageCode.ENGLISH,
                target_language=LanguageCode.SPANISH,
            )

        assert "Either text or audio_data must be provided" in str(
            exc_info.value)

    def test_stt_engine_error_propagation(self) -> None:
        """Test that STT engine errors are properly propagated."""
        # Mock STT engine that raises an error
        stt_engine = Mock()
        stt_engine.transcribe.side_effect = Exception("STT failed")

        translation_engine = Mock()

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        request = TranslationRequest(
            audio_data=b"fake_audio",
            sample_rate=16000,
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.SPANISH,
        )

        with pytest.raises(Exception) as exc_info:
            pipeline.process_request(request)

        assert "STT failed" in str(exc_info.value)

    def test_translation_engine_error_propagation(self) -> None:
        """Test that translation engine errors are properly propagated."""
        stt_engine = Mock()

        # Mock translation engine that raises an error
        translation_engine = Mock()
        translation_engine.translate.side_effect = Exception(
            "Translation failed")

        pipeline = TranslationPipeline(
            stt_engine=stt_engine,
            translation_engine=translation_engine,
        )

        request = TranslationRequest(
            text="Hello, world!",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        with patch.object(pipeline, '_detect_language', return_value="en"):
            with pytest.raises(Exception) as exc_info:
                pipeline.process_request(request)

        assert "Translation failed" in str(exc_info.value)
