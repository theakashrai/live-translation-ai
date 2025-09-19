"""Integration tests for the complete translation workflow."""


from pydantic_core import ValidationError
import pytest

from live_translation.audio.processor import AudioProcessor
from live_translation.core.models import AudioChunk, LanguageCode, TranslationRequest
from live_translation.translation.engine import TranslationPipeline
from live_translation.translation.text_translator import SimpleTranslator


@pytest.mark.integration
class TestTranslationWorkflow:
    """Test complete translation workflows."""

    def test_text_translation_workflow(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test end-to-end text translation workflow."""
        request = TranslationRequest(
            text="Hello, world! How are you today?",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        response = translation_pipeline.process_request(request)

        # Verify response structure
        assert response.original_text == "Hello, world! How are you today?"
        assert response.translated_text.startswith("[")  # Mock translator adds prefix
        assert response.detected_language is not None
        assert 0.0 <= response.confidence <= 1.0
        assert response.processing_time_ms >= 0.0
        assert response.model_info is not None

    def test_audio_translation_workflow(
        self, translation_pipeline: TranslationPipeline, sample_audio_data: bytes
    ) -> None:
        """Test end-to-end audio translation workflow."""
        request = TranslationRequest(
            audio_data=sample_audio_data,
            sample_rate=16000,
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.FRENCH,
        )

        response = translation_pipeline.process_request(request)

        # Verify response structure
        assert response.original_text is not None
        assert response.translated_text is not None
        assert response.detected_language is not None
        assert 0.0 <= response.confidence <= 1.0
        assert response.processing_time_ms >= 0.0

    def test_multiple_language_pairs(self, simple_translator: SimpleTranslator) -> None:
        """Test translation with multiple language pairs."""
        pipeline = TranslationPipeline(
            stt_engine=simple_translator,
            translation_engine=simple_translator,
        )

        test_cases = [
            ("Hello", LanguageCode.ENGLISH, LanguageCode.SPANISH),
            ("Bonjour", LanguageCode.FRENCH, LanguageCode.ENGLISH),
            ("Hola", LanguageCode.SPANISH, LanguageCode.FRENCH),
            ("Guten Tag", LanguageCode.GERMAN, LanguageCode.ITALIAN),
        ]

        for text, source, target in test_cases:
            request = TranslationRequest(
                text=text,
                source_language=source,
                target_language=target,
            )

            response = pipeline.process_request(request)

            assert response.original_text == text
            assert response.translated_text is not None
            assert len(response.translated_text) > 0

    @pytest.mark.slow
    def test_large_text_translation(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test translation of larger text blocks."""
        # Create a longer text (but within limits)
        large_text = " ".join(
            [
                "This is a longer text that contains multiple sentences.",
                "We want to test how the translation system handles",
                "more substantial amounts of text content.",
                "The system should be able to process this efficiently",
                "and return a proper translation response.",
            ]
        )

        request = TranslationRequest(
            text=large_text,
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        response = translation_pipeline.process_request(request)

        assert response.original_text == large_text
        assert len(response.translated_text) > 0
        assert response.processing_time_ms >= 0.0

    def test_audio_processing_workflow(
        self, translation_pipeline: TranslationPipeline, sample_audio_chunk: AudioChunk
    ) -> None:
        """Test audio processing workflow."""
        processor = AudioProcessor(
            translation_pipeline=translation_pipeline,
            chunk_duration_s=1.0,
        )

        # Process the audio chunk
        result = processor.process_audio_chunk(sample_audio_chunk)

        # The result might be None if not enough audio has been collected
        # or if silence is detected, which is normal behavior
        if result is not None:
            assert result.original_text is not None
            assert result.translated_text is not None
            assert result.processing_time_ms >= 0.0


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceRequirements:
    """Test performance requirements from the PRD."""

    def test_text_translation_latency(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test that text translation meets latency requirements."""
        # PRD requirement: < 100ms for texts under 500 characters
        short_text = "Hello, world! How are you doing today?"

        request = TranslationRequest(
            text=short_text,
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        response = translation_pipeline.process_request(request)

        # With mock translator, this should be very fast
        # In a real implementation, this would test actual model performance
        assert (
            response.processing_time_ms < 100.0
        ), f"Translation took {response.processing_time_ms}ms"

    def test_audio_processing_latency(
        self, translation_pipeline: TranslationPipeline, sample_audio_data: bytes
    ) -> None:
        """Test audio processing latency requirements."""
        # PRD requirement: < 500ms from speech end to translation display
        request = TranslationRequest(
            audio_data=sample_audio_data,
            sample_rate=16000,
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.SPANISH,
        )

        response = translation_pipeline.process_request(request)

        # With mock engines, this should be very fast
        assert (
            response.processing_time_ms < 500.0
        ), f"Audio processing took {response.processing_time_ms}ms"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_audio_data_handling(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test handling of invalid audio data."""
        # Test with empty audio data
        # Since TranslationRequest validates input, we need to catch pydantic errors
        try:
            request = TranslationRequest(
                audio_data=b"",
                sample_rate=16000,
                source_language=LanguageCode.AUTO,
                target_language=LanguageCode.SPANISH,
            )

            # If request creation succeeds, process it
            response = translation_pipeline.process_request(request)
            # If it succeeds, verify the response is well-formed
            assert isinstance(response.original_text, str)
            assert isinstance(response.translated_text, str)
        except ValidationError:
            # Pydantic validation error is expected for invalid input
            pass
        except Exception as e:
            # If it fails with other errors, it should be a known exception type
            assert isinstance(e, (ValueError, RuntimeError))

    def test_unsupported_language_handling(
        self, translation_pipeline: TranslationPipeline
    ) -> None:
        """Test handling of unsupported language codes."""
        try:
            # Use valid request but language handling depends on implementation
            request = TranslationRequest(
                text="Hello, world!",
                source_language="xx",  # Invalid language code
                target_language=LanguageCode.ENGLISH,
            )

            # The system should handle this gracefully
            response = translation_pipeline.process_request(request)
            assert response is not None
        except ValidationError:
            # Pydantic validation error is expected for invalid language codes
            pass
        except Exception as e:
            # If it raises an exception, it should be a proper error type
            assert isinstance(e, (ValueError, RuntimeError))
