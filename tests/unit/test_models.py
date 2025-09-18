"""Unit tests for core models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from live_translation.core.models import (
    AudioChunk,
    AudioFormat,
    JobStatus,
    LanguageCode,
    TranslationRequest,
    TranslationResponse,
    TranslationJob,
    SystemStatus,
)


class TestLanguageCode:
    """Test LanguageCode enum."""

    def test_is_valid(self) -> None:
        """Test language code validation."""
        assert LanguageCode.is_valid("en")
        assert LanguageCode.is_valid("auto")
        assert not LanguageCode.is_valid("invalid")

    def test_get_supported_codes(self) -> None:
        """Test getting supported codes."""
        codes = LanguageCode.get_supported_codes()
        assert "en" in codes
        assert "auto" in codes
        assert len(codes) > 10


class TestTranslationRequest:
    """Test TranslationRequest model."""

    def test_valid_text_request(self) -> None:
        """Test creating a valid text translation request."""
        request = TranslationRequest(
            text="Hello, world!",
            source_language=LanguageCode.ENGLISH,
            target_language=LanguageCode.SPANISH,
        )

        assert request.text == "Hello, world!"
        assert request.source_language == "en"  # Now returns string value
        assert request.target_language == "es"
        assert request.audio_data is None
        assert request.request_id is not None

    def test_valid_audio_request(self) -> None:
        """Test creating a valid audio translation request."""
        audio_data = b"fake_audio_data"
        request = TranslationRequest(
            audio_data=audio_data,
            sample_rate=16000,
            source_language=LanguageCode.AUTO,
            target_language=LanguageCode.FRENCH,
        )

        assert request.audio_data == audio_data
        assert request.sample_rate == 16000
        assert request.source_language == "auto"
        assert request.target_language == "fr"
        assert request.text is None

    def test_no_input_validation_error(self) -> None:
        """Test that request without text or audio raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TranslationRequest(
                source_language=LanguageCode.ENGLISH,
                target_language=LanguageCode.SPANISH,
            )

        assert "Either text or audio_data must be provided" in str(
            exc_info.value)

    def test_text_too_long_validation_error(self) -> None:
        """Test that text longer than 5000 characters raises validation error."""
        long_text = "a" * 5001

        with pytest.raises(ValidationError) as exc_info:
            TranslationRequest(
                text=long_text,
                source_language=LanguageCode.ENGLISH,
                target_language=LanguageCode.SPANISH,
            )

        assert "at most 5000 characters" in str(exc_info.value)

    def test_invalid_language_code(self) -> None:
        """Test invalid language code validation."""
        with pytest.raises(ValidationError) as exc_info:
            TranslationRequest(
                text="Hello",
                source_language="invalid_lang",
                target_language=LanguageCode.SPANISH,
            )

        assert "Unsupported language code" in str(exc_info.value)

    def test_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises validation error."""
        with pytest.raises(ValidationError):
            TranslationRequest(
                audio_data=b"fake_audio",
                sample_rate=5000,  # Below minimum of 8000
            )

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        request = TranslationRequest(text="Hello")

        assert request.source_language == "auto"
        assert request.target_language == "en"
        assert request.sample_rate == 16000
        assert request.audio_format == AudioFormat.WAV
        assert isinstance(request.timestamp, datetime)
        assert request.request_id is not None


class TestTranslationResponse:
    """Test TranslationResponse model."""

    def test_valid_response(self) -> None:
        """Test creating a valid translation response."""
        response = TranslationResponse(
            original_text="Hello, world!",
            translated_text="¡Hola, mundo!",
            detected_language="en",
            confidence=0.95,
            processing_time_ms=150.5,
        )

        assert response.original_text == "Hello, world!"
        assert response.translated_text == "¡Hola, mundo!"
        assert response.detected_language == "en"
        assert response.confidence == 0.95
        assert response.processing_time_ms == 150.5
        assert isinstance(response.timestamp, datetime)

    def test_confidence_validation(self) -> None:
        """Test confidence score validation."""
        # Test valid confidence
        response = TranslationResponse(
            original_text="Hello",
            translated_text="Hola",
            confidence=0.5,
            processing_time_ms=100.0,
        )
        assert response.confidence == 0.5

        # Test invalid confidence (too high)
        with pytest.raises(ValidationError):
            TranslationResponse(
                original_text="Hello",
                translated_text="Hola",
                confidence=1.5,
                processing_time_ms=100.0,
            )

        # Test invalid confidence (negative)
        with pytest.raises(ValidationError):
            TranslationResponse(
                original_text="Hello",
                translated_text="Hola",
                confidence=-0.1,
                processing_time_ms=100.0,
            )

    def test_processing_time_validation(self) -> None:
        """Test processing time validation."""
        # Valid processing time
        response = TranslationResponse(
            original_text="Hello",
            translated_text="Hola",
            confidence=0.9,
            processing_time_ms=0.0,
        )
        assert response.processing_time_ms == 0.0

        # Invalid processing time (negative)
        with pytest.raises(ValidationError):
            TranslationResponse(
                original_text="Hello",
                translated_text="Hola",
                confidence=0.9,
                processing_time_ms=-1.0,
            )


class TestAudioChunk:
    """Test AudioChunk model."""

    def test_valid_audio_chunk(self) -> None:
        """Test creating a valid audio chunk."""
        audio_data = b"fake_audio_data"
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            channels=1,
            duration_ms=1000.0,
            sequence_id=1,
        )

        assert chunk.data == audio_data
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.duration_ms == 1000.0
        assert chunk.sequence_id == 1
        assert isinstance(chunk.timestamp, datetime)

    def test_invalid_sample_rate(self) -> None:
        """Test that invalid sample rates raise validation errors."""
        audio_data = b"fake_audio_data"

        # Too low
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=7999,
                duration_ms=1000.0,
                sequence_id=1,
            )

        # Too high
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=48001,
                duration_ms=1000.0,
                sequence_id=1,
            )

    def test_invalid_channels(self) -> None:
        """Test that invalid channel counts raise validation errors."""
        audio_data = b"fake_audio_data"

        # Too few channels
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=16000,
                channels=0,
                duration_ms=1000.0,
                sequence_id=1,
            )

        # Too many channels
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=16000,
                channels=3,
                duration_ms=1000.0,
                sequence_id=1,
            )

    def test_negative_duration(self) -> None:
        """Test that negative duration raises validation error."""
        with pytest.raises(ValidationError):
            AudioChunk(
                data=b"fake_audio",
                sample_rate=16000,
                duration_ms=-1.0,
                sequence_id=1,
            )

    def test_negative_sequence_id(self) -> None:
        """Test that negative sequence ID raises validation error."""
        with pytest.raises(ValidationError):
            AudioChunk(
                data=b"fake_audio",
                sample_rate=16000,
                duration_ms=1000.0,
                sequence_id=-1,
            )


class TestTranslationJob:
    """Test TranslationJob model."""

    def test_job_creation(self) -> None:
        """Test creating a translation job."""
        request = TranslationRequest(text="Hello")
        job = TranslationJob(request=request)

        assert job.job_id is not None
        assert job.request == request
        assert job.status == JobStatus.PENDING
        assert job.response is None
        assert job.error_message is None

    def test_job_state_transitions(self) -> None:
        """Test job state transitions."""
        request = TranslationRequest(text="Hello")
        job = TranslationJob(request=request)

        # Mark as processing
        job.mark_processing()
        assert job.status == JobStatus.PROCESSING

        # Mark as completed
        response = TranslationResponse(
            original_text="Hello",
            translated_text="Hola",
            processing_time_ms=100.0,
        )
        job.mark_completed(response)
        assert job.status == JobStatus.COMPLETED
        assert job.response == response

        # Test failure case with new job
        job2 = TranslationJob(request=request)
        job2.mark_failed("Test error")
        assert job2.status == JobStatus.FAILED
        assert job2.error_message == "Test error"


class TestAudioChunk:
    """Test AudioChunk model."""

    def test_valid_audio_chunk(self) -> None:
        """Test creating a valid audio chunk."""
        audio_data = b"fake_audio_data"
        chunk = AudioChunk(
            data=audio_data,
            sample_rate=16000,
            channels=1,
            duration_ms=1000.0,
            sequence_id=1,
        )

        assert chunk.data == audio_data
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.duration_ms == 1000.0
        assert chunk.sequence_id == 1
        assert isinstance(chunk.timestamp, datetime)

    def test_audio_chunk_properties(self) -> None:
        """Test audio chunk computed properties."""
        chunk = AudioChunk(
            data=b"fake_audio_data" * 100,  # Make it bigger
            sample_rate=16000,
            duration_ms=1000.0,
            sequence_id=1,
        )

        assert chunk.duration_seconds == 1.0
        assert chunk.size_kb > 0

    def test_invalid_sample_rate(self) -> None:
        """Test that invalid sample rates raise validation errors."""
        audio_data = b"fake_audio_data"

        # Too low
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=7999,
                duration_ms=1000.0,
                sequence_id=1,
            )

        # Too high
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=48001,
                duration_ms=1000.0,
                sequence_id=1,
            )

    def test_invalid_channels(self) -> None:
        """Test that invalid channel counts raise validation errors."""
        audio_data = b"fake_audio_data"

        # Too few channels
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=16000,
                channels=0,
                duration_ms=1000.0,
                sequence_id=1,
            )

        # Too many channels
        with pytest.raises(ValidationError):
            AudioChunk(
                data=audio_data,
                sample_rate=16000,
                channels=3,
                duration_ms=1000.0,
                sequence_id=1,
            )

    def test_negative_duration(self) -> None:
        """Test that negative duration raises validation error."""
        with pytest.raises(ValidationError):
            AudioChunk(
                data=b"fake_audio",
                sample_rate=16000,
                duration_ms=-1.0,
                sequence_id=1,
            )

    def test_negative_sequence_id(self) -> None:
        """Test that negative sequence ID raises validation error."""
        with pytest.raises(ValidationError):
            AudioChunk(
                data=b"fake_audio",
                sample_rate=16000,
                duration_ms=1000.0,
                sequence_id=-1,
            )
