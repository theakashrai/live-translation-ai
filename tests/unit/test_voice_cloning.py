"""Unit tests for voice cloning functionality."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from live_translation.core.models import TranslationRequest, TranslationResponse
from live_translation.voice_cloning import (
    TranslationWithVoiceCloning,
    VoiceCloningRequest,
    VoiceCloningResponse,
    VoiceCloningTranslationPipeline,
    XTTSVoiceCloner,
)
from live_translation.voice_cloning.xtts_cloner import VoiceCloningError


class TestVoiceCloningModels:
    """Test voice cloning data models."""

    def test_voice_cloning_request_creation(self):
        """Test VoiceCloningRequest model creation."""
        request = VoiceCloningRequest(
            text="Hello world",
            reference_audio_path="/path/to/audio.wav",
            target_language="en",
        )

        assert request.text == "Hello world"
        assert request.reference_audio_path == "/path/to/audio.wav"
        assert request.target_language == "en"
        assert request.sample_rate == 22050  # default
        assert request.speed == 1.0  # default
        assert request.temperature == 0.75  # default

    def test_voice_cloning_response_creation(self):
        """Test VoiceCloningResponse model creation."""
        response = VoiceCloningResponse(
            audio_data=b"fake_audio_data",
            sample_rate=22050,
            duration_seconds=2.5,
            text="Hello world",
            processing_time_ms=150.0,
            reference_audio_path="/path/to/audio.wav",
        )

        assert response.audio_data == b"fake_audio_data"
        assert response.sample_rate == 22050
        assert response.duration_seconds == 2.5
        assert response.text == "Hello world"
        assert response.processing_time_ms == 150.0
        assert response.reference_audio_path == "/path/to/audio.wav"


class TestXTTSVoiceCloner:
    """Test XTTS voice cloning engine."""

    @pytest.fixture
    def mock_tts(self):
        """Mock TTS class."""
        with patch("live_translation.voice_cloning.xtts_cloner.TTS") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake_audio_content")
            temp_path = f.name

        yield temp_path

        # Clean up
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    def test_xtts_cloner_initialization(self):
        """Test XTTS cloner initialization."""
        cloner = XTTSVoiceCloner(
            model_name="custom_model",
            device="cpu",
        )

        assert cloner.model_name == "custom_model"
        assert cloner.device == "cpu"
        assert not cloner.is_loaded()

    def test_xtts_cloner_default_initialization(self):
        """Test XTTS cloner with default settings."""
        cloner = XTTSVoiceCloner()

        assert cloner.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert cloner.device in ["auto", "cpu", "cuda", "mps"]  # auto-detected

    def test_supported_languages(self):
        """Test getting supported languages."""
        cloner = XTTSVoiceCloner()
        languages = cloner.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages

    def test_language_support_check(self):
        """Test language support checking."""
        cloner = XTTSVoiceCloner()

        assert cloner.is_language_supported("en")
        assert cloner.is_language_supported("es")
        assert not cloner.is_language_supported("xyz")

    def test_reference_voice_validation_missing_file(self):
        """Test reference voice validation with missing file."""
        cloner = XTTSVoiceCloner()

        result = cloner.load_reference_voice("/nonexistent/path.wav")
        assert not result

    def test_reference_voice_validation_existing_file(self, temp_audio_file):
        """Test reference voice validation with existing file."""
        cloner = XTTSVoiceCloner()

        with patch.object(
            cloner, "_read_audio_file", return_value=(b"fake_data", 22050)
        ):
            result = cloner.load_reference_voice(temp_audio_file)
            assert result

    @patch("live_translation.voice_cloning.xtts_cloner.sf.read")
    def test_read_audio_file_success(self, mock_sf_read):
        """Test successful audio file reading."""
        mock_sf_read.return_value = (b"audio_data", 22050)

        cloner = XTTSVoiceCloner()
        audio_data, sample_rate = cloner._read_audio_file("/fake/path.wav")

        assert audio_data == b"audio_data"
        assert sample_rate == 22050
        mock_sf_read.assert_called_once_with("/fake/path.wav", dtype="float32")

    @patch("live_translation.voice_cloning.xtts_cloner.sf.read")
    def test_read_audio_file_failure(self, mock_sf_read):
        """Test audio file reading failure."""
        mock_sf_read.side_effect = Exception("File not found")

        cloner = XTTSVoiceCloner()

        with pytest.raises(VoiceCloningError) as exc_info:
            cloner._read_audio_file("/fake/path.wav")

        assert "AUDIO_READ_FAILED" in str(exc_info.value)

    def test_audio_to_bytes_conversion(self):
        """Test audio array to bytes conversion."""
        cloner = XTTSVoiceCloner()
        audio_array = np.array([0.5, -0.3, 0.1], dtype=np.float32)

        result = cloner._audio_to_bytes(audio_array)

        assert isinstance(result, bytes)
        assert len(result) == 6  # 3 samples * 2 bytes each (int16)

    def test_clone_voice_missing_reference_file(self, mock_tts):
        """Test voice cloning with missing reference file."""
        cloner = XTTSVoiceCloner()
        cloner._loaded = True
        cloner._model = mock_tts

        request = VoiceCloningRequest(
            text="Hello world",
            reference_audio_path="/nonexistent/path.wav",
            target_language="en",
        )

        with pytest.raises(VoiceCloningError) as exc_info:
            cloner.clone_voice(request)

        assert "REFERENCE_AUDIO_NOT_FOUND" in str(exc_info.value)


class TestTranslationWithVoiceCloning:
    """Test TranslationWithVoiceCloning wrapper."""

    @pytest.fixture
    def mock_translation_response(self):
        """Create a mock translation response."""
        return TranslationResponse(
            original_text="Hello",
            translated_text="Hola",
            detected_language="en",
            processing_time_ms=100.0,
        )

    @pytest.fixture
    def mock_voice_cloning_response(self):
        """Create a mock voice cloning response."""
        return VoiceCloningResponse(
            audio_data=b"fake_audio",
            sample_rate=22050,
            duration_seconds=1.5,
            text="Hola",
            processing_time_ms=200.0,
            reference_audio_path="/path/to/ref.wav",
        )

    def test_translation_with_voice_cloning_creation(self, mock_translation_response):
        """Test TranslationWithVoiceCloning creation."""
        wrapper = TranslationWithVoiceCloning(
            translation_response=mock_translation_response
        )

        assert wrapper.original_text == "Hello"
        assert wrapper.translated_text == "Hola"
        assert wrapper.detected_language == "en"
        assert not wrapper.has_voice_cloning
        assert wrapper.cloned_audio_data is None

    def test_translation_with_voice_cloning_with_audio(
        self, mock_translation_response, mock_voice_cloning_response
    ):
        """Test TranslationWithVoiceCloning with voice cloning."""
        wrapper = TranslationWithVoiceCloning(
            translation_response=mock_translation_response,
            voice_cloning_response=mock_voice_cloning_response,
        )

        assert wrapper.has_voice_cloning
        assert wrapper.cloned_audio_data == b"fake_audio"
        assert wrapper.total_processing_time_ms == 300.0  # 100 + 200


class TestVoiceCloningTranslationPipeline:
    """Test VoiceCloningTranslationPipeline integration."""

    @pytest.fixture
    def mock_translation_pipeline(self):
        """Create a mock translation pipeline."""
        mock_pipeline = Mock()
        mock_response = TranslationResponse(
            original_text="Hello",
            translated_text="Hola",
            detected_language="en",
            processing_time_ms=100.0,
        )
        mock_pipeline.process_request.return_value = mock_response
        return mock_pipeline

    @pytest.fixture
    def mock_voice_engine(self):
        """Create a mock voice cloning engine."""
        mock_engine = Mock()
        mock_engine.is_loaded.return_value = True
        mock_engine.load_reference_voice.return_value = True

        mock_response = VoiceCloningResponse(
            audio_data=b"fake_audio",
            sample_rate=22050,
            duration_seconds=1.5,
            text="Hola",
            processing_time_ms=200.0,
            reference_audio_path="/path/to/ref.wav",
        )
        mock_engine.clone_voice.return_value = mock_response
        return mock_engine

    def test_pipeline_initialization_without_voice_cloning(
        self, mock_translation_pipeline
    ):
        """Test pipeline initialization without voice cloning."""
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline
        )

        assert not pipeline.is_voice_cloning_available()
        assert pipeline.reference_audio_path is None

    def test_pipeline_initialization_with_voice_cloning(
        self, mock_translation_pipeline, mock_voice_engine
    ):
        """Test pipeline initialization with voice cloning."""
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
            reference_audio_path="/path/to/ref.wav",
        )

        assert pipeline.is_voice_cloning_available()
        assert pipeline.reference_audio_path == "/path/to/ref.wav"
        mock_voice_engine.load_reference_voice.assert_called_once_with(
            "/path/to/ref.wav"
        )

    def test_process_request_without_voice_cloning(self, mock_translation_pipeline):
        """Test processing request without voice cloning."""
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline
        )

        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        result = pipeline.process_request(request)

        assert isinstance(result, TranslationWithVoiceCloning)
        assert result.original_text == "Hello"
        assert result.translated_text == "Hola"
        assert not result.has_voice_cloning

    def test_process_request_with_voice_cloning(
        self, mock_translation_pipeline, mock_voice_engine
    ):
        """Test processing request with voice cloning."""
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
            reference_audio_path="/path/to/ref.wav",
        )

        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        with patch("live_translation.voice_cloning.pipeline.settings") as mock_settings:
            mock_settings.voice_cloning_enabled = True
            mock_settings.voice_sample_rate = 22050
            mock_settings.voice_cloning_speed = 1.0
            mock_settings.voice_cloning_temperature = 0.75

            result = pipeline.process_request(request)

        assert isinstance(result, TranslationWithVoiceCloning)
        assert result.has_voice_cloning
        assert result.cloned_audio_data == b"fake_audio"
        mock_voice_engine.clone_voice.assert_called_once()

    def test_set_reference_voice(self, mock_translation_pipeline, mock_voice_engine):
        """Test setting reference voice."""
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
        )

        result = pipeline.set_reference_voice("/new/path.wav")

        assert result
        assert pipeline.reference_audio_path == "/new/path.wav"
        mock_voice_engine.load_reference_voice.assert_called_with("/new/path.wav")

    def test_set_reference_voice_failure(
        self, mock_translation_pipeline, mock_voice_engine
    ):
        """Test setting reference voice failure."""
        mock_voice_engine.load_reference_voice.return_value = False

        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
        )

        result = pipeline.set_reference_voice("/invalid/path.wav")

        assert not result
        assert pipeline.reference_audio_path is None

    def test_get_supported_voice_languages(
        self, mock_translation_pipeline, mock_voice_engine
    ):
        """Test getting supported voice languages."""
        # Make the mock engine behave like XTTSVoiceCloner
        mock_voice_engine.get_supported_languages.return_value = ["en", "es", "fr"]

        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
        )

        with patch(
            "live_translation.voice_cloning.pipeline.XTTSVoiceCloner",
            return_value=mock_voice_engine,
        ):
            languages = pipeline.get_supported_voice_languages()

        assert languages == ["en", "es", "fr"]
