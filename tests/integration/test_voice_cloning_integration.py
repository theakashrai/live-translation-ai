"""Integration tests for voice cloning functionality."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
from live_translation.core.models import TranslationRequest
from live_translation.voice_cloning import (
    TranslationWithVoiceCloning,
    VoiceCloningRequest,
    VoiceCloningTranslationPipeline,
    XTTSVoiceCloner,
)


class TestVoiceCloningIntegration:
    """Integration tests for voice cloning with translation pipeline."""

    @pytest.fixture
    def temp_reference_audio(self):
        """Create a temporary reference audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write a minimal WAV file header + some fake audio data
            wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
            f.write(wav_header)
            f.write(b"\x00" * 1000)  # Some fake audio data
            temp_path = f.name

        yield temp_path

        # Clean up
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    @pytest.mark.skip(reason="Requires TTS library and models to be installed")
    def test_xtts_voice_cloner_full_workflow(self, temp_reference_audio):
        """Test full XTTS voice cloning workflow (requires TTS installation)."""
        # This test requires the actual TTS library and models
        # Skip by default since it's resource-intensive

        cloner = XTTSVoiceCloner(device="cpu")

        # Test reference voice loading
        assert cloner.load_reference_voice(temp_reference_audio)

        # Test voice cloning request
        request = VoiceCloningRequest(
            text="Hello, this is a test.",
            reference_audio_path=temp_reference_audio,
            target_language="en",
            speed=1.0,
            temperature=0.75,
        )

        # This would require actual model loading and processing
        # response = cloner.clone_voice(request)
        # assert isinstance(response.audio_data, bytes)
        # assert response.duration_seconds > 0

    def test_voice_cloning_pipeline_integration_mock(self, temp_reference_audio):
        """Test voice cloning pipeline integration with mocked components."""
        from unittest.mock import Mock

        # Mock the translation pipeline
        mock_translation_pipeline = Mock()
        mock_translation_response = Mock()
        mock_translation_response.original_text = "Hello world"
        mock_translation_response.translated_text = "Hola mundo"
        mock_translation_response.detected_language = "en"
        mock_translation_response.processing_time_ms = 150.0
        mock_translation_pipeline.process_request.return_value = (
            mock_translation_response
        )

        # Mock the voice cloning engine
        mock_voice_engine = Mock()
        mock_voice_engine.is_loaded.return_value = True
        mock_voice_engine.load_reference_voice.return_value = True

        mock_voice_response = Mock()
        mock_voice_response.audio_data = b"fake_synthesized_audio"
        mock_voice_response.sample_rate = 22050
        mock_voice_response.duration_seconds = 2.1
        mock_voice_response.text = "Hola mundo"
        mock_voice_response.processing_time_ms = 300.0
        mock_voice_response.reference_audio_path = temp_reference_audio
        mock_voice_engine.clone_voice.return_value = mock_voice_response

        # Create the voice cloning pipeline
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
            reference_audio_path=temp_reference_audio,
        )

        # Test the pipeline
        assert pipeline.is_voice_cloning_available()

        # Create and process a request
        request = TranslationRequest(
            text="Hello world",
            source_language="en",
            target_language="es",
        )

        with patch("live_translation.voice_cloning.pipeline.settings") as mock_settings:
            mock_settings.voice_cloning_enabled = True
            mock_settings.voice_sample_rate = 22050
            mock_settings.voice_cloning_speed = 1.0
            mock_settings.voice_cloning_temperature = 0.75

            result = pipeline.process_request(request, enable_voice_cloning=True)

        # Verify the result
        assert isinstance(result, TranslationWithVoiceCloning)
        assert result.original_text == "Hello world"
        assert result.translated_text == "Hola mundo"
        assert result.has_voice_cloning
        assert result.cloned_audio_data == b"fake_synthesized_audio"
        assert result.total_processing_time_ms == 450.0  # 150 + 300

        # Verify mock calls
        mock_translation_pipeline.process_request.assert_called_once_with(request)
        mock_voice_engine.clone_voice.assert_called_once()
        mock_voice_engine.load_reference_voice.assert_called_once_with(
            temp_reference_audio
        )

    def test_voice_cloning_pipeline_fallback_on_error(self, temp_reference_audio):
        """Test that pipeline gracefully handles voice cloning errors."""
        from unittest.mock import Mock

        # Mock the translation pipeline
        mock_translation_pipeline = Mock()
        mock_translation_response = Mock()
        mock_translation_response.original_text = "Hello"
        mock_translation_response.translated_text = "Hola"
        mock_translation_response.detected_language = "en"
        mock_translation_response.processing_time_ms = 100.0
        mock_translation_pipeline.process_request.return_value = (
            mock_translation_response
        )

        # Mock voice cloning engine that raises an error
        mock_voice_engine = Mock()
        mock_voice_engine.is_loaded.return_value = True
        mock_voice_engine.load_reference_voice.return_value = True
        mock_voice_engine.clone_voice.side_effect = Exception("Voice cloning failed")

        # Create pipeline
        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
            reference_audio_path=temp_reference_audio,
        )

        # Process request
        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        with patch("live_translation.voice_cloning.pipeline.settings") as mock_settings:
            mock_settings.voice_cloning_enabled = True

            result = pipeline.process_request(request, enable_voice_cloning=True)

        # Should still return translation result, but without voice cloning
        assert isinstance(result, TranslationWithVoiceCloning)
        assert result.original_text == "Hello"
        assert result.translated_text == "Hola"
        assert not result.has_voice_cloning
        assert result.cloned_audio_data is None

    def test_voice_cloning_disabled_by_setting(self, temp_reference_audio):
        """Test voice cloning when disabled by settings."""
        from unittest.mock import Mock

        mock_translation_pipeline = Mock()
        mock_translation_response = Mock()
        mock_translation_response.original_text = "Hello"
        mock_translation_response.translated_text = "Hola"
        mock_translation_response.detected_language = "en"
        mock_translation_response.processing_time_ms = 100.0
        mock_translation_pipeline.process_request.return_value = (
            mock_translation_response
        )

        mock_voice_engine = Mock()
        mock_voice_engine.is_loaded.return_value = True
        mock_voice_engine.load_reference_voice.return_value = True

        pipeline = VoiceCloningTranslationPipeline(
            translation_pipeline=mock_translation_pipeline,
            voice_cloning_engine=mock_voice_engine,
            reference_audio_path=temp_reference_audio,
        )

        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        # Process with voice cloning explicitly disabled
        result = pipeline.process_request(request, enable_voice_cloning=False)

        assert isinstance(result, TranslationWithVoiceCloning)
        assert not result.has_voice_cloning
        mock_voice_engine.clone_voice.assert_not_called()

    def test_reference_voice_path_override(self, temp_reference_audio):
        """Test overriding reference voice path per request."""
        from unittest.mock import Mock

        # Create second temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"different_audio_data")
            temp_path2 = f.name

        try:
            mock_translation_pipeline = Mock()
            mock_translation_response = Mock()
            mock_translation_response.original_text = "Test"
            mock_translation_response.translated_text = "Prueba"
            mock_translation_response.detected_language = "en"
            mock_translation_response.processing_time_ms = 100.0
            mock_translation_pipeline.process_request.return_value = (
                mock_translation_response
            )

            mock_voice_engine = Mock()
            mock_voice_engine.is_loaded.return_value = True
            mock_voice_engine.load_reference_voice.return_value = True

            mock_voice_response = Mock()
            mock_voice_response.audio_data = b"override_audio"
            mock_voice_response.sample_rate = 22050
            mock_voice_response.duration_seconds = 1.0
            mock_voice_response.text = "Prueba"
            mock_voice_response.processing_time_ms = 200.0
            mock_voice_engine.clone_voice.return_value = mock_voice_response

            # Create pipeline with first reference
            pipeline = VoiceCloningTranslationPipeline(
                translation_pipeline=mock_translation_pipeline,
                voice_cloning_engine=mock_voice_engine,
                reference_audio_path=temp_reference_audio,
            )

            request = TranslationRequest(
                text="Test",
                source_language="en",
                target_language="es",
            )

            with patch(
                "live_translation.voice_cloning.pipeline.settings"
            ) as mock_settings:
                mock_settings.voice_cloning_enabled = True
                mock_settings.voice_sample_rate = 22050
                mock_settings.voice_cloning_speed = 1.0
                mock_settings.voice_cloning_temperature = 0.75

                # Process with override reference path
                result = pipeline.process_request(
                    request, reference_audio_path=temp_path2, enable_voice_cloning=True
                )

            assert result.has_voice_cloning

            # Verify the voice cloning was called with correct request
            voice_cloning_call = mock_voice_engine.clone_voice.call_args[0][0]
            assert voice_cloning_call.reference_audio_path == temp_path2

        finally:
            # Clean up second temp file
            try:
                os.unlink(temp_path2)
            except FileNotFoundError:
                pass
