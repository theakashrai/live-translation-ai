"""Integration test for voice cloning with real XTTS model and audio sample."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from live_translation.voice_cloning import VoiceCloningRequest, XTTSVoiceCloner
from live_translation.voice_cloning.models import VoiceCloningError


class TestVoiceCloningIntegrationReal:
    """Integration tests using real XTTS model and audio samples."""

    @pytest.fixture(scope="class")
    def audio_sample_path(self) -> str:
        """Get path to the 8kHz audio sample."""
        # Assume tests are run from project root
        sample_path = "examples/audio_sample_8k_3.wav"
        if not os.path.exists(sample_path):
            pytest.skip("Audio sample file not found: examples/audio_sample_8k.wav")
        return sample_path

    @pytest.fixture(scope="class")
    def voice_cloner(self) -> Generator[XTTSVoiceCloner, None, None]:
        """Create real XTTS voice cloner instance."""
        try:
            # Use CPU to avoid GPU dependencies in tests
            cloner = XTTSVoiceCloner(device="cpu")
            yield cloner
        except ImportError as e:
            pytest.skip(f"TTS library not available: {e}")

    def test_reference_voice_validation_real_file(
        self, voice_cloner: XTTSVoiceCloner, audio_sample_path: str
    ):
        """Test reference voice validation with real audio file."""
        result = voice_cloner.load_reference_voice(audio_sample_path)
        assert result is True, "Should successfully validate the 8kHz audio sample"

    def test_reference_voice_validation_missing_file(
        self, voice_cloner: XTTSVoiceCloner
    ):
        """Test reference voice validation with missing file."""
        result = voice_cloner.load_reference_voice("/nonexistent/path.wav")
        assert result is False, "Should fail for non-existent file"

    def test_supported_languages(self, voice_cloner: XTTSVoiceCloner):
        """Test getting supported languages from real XTTS model."""
        languages = voice_cloner.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0

        # Check for common languages that XTTS v2 supports
        expected_languages = ["en", "es", "fr", "de", "it"]
        for lang in expected_languages:
            assert lang in languages, f"Language {lang} should be supported by XTTS"

    def test_language_support_check(self, voice_cloner: XTTSVoiceCloner):
        """Test language support checking."""
        # Test supported languages
        assert voice_cloner.is_language_supported("en") is True
        assert voice_cloner.is_language_supported("es") is True
        assert voice_cloner.is_language_supported("fr") is True

        # Test unsupported language
        assert voice_cloner.is_language_supported("xyz") is False

    def test_voice_cloning_end_to_end(
        self, voice_cloner: XTTSVoiceCloner, audio_sample_path: str
    ):
        """End-to-end test of voice cloning with real model and audio sample.

        This test loads the XTTS model and performs actual voice synthesis.
        """
        # Create a simple voice cloning request
        audio_reference_text = """
my dear grandson, where are you, I have been waiting since morning, come back home right now
"""

        request = VoiceCloningRequest(
            text=audio_reference_text,
            reference_audio_path=audio_sample_path,
            target_language="en",
            sample_rate=22050,
            speed=1.0,
            temperature=0.75,
        )

        # Perform voice cloning
        response = voice_cloner.clone_voice(request)

        # Verify response structure
        assert response.text == audio_reference_text
        assert response.reference_audio_path == audio_sample_path
        assert response.sample_rate > 0  # XTTS uses 24kHz by default
        assert isinstance(response.audio_data, bytes)
        assert len(response.audio_data) > 0
        assert response.duration_seconds > 0
        assert response.processing_time_ms > 0

        # Verify model info is provided
        assert response.model_info is not None
        assert "model_name" in response.model_info
        assert "device" in response.model_info
        assert "language" in response.model_info

        print("✅ Voice cloning successful!")
        print(f"   Audio size: {len(response.audio_data)} bytes")
        print(f"   Duration: {response.duration_seconds:.2f}s")
        print(f"   Sample rate: {response.sample_rate}Hz")
        print(f"   Processing time: {response.processing_time_ms:.1f}ms")

        # Save the generated audio to temp-data for listening
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("temp-data/voice-cloning-test-outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"hello_world_en_{timestamp}.wav"

        # Save raw audio bytes as WAV file
        self._save_audio_as_wav(response.audio_data, response.sample_rate, output_file)

        print(f"🎵 Audio saved: {output_file}")
        print(f"   Listen with: afplay {output_file}  # macOS")
        print(f"               aplay {output_file}   # Linux")

    @pytest.mark.slow
    def test_voice_cloning_multilingual(
        self, voice_cloner: XTTSVoiceCloner, audio_sample_path: str
    ):
        """Test voice cloning with different target languages."""
        test_cases = [
            ("Hello world", "en", "English"),
            ("Hello world", "es", "Spanish"),
            ("Hello world", "fr", "French"),
        ]

        for text, target_lang, lang_name in test_cases:
            if not voice_cloner.is_language_supported(target_lang):
                continue

            request = VoiceCloningRequest(
                text=text,
                reference_audio_path=audio_sample_path,
                target_language=target_lang,
                sample_rate=22050,
                speed=1.0,
                temperature=0.75,
            )

            try:
                response = voice_cloner.clone_voice(request)

                assert response.text == text
                assert isinstance(response.audio_data, bytes)
                assert len(response.audio_data) > 0
                assert response.duration_seconds > 0

                print(f"✅ {lang_name} ({target_lang}) synthesis successful!")
                print(f"   Duration: {response.duration_seconds:.2f}s")

                # Save the generated audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("temp-data/voice-cloning-test-outputs")
                output_dir.mkdir(parents=True, exist_ok=True)

                safe_text = text.replace(" ", "_").replace(",", "").lower()
                output_file = output_dir / f"{safe_text}_{target_lang}_{timestamp}.wav"

                self._save_audio_as_wav(
                    response.audio_data, response.sample_rate, output_file
                )
                print(f"🎵 Audio saved: {output_file}")

            except Exception as e:
                pytest.fail(f"Voice cloning failed for {lang_name}: {e}")

    @pytest.mark.slow
    def test_voice_cloning_with_output_file(
        self, voice_cloner: XTTSVoiceCloner, audio_sample_path: str
    ):
        """Test voice cloning and save output to file."""
        request = VoiceCloningRequest(
            text="This is a test of voice cloning",
            reference_audio_path=audio_sample_path,
            target_language="en",
            sample_rate=22050,
        )

        try:
            response = voice_cloner.clone_voice(request)

            # Save output to temp-data directory instead of temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("temp-data/voice-cloning-test-outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"test_file_output_{timestamp}.wav"
            self._save_audio_as_wav(
                response.audio_data, response.sample_rate, output_path
            )

            # Verify file was created and has content
            assert output_path.exists()
            file_size = output_path.stat().st_size
            assert file_size > 0

            print(f"✅ Audio saved to: {output_path}")
            print(f"   File size: {file_size} bytes")
            print(f"🎵 Listen with: afplay {output_path}")  # macOS command

        except Exception as e:
            pytest.fail(f"Voice cloning with file output failed: {e}")

    def test_voice_cloning_error_handling(self, voice_cloner: XTTSVoiceCloner):
        """Test error handling with invalid inputs."""
        # Test with non-existent reference audio
        request = VoiceCloningRequest(
            text="Test text",
            reference_audio_path="/nonexistent/file.wav",
            target_language="en",
        )

        with pytest.raises(VoiceCloningError) as exc_info:
            voice_cloner.clone_voice(request)

        assert "REFERENCE_AUDIO_NOT_FOUND" in str(exc_info.value)

    def _save_audio_as_wav(
        self, audio_data: bytes, sample_rate: int, output_path: Path
    ) -> None:
        """Save raw audio bytes as a WAV file using soundfile."""
        try:
            # Convert bytes to numpy array (assuming int16 format from XTTS)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 for soundfile
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Save as WAV file
            sf.write(str(output_path), audio_float, sample_rate)

        except ImportError:
            # Fallback: save raw bytes (not playable, but preserved for debugging)
            with open(output_path, "wb") as f:
                f.write(audio_data)
