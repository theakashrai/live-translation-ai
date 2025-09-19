"""Audio processing utilities for Whisper transcription."""

from __future__ import annotations

import io
import tempfile

import librosa
import numpy as np
import soundfile as sf

from live_translation.core.exceptions import TranslationError
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperAudioProcessor:
    """Handles audio processing using standard libraries for Whisper transcription."""

    @staticmethod
    def load_from_bytes(audio_data: bytes, sample_rate: int) -> np.ndarray:
        """Load audio from raw PCM bytes or audio file bytes."""
        try:
            # Handle empty or too small audio data
            if not audio_data or len(audio_data) < 100:
                logger.warning(
                    f"Audio data too small: {len(audio_data)} bytes, returning silence"
                )
                return np.zeros(1600)  # 0.1 seconds of silence at 16kHz

            # First try: assume it's raw PCM audio bytes (from microphone)
            try:
                # Convert raw bytes to int16 array, then to float32
                # Assumes 16-bit PCM audio (most common for microphone input)
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = (
                    audio_int16.astype(np.float32) / 32768.0
                )  # Normalize to [-1, 1]

                # Ensure we have float32 dtype for Whisper compatibility
                audio_array = audio_array.astype(np.float32)

                # Basic validation - check if we have reasonable audio values
                if len(audio_array) > 0 and np.max(np.abs(audio_array)) > 0.001:
                    logger.debug(
                        f"Processed raw PCM audio: {len(audio_array)} samples, dtype: {audio_array.dtype}"
                    )
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        audio_array = librosa.resample(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=16000,
                            res_type="kaiser_fast",
                        )
                        # Ensure float32 after resampling
                        audio_array = audio_array.astype(np.float32)
                    return audio_array

            except Exception as pcm_error:
                logger.debug(
                    f"Raw PCM processing failed: {pcm_error}, trying as audio file"
                )

            # Second try: assume it's an audio file (WAV, MP3, etc.)
            try:
                audio_buffer = io.BytesIO(audio_data)
                audio_array, orig_sr = sf.read(audio_buffer, dtype="float32")

                # Ensure float32 dtype
                audio_array = audio_array.astype(np.float32)

                # Resample to 16kHz if needed
                if orig_sr != 16000:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=orig_sr,
                        target_sr=16000,
                        res_type="kaiser_fast",
                    )
                    # Ensure float32 after resampling
                    audio_array = audio_array.astype(np.float32)

                logger.debug(
                    f"Processed audio file: {len(audio_array)} samples, dtype: {audio_array.dtype}"
                )
                return audio_array

            except Exception as file_error:
                logger.debug(f"Audio file processing failed: {file_error}")

            # If both methods fail, return silence to prevent pipeline breaks
            logger.warning("Could not process audio data, returning silence")
            return np.zeros(1600, dtype=np.float32)  # 0.1 seconds of silence

        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            # Return silence instead of raising exception to keep pipeline running
            return np.zeros(1600, dtype=np.float32)

    @staticmethod
    def load_from_file(file_path: str) -> np.ndarray:
        """Load audio file using librosa (standard approach)."""
        try:
            # Use librosa's standard file loading, ensure float32
            audio_array, _ = librosa.load(file_path, sr=16000, dtype=np.float32)
            # Explicitly cast to float32 to avoid dtype issues
            audio_array = audio_array.astype(np.float32)
            return audio_array
        except Exception as e:
            raise TranslationError(
                f"Failed to load audio file {file_path}: {str(e)}",
                error_code="AUDIO_FILE_LOAD_FAILED",
            ) from e

    @staticmethod
    def save_to_tempfile(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Save audio array to temporary file for processing."""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # Use soundfile to write the audio
            sf.write(temp_file.name, audio_array, sample_rate)
            return temp_file.name
        except Exception as e:
            raise TranslationError(
                f"Failed to save audio to temp file: {str(e)}",
                error_code="TEMP_FILE_CREATION_FAILED",
            ) from e
