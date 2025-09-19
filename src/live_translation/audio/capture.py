"""Audio capture functionality using sounddevice."""

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd

from live_translation.core.config import settings
from live_translation.core.exceptions import AudioCaptureError
from live_translation.core.models import AudioChunk
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class AudioCapture:
    """Real-time audio capture using sounddevice with enhanced error handling."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device: int | None = None,
        max_queue_size: int = 100,
    ) -> None:
        """Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Size of audio chunks to process
            device: Audio device index (None for default)
            max_queue_size: Maximum number of chunks in queue
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.max_queue_size = max_queue_size

        # State management
        self.is_recording = False
        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=max_queue_size)
        self.stream: sd.InputStream | None = None
        self.callback_fn: Callable[[AudioChunk], None] | None = None
        self.sequence_id = 0
        self._lock = threading.Lock()

        # Voice Activity Detection settings from config
        self.vad_threshold = settings.vad_threshold
        self.silence_duration = settings.silence_duration
        self.last_voice_time = 0.0

        # Statistics
        self.total_chunks_processed = 0
        self.dropped_chunks = 0

        # Validate device on initialization
        self._validate_audio_device()

    def _validate_audio_device(self) -> None:
        """Validate the selected audio device."""
        try:
            devices = sd.query_devices()
            if self.device is not None:
                if self.device >= len(devices) or self.device < 0:
                    raise AudioCaptureError(
                        f"Audio device index {self.device} not found",
                        error_code="DEVICE_NOT_FOUND",
                        device_id=self.device,
                    )
            else:
                # Use default input device
                default_device = sd.default.device[0]
                if default_device is None:
                    raise AudioCaptureError(
                        "No default audio input device found",
                        error_code="NO_DEFAULT_DEVICE",
                    )
                self.device = default_device

            device_info = devices[self.device]
            logger.info(
                f"Using audio device: {device_info['name']} "
                f"(channels: {device_info['max_input_channels']}, "
                f"sample_rate: {device_info['default_samplerate']}Hz)"
            )

        except Exception as e:
            if isinstance(e, AudioCaptureError):
                raise
            raise AudioCaptureError(
                f"Failed to validate audio device: {str(e)}",
                error_code="DEVICE_VALIDATION_FAILED",
            ) from e

    def start_recording(
        self, callback: Callable[[AudioChunk], None] | None = None
    ) -> None:
        """Start recording audio."""
        if self.is_recording:
            return

        self.callback_fn = callback
        self.sequence_id = 0

        try:
            logger.info(f"Starting audio recording at {self.sample_rate}Hz")

            def audio_callback(
                indata: np.ndarray,
                frames: int,
                time_info: Any,
                status: sd.CallbackFlags,
            ) -> None:
                """Callback function for audio stream."""
                try:
                    if status:
                        logger.warning(f"Audio callback status: {status}")

                    # Skip processing if input data is invalid
                    if indata is None or len(indata) == 0:
                        logger.warning("Empty audio input, skipping")
                        return

                    # Validate audio data range
                    if np.max(np.abs(indata)) == 0:
                        logger.debug("Silent audio chunk detected")
                    elif np.max(np.abs(indata)) > 1.5:
                        logger.warning("Audio input may be clipping, normalizing")
                        indata = np.clip(indata, -1.0, 1.0)

                    # Convert to bytes (16-bit PCM)
                    audio_data = (indata[:, 0] * 32767).astype(np.int16).tobytes()

                    # Calculate duration
                    duration_ms = (frames / self.sample_rate) * 1000

                    # Create audio chunk
                    chunk = AudioChunk(
                        data=audio_data,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        duration_ms=duration_ms,
                        sequence_id=self.sequence_id,
                    )

                    self.sequence_id += 1

                    # Add to queue with error handling
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        logger.warning("Audio queue full, dropping chunk")
                        # Remove oldest chunk and add new one
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(chunk)
                        except queue.Empty:
                            pass

                    # Call callback if provided
                    if self.callback_fn:
                        try:
                            self.callback_fn(chunk)
                        except Exception as callback_error:
                            logger.error(f"Audio callback error: {callback_error}")
                            # Don't re-raise to prevent breaking the audio stream

                except Exception as e:
                    logger.error(f"Critical error in audio callback: {e}")
                    # Don't re-raise to prevent breaking the audio stream

            # Create and start stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32,
            )

            if self.stream:
                self.stream.start()
                self.is_recording = True

        except Exception as e:
            raise AudioCaptureError(
                f"Failed to start audio recording: {str(e)}",
                error_code="RECORDING_START_FAILED",
            ) from e

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self.is_recording:
            return

        try:
            logger.info("Stopping audio recording")

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            self.is_recording = False
            self.callback_fn = None

        except Exception as e:
            logger.error(f"Error stopping audio recording: {str(e)}")

    def get_audio_chunk(self, timeout: float | None = None) -> AudioChunk | None:
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Clear all pending audio chunks from the queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def has_voice_activity(self, audio_data: bytes) -> bool:
        """Simple voice activity detection based on energy threshold."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_float**2))

            has_voice = rms_energy > self.vad_threshold

            if has_voice:
                self.last_voice_time = time.time()

            return has_voice

        except Exception as e:
            logger.error(f"Error in voice activity detection: {str(e)}")
            return False

    def should_stop_on_silence(self) -> bool:
        """Check if recording should stop due to silence."""
        if self.last_voice_time == 0:
            return False

        silence_duration = time.time() - self.last_voice_time
        return silence_duration > self.silence_duration

    @staticmethod
    def list_audio_devices() -> list[dict[str, Any]]:
        """List available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = []

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    input_devices.append(
                        {
                            "index": i,
                            "name": device["name"],
                            "channels": device["max_input_channels"],
                            "sample_rate": device["default_samplerate"],
                        }
                    )

            return input_devices

        except Exception as e:
            logger.error(f"Failed to list audio devices: {str(e)}")
            return []

    def __enter__(self) -> "AudioCapture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_recording()


class AudioBuffer:
    """Circular buffer for audio data with voice activity detection."""

    def __init__(self, max_duration_s: float = 30.0, sample_rate: int = 16000) -> None:
        self.max_duration_s = max_duration_s
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)

        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0

        # Voice activity detection
        self.vad_history: list[bool] = []
        self.speech_start_idx: int | None = None
        self.speech_end_idx: int | None = None

    def add_audio(self, audio_data: bytes) -> None:
        """Add audio data to the buffer."""
        # Convert bytes to float array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Add to circular buffer
        for sample in audio_float:
            self.buffer[self.write_pos] = sample
            self.write_pos = (self.write_pos + 1) % self.max_samples
            self.samples_written += 1

    def get_speech_segment(self) -> np.ndarray | None:
        """Extract speech segment from buffer based on VAD."""
        if self.samples_written == 0:
            return None

        # For simplicity, return the last N seconds of audio
        # In practice, you'd use proper VAD to find speech boundaries
        segment_duration_s = min(
            self.max_duration_s, self.samples_written / self.sample_rate
        )
        segment_samples = int(segment_duration_s * self.sample_rate)

        if self.samples_written < self.max_samples:
            # Buffer not full yet
            return self.buffer[: self.samples_written].copy()
        # Buffer is full, extract from circular buffer
        elif self.write_pos >= segment_samples:
            return self.buffer[self.write_pos - segment_samples : self.write_pos].copy()
        else:
            # Wrap around
            part1 = self.buffer[self.max_samples - (segment_samples - self.write_pos) :]
            part2 = self.buffer[: self.write_pos]
            return np.concatenate([part1, part2])

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.samples_written = 0
        self.vad_history.clear()
        self.speech_start_idx = None
        self.speech_end_idx = None

    def get_duration_seconds(self) -> float:
        """Get the duration of audio currently in the buffer.

        Returns:
            Duration in seconds of the buffered audio
        """
        return min(self.samples_written / self.sample_rate, self.max_duration_s)
