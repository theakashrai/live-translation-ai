"""Audio processing utilities and streaming pipeline."""

import asyncio
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Optional

import numpy as np

from live_translation.audio.capture import AudioBuffer, AudioCapture
from live_translation.core.config import settings
from live_translation.core.exceptions import AudioProcessingError, handle_audio_error
from live_translation.core.models import (AudioChunk, TranslationRequest,
                                          TranslationResponse)
from live_translation.translation.engine import TranslationPipeline
from live_translation.utils.logger import get_logger, log_audio_metrics

logger = get_logger(__name__)


class AudioProcessor:
    """Process audio chunks for translation with enhanced performance and error handling."""

    def __init__(
        self,
        translation_pipeline: TranslationPipeline,
        chunk_duration_s: Optional[float] = None,
        vad_threshold: Optional[float] = None,
    ) -> None:
        """Initialize audio processor.

        Args:
            translation_pipeline: The translation pipeline to use
            chunk_duration_s: Duration between translations (uses config default if None)
            vad_threshold: Voice activity detection threshold (uses config default if None)
        """
        self.translation_pipeline = translation_pipeline
        self.chunk_duration_s = chunk_duration_s or 3.0
        self.vad_threshold = vad_threshold or settings.vad_threshold

        # Initialize buffer with settings from config
        self.audio_buffer = AudioBuffer(
            max_duration_s=settings.chunk_length,
            sample_rate=settings.sample_rate,
        )

        # Processing state
        self.is_processing = False
        self.last_translation_time = 0.0

        # Statistics
        self.total_chunks_processed = 0
        self.successful_translations = 0
        self.failed_translations = 0

    def process_audio_chunk(self, chunk: AudioChunk) -> Optional[TranslationResponse]:
        """Process a single audio chunk.

        Args:
            chunk: The audio chunk to process

        Returns:
            TranslationResponse if translation was triggered, None otherwise

        Raises:
            AudioProcessingError: If audio processing fails
        """
        start_time = time.time()
        self.total_chunks_processed += 1

        try:
            # Add to buffer
            self.audio_buffer.add_audio(chunk.data)

            # Check if we should trigger translation
            should_translate = self._should_trigger_translation(chunk)

            if should_translate:
                return self._perform_translation(start_time)

        except Exception as e:
            self.failed_translations += 1
            handle_audio_error(
                e,
                "chunk_processing",
                chunk_size=len(chunk.data),
                sequence_id=chunk.sequence_id,
            )

        return None

    def _should_trigger_translation(self, chunk: AudioChunk) -> bool:
        """Determine if we should trigger translation.

        Args:
            chunk: The current audio chunk

        Returns:
            True if translation should be triggered
        """
        # Time-based trigger
        time_since_last = time.time() - self.last_translation_time
        if time_since_last >= self.chunk_duration_s:
            return True

        # Voice activity detection trigger
        if self._detect_voice_activity(chunk.data):
            # Check if we've accumulated enough audio after voice activity
            buffer_duration = self.audio_buffer.get_duration_seconds()
            return buffer_duration >= self.chunk_duration_s / 2

        return False

    def _detect_voice_activity(self, audio_data: bytes) -> bool:
        """Simple voice activity detection.

        Args:
            audio_data: Raw audio data

        Returns:
            True if voice activity is detected
        """
        try:
            # Convert bytes to numpy array for energy calculation
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return False

            # Calculate RMS energy
            audio_float = audio_array.astype(np.float32) / 32768.0
            rms_energy = np.sqrt(np.mean(audio_float ** 2))

            return rms_energy > self.vad_threshold

        except Exception as e:
            logger.warning(f"Voice activity detection failed: {e}")
            return False

    def _perform_translation(self, start_time: float) -> Optional[TranslationResponse]:
        """Perform translation on buffered audio.

        Args:
            start_time: Time when processing started

        Returns:
            TranslationResponse if successful, None otherwise
        """
        try:
            # Extract speech segment
            speech_audio = self.audio_buffer.get_speech_segment()

            if speech_audio is not None and len(speech_audio) > 0:
                # Convert to bytes for translation pipeline
                audio_bytes = (speech_audio * 32767).astype(np.int16).tobytes()

                # Create translation request
                request = TranslationRequest(
                    audio_data=audio_bytes,
                    sample_rate=settings.sample_rate,
                    source_language=settings.default_source_lang,
                    target_language=settings.default_target_lang,
                )

                # Process translation
                response = self.translation_pipeline.process_request(request)

                # Update statistics
                self.successful_translations += 1
                self.last_translation_time = time.time()

                # Log metrics
                processing_time = (time.time() - start_time) * 1000
                log_audio_metrics(
                    sample_rate=settings.sample_rate,
                    duration_ms=len(speech_audio) /
                    settings.sample_rate * 1000,
                    processing_time_ms=processing_time,
                )

                # Clear buffer after successful translation
                self.audio_buffer.clear()

                return response

        except Exception as e:
            self.failed_translations += 1
            logger.error(f"Translation failed: {e}")

        return None

    def reset(self) -> None:
        """Reset the processor state."""
        self.audio_buffer.clear()
        self.last_translation_time = 0.0
        self.total_chunks_processed = 0
        self.successful_translations = 0
        self.failed_translations = 0

    def get_statistics(self) -> dict[str, Any]:
        """Get processor statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "successful_translations": self.successful_translations,
            "failed_translations": self.failed_translations,
            "success_rate": (
                self.successful_translations /
                max(1, self.successful_translations + self.failed_translations)
            ),
            "buffer_duration_s": self.audio_buffer.get_duration_seconds(),
        }


class StreamingTranslator:
    """Real-time streaming translation system."""

    def __init__(
        self,
        translation_pipeline: TranslationPipeline,
        sample_rate: int = 16000,
        chunk_duration_s: float = 3.0,
    ) -> None:
        self.translation_pipeline = translation_pipeline
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s

        self.audio_capture: Optional[AudioCapture] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.translation_callback: Optional[Callable[[
            TranslationResponse], None]] = None

        self.is_streaming = False
        self._stop_event = asyncio.Event()

    async def start_streaming(
        self,
        translation_callback: Callable[[TranslationResponse], None],
        device: Optional[int] = None,
    ) -> None:
        """Start real-time streaming translation."""
        if self.is_streaming:
            return

        try:
            logger.info("Starting streaming translation")

            self.translation_callback = translation_callback
            self._stop_event.clear()

            # Initialize components
            self.audio_capture = AudioCapture(
                sample_rate=self.sample_rate,
                device=device,
            )

            self.audio_processor = AudioProcessor(
                translation_pipeline=self.translation_pipeline,
                chunk_duration_s=self.chunk_duration_s,
            )

            # Start audio capture
            self.audio_capture.start_recording()
            self.is_streaming = True

            # Process audio chunks in a loop
            await self._processing_loop()

        except Exception as e:
            logger.error(f"Failed to start streaming: {str(e)}")
            await self.stop_streaming()
            raise AudioProcessingError(
                f"Failed to start streaming translation: {str(e)}",
                error_code="STREAMING_START_FAILED",
            ) from e

    async def stop_streaming(self) -> None:
        """Stop streaming translation."""
        if not self.is_streaming:
            return

        try:
            logger.info("Stopping streaming translation")

            self._stop_event.set()
            self.is_streaming = False

            if self.audio_capture:
                self.audio_capture.stop_recording()
                self.audio_capture = None

            self.audio_processor = None
            self.translation_callback = None

        except Exception as e:
            logger.error(f"Error stopping streaming: {str(e)}")

    async def _processing_loop(self) -> None:
        """Main processing loop for audio chunks."""
        chunk_timeout = 0.1  # 100ms timeout

        while not self._stop_event.is_set() and self.is_streaming:
            try:
                if not self.audio_capture or not self.audio_processor:
                    break

                # Get next audio chunk
                chunk = self.audio_capture.get_audio_chunk(
                    timeout=chunk_timeout)

                if chunk is None:
                    # Yield control to other async tasks
                    await asyncio.sleep(0.01)
                    continue

                # Process chunk
                response = self.audio_processor.process_audio_chunk(chunk)

                # Call callback if translation was produced
                if response and self.translation_callback:
                    try:
                        self.translation_callback(response)
                    except Exception as e:
                        logger.error(f"Translation callback error: {str(e)}")

                # Yield control
                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(0.1)  # Brief pause before continuing

    async def process_audio_file(
        self,
        file_path: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> AsyncGenerator[TranslationResponse, None]:
        """Process an audio file and yield translation responses."""
        try:
            logger.info(f"Processing audio file: {file_path}")

            # Load audio file (simplified - in practice use librosa or similar)
            # For now, assume we have a way to load and chunk the file

            # This is a placeholder - you'd implement actual file loading here
            # For demonstration, we'll create a mock response
            yield TranslationResponse(
                original_text="[File processing not implemented yet]",
                translated_text=f"[{source_lang} -> {target_lang}] File processing placeholder",
                detected_language=source_lang,
                confidence=0.95,
                processing_time_ms=100.0,
                model_info={"note": "File processing is not yet implemented"},
            )

        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise AudioProcessingError(
                f"Failed to process audio file: {str(e)}",
                error_code="FILE_PROCESSING_FAILED",
                details={"file_path": file_path},
            ) from e


class BatchAudioProcessor:
    """Process multiple audio files in batch."""

    def __init__(self, translation_pipeline: TranslationPipeline) -> None:
        self.translation_pipeline = translation_pipeline

    async def process_files(
        self,
        file_paths: list[str],
        source_lang: str = "auto",
        target_lang: str = "en",
        max_concurrent: int = 3,
    ) -> AsyncGenerator[tuple[str, TranslationResponse], None]:
        """Process multiple audio files concurrently."""

        async def process_single_file(file_path: str) -> tuple[str, TranslationResponse]:
            """Process a single file."""
            streaming_translator = StreamingTranslator(
                self.translation_pipeline)

            responses = []
            async for response in streaming_translator.process_audio_file(
                file_path, source_lang, target_lang
            ):
                responses.append(response)

            # For now, return the last response
            # In practice, you might want to concatenate all responses
            final_response = responses[-1] if responses else TranslationResponse(
                original_text="",
                translated_text="",
                detected_language=source_lang,
                confidence=0.0,
                processing_time_ms=0.0,
            )

            return file_path, final_response

        # Process files with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_process(file_path: str) -> tuple[str, TranslationResponse]:
            async with semaphore:
                return await process_single_file(file_path)

        # Create tasks for all files
        tasks = [bounded_process(file_path) for file_path in file_paths]

        # Yield results as they complete
        for task in asyncio.as_completed(tasks):
            try:
                file_path, response = await task
                yield file_path, response
            except Exception as e:
                logger.error(f"Failed to process file: {str(e)}")
                # Continue with other files
