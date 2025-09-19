"""Streaming Whisper processor for real-time audio processing."""

from __future__ import annotations

import os

from langchain_core.documents import Document
import numpy as np

from live_translation.audio.whisper_utils import WhisperAudioProcessor
from live_translation.core.config import settings
from live_translation.translation.whisper_document_loader import WhisperDocumentLoader
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperStreamProcessor:
    """Streaming audio processor using standard libraries only."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        chunk_length_s: int = 30,
    ) -> None:
        """Initialize the stream processor."""
        self.model_name = model_name or settings.whisper_model
        self.device = device or "auto"
        self.chunk_length_s = chunk_length_s
        self.audio_chunks: list[np.ndarray] = []
        self._target_sample_rate = 16000
        self._audio_processor = WhisperAudioProcessor()

    def add_audio_chunk(self, audio_data: bytes, sample_rate: int) -> Document | None:
        """Add audio chunk using standard library processing."""
        try:
            # Use standard library to process chunk
            audio_array = self._audio_processor.load_from_bytes(audio_data, sample_rate)
            self.audio_chunks.append(audio_array)

            # Calculate total duration
            total_samples = sum(len(chunk) for chunk in self.audio_chunks)
            total_duration = total_samples / self._target_sample_rate

            if total_duration >= self.chunk_length_s:
                # Use standard library approach - combine and save to temp file
                combined_audio = np.concatenate(self.audio_chunks)
                temp_file = self._audio_processor.save_to_tempfile(
                    combined_audio, self._target_sample_rate
                )

                try:
                    # Use standard document loader
                    loader = WhisperDocumentLoader(
                        temp_file, self.model_name, self.device
                    )
                    documents = loader.load()
                    result = documents[0] if documents else None
                finally:
                    # Clean up temp file
                    os.unlink(temp_file)

                # Keep only recent chunks for overlap
                self.audio_chunks = (
                    self.audio_chunks[-1:] if len(self.audio_chunks) > 1 else []
                )
                return result

            return None
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None

    def flush_buffer(self) -> Document | None:
        """Flush remaining audio using standard libraries."""
        if self.audio_chunks:
            try:
                combined_audio = np.concatenate(self.audio_chunks)
                temp_file = self._audio_processor.save_to_tempfile(
                    combined_audio, self._target_sample_rate
                )

                try:
                    loader = WhisperDocumentLoader(
                        temp_file, self.model_name, self.device
                    )
                    documents = loader.load()
                    result = documents[0] if documents else None
                finally:
                    os.unlink(temp_file)

                self.audio_chunks = []
                return result
            except Exception as e:
                logger.error(f"Error flushing audio buffer: {e}")
                return None
        return None
