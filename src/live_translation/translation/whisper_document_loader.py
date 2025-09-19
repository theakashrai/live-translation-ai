"""Whisper adapter for speech-to-text functionality using LangChain patterns."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from live_translation.audio.whisper_utils import WhisperAudioProcessor
from live_translation.core.exceptions import TranslationError
from live_translation.translation.whisper_transcriber import WhisperTranscriber
from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperDocumentLoader(BaseLoader):
    """LangChain document loader for Whisper audio transcription.

    This class provides a LangChain-compatible interface for transcribing
    audio files using Whisper. It delegates the actual transcription work
    to WhisperTranscriber to avoid code duplication.
    """

    def __init__(
        self,
        file_path: str,
        model_name: str | None = None,
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        """Initialize the Whisper document loader.

        Args:
            file_path: Path to audio file
            model_name: Whisper model name (base, small, medium, large)
            device: Device to use (auto, cpu, cuda, mps)
            language: Language code for transcription
        """
        self.file_path = file_path
        self.language = language
        # Use WhisperTranscriber to avoid duplicating model loading logic
        self._transcriber = WhisperTranscriber(model_name, device)
        self._audio_processor = WhisperAudioProcessor()

    def lazy_load(self) -> Iterator[Document]:
        """Load and transcribe audio file."""
        try:
            # Load audio file using standard library approach
            audio_array = self._audio_processor.load_from_file(self.file_path)

            # Convert numpy array to bytes for transcriber interface
            audio_bytes = audio_array.astype("int16").tobytes()

            # Delegate transcription to WhisperTranscriber
            document = self._transcriber.transcribe(
                audio_bytes,
                sample_rate=16000,  # WhisperAudioProcessor loads at 16kHz
                language=self.language,
            )

            # Update metadata with file source
            document.metadata["source"] = self.file_path

            yield document

        except Exception as e:
            raise TranslationError(
                f"Transcription failed for {self.file_path}: {str(e)}",
                error_code="TRANSCRIPTION_FAILED",
                details={"file_path": self.file_path},
            ) from e
