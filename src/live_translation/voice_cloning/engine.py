"""Voice cloning engine interface and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Protocol

from .models import VoiceCloningRequest, VoiceCloningResponse


class VoiceCloningProtocol(Protocol):
    """Protocol for voice cloning engines.

    This protocol defines the interface that all voice cloning engines must implement
    to be compatible with the translation pipeline.
    """

    def clone_voice(self, request: VoiceCloningRequest) -> VoiceCloningResponse:
        """Clone voice and synthesize speech.

        Args:
            request: Voice cloning request containing text and reference audio

        Returns:
            VoiceCloningResponse with generated audio data

        Raises:
            VoiceCloningError: If voice cloning fails
        """
        ...

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready.

        Returns:
            True if the model is loaded and ready for use
        """
        ...

    def load_reference_voice(self, audio_path: str) -> bool:
        """Pre-load and validate reference voice.

        Args:
            audio_path: Path to reference audio file

        Returns:
            True if reference voice was successfully loaded
        """
        ...


class BaseVoiceCloningEngine(ABC):
    """Base class for voice cloning engines."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the voice cloning engine.

        Args:
            model_name: Name/path of the model to use
            device: Device to run the model on
            **kwargs: Additional engine-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._model = None

    @abstractmethod
    def _load_model(self) -> None:
        """Load the voice cloning model."""
        ...

    @abstractmethod
    def _clone_voice_impl(self, request: VoiceCloningRequest) -> VoiceCloningResponse:
        """Implement the actual voice cloning logic."""
        ...

    def clone_voice(self, request: VoiceCloningRequest) -> VoiceCloningResponse:
        """Clone voice and synthesize speech."""
        if not self._loaded:
            self._load_model()

        return self._clone_voice_impl(request)

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    def load_reference_voice(self, audio_path: str) -> bool:
        """Pre-load and validate reference voice."""
        # Default implementation - can be overridden by subclasses
        try:
            return os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
        except Exception:
            return False


# Type alias for convenience
VoiceCloningEngine = VoiceCloningProtocol
