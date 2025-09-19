"""Voice cloning module for live translation."""

from __future__ import annotations

from .engine import VoiceCloningEngine, VoiceCloningProtocol
from .models import VoiceCloningRequest, VoiceCloningResponse
from .pipeline import TranslationWithVoiceCloning, VoiceCloningTranslationPipeline
from .xtts_cloner import XTTSVoiceCloner

__all__ = [
    "VoiceCloningEngine",
    "VoiceCloningProtocol",
    "VoiceCloningRequest",
    "VoiceCloningResponse",
    "XTTSVoiceCloner",
    "TranslationWithVoiceCloning",
    "VoiceCloningTranslationPipeline",
]
