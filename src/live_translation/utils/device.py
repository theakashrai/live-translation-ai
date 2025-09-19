"""Device management utilities for ML models."""

from __future__ import annotations

from typing import Any

import torch

from live_translation.utils.logger import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """Handles device detection and optimization."""

    @staticmethod
    def get_optimal_device() -> str:
        """Get the optimal device for the current system."""
        try:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def get_transcription_options(
        device: str, language: str | None = None
    ) -> dict[str, Any]:
        """Get device-optimized transcription options."""
        options: dict[str, Any] = {
            "language": language,
            "task": "transcribe",
            "verbose": False,
        }

        # Device-specific optimizations
        if device == "mps":
            options.update(
                {
                    "fp16": False,
                    "beam_size": 1,
                    "best_of": 1,
                    "temperature": 0.0,
                }
            )
        elif device == "cpu":
            options.update(
                {
                    "fp16": False,
                    "beam_size": 1,
                }
            )
        else:
            # CUDA optimizations
            options["fp16"] = True

        # Remove None values
        return {k: v for k, v in options.items() if v is not None}
