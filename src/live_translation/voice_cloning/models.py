"""Data models for voice cloning."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceCloningError(Exception):
    """Exception raised when voice cloning fails."""

    def __init__(self, message: str, error_code: str = "VOICE_CLONING_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class VoiceCloningRequest(BaseModel):
    """Request for voice cloning synthesis."""

    text: str = Field(..., description="Text to synthesize in the cloned voice")
    reference_audio_path: str = Field(
        ..., description="Path to reference audio file for voice cloning"
    )
    target_language: str = Field(default="en", description="Target language code")
    sample_rate: int = Field(default=22050, description="Audio sample rate for output")
    speed: float = Field(
        default=1.0, ge=0.1, le=3.0, description="Speech speed multiplier"
    )
    temperature: float = Field(
        default=0.75, ge=0.1, le=1.0, description="Model temperature for generation"
    )


class VoiceCloningResponse(BaseModel):
    """Response from voice cloning synthesis."""

    audio_data: bytes = Field(..., description="Generated audio data")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration_seconds: float = Field(..., description="Duration of generated audio")
    text: str = Field(..., description="Text that was synthesized")
    processing_time_ms: float = Field(
        ..., description="Time taken to process the request"
    )
    reference_audio_path: str = Field(..., description="Path to reference audio used")
    model_info: dict | None = Field(
        default=None, description="Model information and metadata"
    )
