"""Core Pydantic models for translation requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


class LanguageCode(str, Enum):
    """Supported language codes."""

    AUTO = "auto"
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"

    @classmethod
    def is_valid(cls, code: str) -> bool:
        """Check if a language code is valid."""
        return code in [lang.value for lang in cls]

    @classmethod
    def get_supported_codes(cls) -> list[str]:
        """Get all supported language codes."""
        return [lang.value for lang in cls]


class AudioFormat(str, Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class JobStatus(str, Enum):
    """Translation job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationRequest(BaseModel):
    """Request model for translation operations."""

    # Required fields
    source_language: Union[LanguageCode, str] = Field(
        default=LanguageCode.AUTO,
        description="Source language code"
    )
    target_language: Union[LanguageCode, str] = Field(
        default=LanguageCode.ENGLISH,
        description="Target language code"
    )

    # Input data (one must be provided)
    text: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Text to translate"
    )
    audio_data: Optional[bytes] = Field(
        default=None,
        description="Audio data as bytes"
    )

    # Audio settings
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz"
    )
    audio_format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="Audio format"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Request timestamp"
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )

    @model_validator(mode="after")
    def validate_input_provided(self) -> "TranslationRequest":
        """Ensure at least one input method is provided."""
        if not self.text and not self.audio_data:
            raise ValueError("Either text or audio_data must be provided")
        return self

    @field_validator("source_language", "target_language")
    @classmethod
    def validate_language_codes(cls, v: Union[LanguageCode, str]) -> str:
        """Validate language codes."""
        if isinstance(v, LanguageCode):
            return v.value
        if not LanguageCode.is_valid(v):
            raise ValueError(f"Unsupported language code: {v}")
        return v


class TranslationResponse(BaseModel):
    """Response model for translation operations."""

    # Core response data
    original_text: str = Field(
        description="Original text (transcribed or provided)"
    )
    translated_text: str = Field(
        description="Translated text"
    )

    # Language information
    detected_language: Optional[str] = Field(
        default=None,
        description="Detected source language code"
    )

    # Quality metrics
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Translation confidence score"
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Processing time in milliseconds"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Associated request ID"
    )
    model_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Information about models used"
    )

    @field_validator("detected_language")
    @classmethod
    def validate_detected_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate detected language code."""
        if v is not None and not LanguageCode.is_valid(v):
            # Don't fail validation, just warn in logs if needed
            pass
        return v


class AudioChunk(BaseModel):
    """Model for audio chunk processing."""

    data: bytes = Field(description="Audio data as bytes")
    sample_rate: int = Field(
        ge=8000,
        le=48000,
        description="Sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels"
    )
    duration_ms: float = Field(
        ge=0.0,
        description="Duration in milliseconds"
    )
    sequence_id: int = Field(
        ge=0,
        description="Sequence ID for ordering"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Chunk timestamp"
    )

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0

    @property
    def size_kb(self) -> float:
        """Get data size in kilobytes."""
        return len(self.data) / 1024.0


class TranslationJob(BaseModel):
    """Model for tracking translation jobs."""

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique job identifier"
    )
    request: TranslationRequest = Field(
        description="Original translation request"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Job status"
    )
    response: Optional[TranslationResponse] = Field(
        default=None,
        description="Translation response if completed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Job creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )

    def mark_processing(self) -> None:
        """Mark job as processing."""
        self.status = JobStatus.PROCESSING
        self.updated_at = datetime.now()

    def mark_completed(self, response: TranslationResponse) -> None:
        """Mark job as completed with response."""
        self.status = JobStatus.COMPLETED
        self.response = response
        self.updated_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark job as failed with error message."""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.now()


class SystemStatus(BaseModel):
    """Model for system status information."""

    version: str = Field(description="Application version")
    models_loaded: dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded models"
    )
    device_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Device and hardware information"
    )
    memory_usage_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Memory usage in MB"
    )
    cache_size_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Cache size in MB"
    )
    uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="System uptime in seconds"
    )
    active_jobs: int = Field(
        default=0,
        ge=0,
        description="Number of active translation jobs"
    )
    total_translations: int = Field(
        default=0,
        ge=0,
        description="Total translations performed"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Status last updated timestamp"
    )

    @property
    def uptime_hours(self) -> float:
        """Get uptime in hours."""
        return self.uptime_seconds / 3600.0

    def add_translation(self) -> None:
        """Increment translation counter."""
        self.total_translations += 1
        self.last_updated = datetime.now()


# Type aliases for common use cases
TranslationInput = Union[str, bytes]
LanguageCodeType = Union[LanguageCode, str]
