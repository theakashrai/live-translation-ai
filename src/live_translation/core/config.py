"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRANSLATION_",
        case_sensitive=False,
        extra="forbid",
        validate_assignment=True,
    )

    # Model settings
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        description="Whisper model size to use"
    )
    translation_model: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="Translation model name"
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device to run models on"
    )

    # Audio settings
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz"
    )
    chunk_length: int = Field(
        default=30,
        ge=5,
        le=60,
        description="Audio chunk length in seconds"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels"
    )
    audio_device: Optional[int] = Field(
        default=None,
        description="Audio device index (None for default)"
    )

    # Performance settings
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for processing"
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of processing threads"
    )
    max_text_length: int = Field(
        default=5000,
        ge=1,
        le=50000,
        description="Maximum text length for translation"
    )

    # Language defaults
    default_source_lang: str = Field(
        default="auto",
        description="Default source language code"
    )
    default_target_lang: str = Field(
        default="en",
        description="Default target language code"
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "structured", "simple"] = Field(
        default="structured",
        description="Log format style"
    )
    enable_performance_logging: bool = Field(
        default=True,
        description="Enable performance metrics logging"
    )

    # Cache and storage
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "live-translation-ai",
        description="Cache directory path",
    )
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "live-translation-ai" / "models",
        description="Model cache directory path",
    )
    clear_cache_on_startup: bool = Field(
        default=False,
        description="Clear cache directories on startup"
    )

    # Voice Activity Detection settings
    vad_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Voice activity detection threshold"
    )
    silence_duration: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Silence duration before stopping recording (seconds)"
    )

    # Translation settings
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for translations"
    )
    enable_language_detection: bool = Field(
        default=True,
        description="Enable automatic language detection"
    )

    @field_validator("cache_dir", "model_cache_dir", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """Expand user home directory in paths."""
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser().resolve()

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and auto-detect device availability."""
        if v == "auto":
            # Auto-detect best available device
            # Note: CPU is preferred over MPS for stability with Whisper/NLLB models
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    # MPS available but may have compatibility issues
                    # Applications should handle MPS fallback gracefully
                    return "cpu"  # Default to CPU for better stability
                else:
                    return "cpu"
            except ImportError:
                return "cpu"

        if v == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available")
            except ImportError:
                raise ValueError("PyTorch is not installed")
        elif v == "mps":
            try:
                import torch
                if not torch.backends.mps.is_available():
                    raise ValueError("MPS is not available")
            except ImportError:
                raise ValueError("PyTorch is not installed")
        return v

    @model_validator(mode="after")
    def validate_model_cache_consistency(self) -> "Settings":
        """Ensure model cache directory is within cache directory."""
        if not str(self.model_cache_dir).startswith(str(self.cache_dir)):
            self.model_cache_dir = self.cache_dir / "models"
        return self

    def model_post_init(self, __context: Any) -> None:
        """Create cache directories after initialization."""
        if self.clear_cache_on_startup:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about configured models."""
        return {
            "whisper_model": self.whisper_model,
            "translation_model": self.translation_model,
            "device": self.device,
            "model_cache_dir": str(self.model_cache_dir),
        }

    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration."""
        return {
            "sample_rate": self.sample_rate,
            "chunk_length": self.chunk_length,
            "channels": self.channels,
            "audio_device": self.audio_device,
            "vad_threshold": self.vad_threshold,
            "silence_duration": self.silence_duration,
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [
            "auto", "en", "es", "fr", "de", "it", "pt", "ru",
            "zh", "ja", "ko", "ar", "hi"
        ]

    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language code is supported."""
        return lang_code in self.get_supported_languages()


# Global settings instance
settings = Settings()
