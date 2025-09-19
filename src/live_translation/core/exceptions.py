"""Custom exceptions for the live translation system."""

import traceback
from typing import Any, Dict, Optional, Type


class LiveTranslationError(Exception):
    """Base exception for all live translation errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.cause = cause

        # Store stack trace for debugging
        self.stack_trace = traceback.format_stack()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    @classmethod
    def from_exception(
        cls: Type["LiveTranslationError"],
        exc: Exception,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        **details: Any,
    ) -> "LiveTranslationError":
        """Create a LiveTranslationError from another exception."""
        return cls(
            message=message or str(exc),
            error_code=error_code,
            details=details,
            cause=exc,
        )

    def __str__(self) -> str:
        """String representation of the exception."""
        base = f"{self.error_code}: {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"
        return base


class ConfigurationError(LiveTranslationError):
    """Raised when there are configuration issues."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, **kwargs)


class ModelLoadError(LiveTranslationError):
    """Raised when model loading fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if model_type:
            details["model_type"] = model_type
        super().__init__(message, details=details, **kwargs)


class AudioCaptureError(LiveTranslationError):
    """Raised when audio capture fails."""

    def __init__(
        self,
        message: str,
        device_id: Optional[int] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if device_id is not None:
            details["device_id"] = device_id
        if sample_rate:
            details["sample_rate"] = sample_rate
        super().__init__(message, details=details, **kwargs)


class AudioProcessingError(LiveTranslationError):
    """Raised when audio processing fails."""

    def __init__(
        self,
        message: str,
        audio_length: Optional[int] = None,
        processing_stage: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if audio_length is not None:
            details["audio_length"] = audio_length
        if processing_stage:
            details["processing_stage"] = processing_stage
        super().__init__(message, details=details, **kwargs)


class TranslationError(LiveTranslationError):
    """Raised when translation fails."""

    def __init__(
        self,
        message: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if source_lang:
            details["source_lang"] = source_lang
        if target_lang:
            details["target_lang"] = target_lang
        if text_length is not None:
            details["text_length"] = text_length
        super().__init__(message, details=details, **kwargs)


class LanguageDetectionError(LiveTranslationError):
    """Raised when language detection fails."""

    def __init__(
        self,
        message: str,
        text_sample: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if text_sample:
            # Only store a sample for debugging
            details["text_sample"] = (
                text_sample[:100] + "..." if len(text_sample) > 100 else text_sample
            )
        super().__init__(message, details=details, **kwargs)


class UnsupportedLanguageError(LiveTranslationError):
    """Raised when an unsupported language is requested."""

    def __init__(
        self,
        message: str,
        language_code: Optional[str] = None,
        supported_languages: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if language_code:
            details["language_code"] = language_code
        if supported_languages:
            details["supported_languages"] = supported_languages
        super().__init__(message, details=details, **kwargs)


class InvalidInputError(LiveTranslationError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        input_type: Optional[str] = None,
        validation_rule: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if input_type:
            details["input_type"] = input_type
        if validation_rule:
            details["validation_rule"] = validation_rule
        super().__init__(message, details=details, **kwargs)


class ResourceNotFoundError(LiveTranslationError):
    """Raised when required resources are not found."""

    def __init__(
        self,
        message: str,
        resource_path: Optional[str] = None,
        resource_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if resource_path:
            details["resource_path"] = resource_path
        if resource_type:
            details["resource_type"] = resource_type
        super().__init__(message, details=details, **kwargs)


class PerformanceError(LiveTranslationError):
    """Raised when performance requirements are not met."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        duration_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        if duration_ms is not None:
            details["duration_ms"] = duration_ms
        if threshold_ms is not None:
            details["threshold_ms"] = threshold_ms
        super().__init__(message, details=details, **kwargs)


class DeviceError(LiveTranslationError):
    """Raised when device-related operations fail."""

    def __init__(
        self,
        message: str,
        device_type: Optional[str] = None,
        device_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if device_type:
            details["device_type"] = device_type
        if device_id:
            details["device_id"] = device_id
        super().__init__(message, details=details, **kwargs)


# Helper functions for common error scenarios
def handle_model_load_error(exc: Exception, model_name: str, model_type: str) -> None:
    """Handle model loading errors consistently."""
    raise ModelLoadError.from_exception(
        exc,
        message=f"Failed to load {model_type} model: {model_name}",
        model_name=model_name,
        model_type=model_type,
    )


def handle_translation_error(
    exc: Exception,
    source_lang: str,
    target_lang: str,
    text_length: int,
) -> None:
    """Handle translation errors consistently."""
    raise TranslationError.from_exception(
        exc,
        message=f"Translation failed ({source_lang} → {target_lang})",
        source_lang=source_lang,
        target_lang=target_lang,
        text_length=text_length,
    )


def handle_audio_error(exc: Exception, stage: str, **details: Any) -> None:
    """Handle audio processing errors consistently."""
    raise AudioProcessingError.from_exception(
        exc,
        message=f"Audio processing failed at {stage}",
        processing_stage=stage,
        **details,
    )
