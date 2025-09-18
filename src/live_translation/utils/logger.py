"""Structured logging configuration using structlog."""

from typing import Any

from live_translation.core.config import settings
from live_translation.core.structured_logging import configure_logging
from live_translation.core.structured_logging import get_logger as _get_logger

# Initialize logging on module import
configure_logging(
    json_logs=settings.log_format == "json",
    log_level=settings.log_level,
)


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name."""
    return _get_logger(name)


def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """Log performance metrics for an operation."""
    logger = _get_logger(__name__)
    logger.info(
        f"PERF: {operation} took {duration_ms:.2f}ms",
        operation=operation,
        duration_ms=duration_ms,
        performance=True,
        **kwargs,
    )


def log_model_info(model_name: str, model_size: str, load_time_ms: float) -> None:
    """Log model loading information."""
    logger = _get_logger(__name__)
    logger.info(
        f"Loaded model {model_name} (size: {model_size}) in {load_time_ms:.2f}ms",
        model_name=model_name,
        model_size=model_size,
        load_time_ms=load_time_ms,
    )


def log_translation_metrics(
    source_lang: str,
    target_lang: str,
    text_length: int,
    processing_time_ms: float,
    confidence: float,
) -> None:
    """Log translation metrics."""
    logger = _get_logger(__name__)
    logger.info(
        f"Translation {source_lang}->{target_lang}: {text_length} chars, "
        f"{processing_time_ms:.2f}ms, confidence {confidence:.3f}",
        source_lang=source_lang,
        target_lang=target_lang,
        text_length=text_length,
        processing_time_ms=processing_time_ms,
        confidence=confidence,
    )


def log_audio_metrics(
    sample_rate: int, duration_ms: float, processing_time_ms: float
) -> None:
    """Log audio processing metrics."""
    logger = _get_logger(__name__)
    logger.info(
        f"Audio processed: {sample_rate}Hz, {duration_ms:.1f}ms duration, "
        f"processed in {processing_time_ms:.2f}ms",
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        processing_time_ms=processing_time_ms,
    )
