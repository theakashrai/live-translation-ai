"""
Structured logging configuration using structlog.

This module provides the configuration and setup for structured logging
using structlog for the application.
"""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import Processor


def configure_logging(
    json_logs: bool = True,
    log_level: str = "INFO",
    cache_logger_on_first_use: bool = True,
) -> None:
    """
    Configure structlog for the application.

    Args:
        json_logs: Whether to output logs as JSON (True) or human-readable format (False)
        log_level: The minimum log level to output
        cache_logger_on_first_use: Whether to cache loggers for performance
    """

    # Common processors for all configurations
    shared_processors: list[Processor] = [
        # Merge context vars into event dict
        structlog.contextvars.merge_contextvars,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add call site information
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ],
            additional_ignores=["logging", "__main__"],
        ),
        # Process positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Process exception info
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure renderer based on output format
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable format for development
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
            exception_formatter=structlog.dev.RichTracebackFormatter(
                show_locals=True, max_frames=5
            ),
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=cache_logger_on_first_use,
    )

    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True,  # Force reconfiguration
    )

    # Add structlog processor to standard library logger
    logging.getLogger().handlers[0].setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )


def get_logger(
    name: Optional[str] = None, **initial_values: Any
) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger instance.

    Args:
        name: Logger name (typically __name__)
        **initial_values: Initial context values to bind to the logger

    Returns:
        A bound logger instance
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger


def temporary_context(**values: Any):
    """
    Context manager for temporarily binding values to the logger context.

    Usage:
        with temporary_context(request_id="123"):
            logger.info("Processing request")
    """
    return structlog.contextvars.bound_contextvars(**values)
