"""
ARTEMIS Logging

Structured logging configuration using structlog.
Provides consistent, context-rich logging across the framework.
"""

import logging
import sys
from typing import Any, Literal, cast

import structlog
from structlog.typing import Processor

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    json_format: bool = False,
    include_timestamps: bool = True,
) -> None:
    """
    Configure structured logging for ARTEMIS.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON logs (for production). If False, colored console output.
        include_timestamps: Whether to include timestamps in log output.
    """
    # Shared processors for all output formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamps:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        # JSON format for production/log aggregation
        shared_processors.append(structlog.processors.format_exc_info)
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Colored console output for development
        shared_processors.append(
            structlog.dev.set_exc_info,
        )
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level))

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        A structlog BoundLogger instance.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        >>> bind_context(debate_id="abc123", agent="Proponent")
        >>> log.info("generating argument")  # Will include debate_id and agent
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Remove context variables from the logging context.

    Args:
        *keys: Keys to remove from the context.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


class LogContext:
    """
    Context manager for temporary logging context.

    Example:
        >>> with LogContext(debate_id="abc123", round=1):
        ...     log.info("starting round")  # Includes debate_id and round
        >>> log.info("after context")  # No longer includes those fields
    """

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


# Convenience logger for quick imports
log = get_logger("artemis")
