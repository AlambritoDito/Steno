"""Structured logging for steno-server.

structlog renders one JSON object per log line. In dev, output goes to stdout;
in any deploy mode, the same JSON is also written to a rotating file at
``$STENO_SERVER_LOG_DIR/server.log`` (default ``~/Library/Logs/steno-server/``)
so the /api/logs/recent endpoint can replay the last 200 lines back to the UI.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import structlog

from .config import settings

_INITIALIZED = False
_LOG_FILE: Path | None = None


def get_log_file() -> Path:
    """Path of the JSON log file. Created on first call."""
    global _LOG_FILE
    if _LOG_FILE is None:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = settings.log_dir / "server.log"
    return _LOG_FILE


def configure_logging() -> None:
    """Configure stdlib logging + structlog. Idempotent."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_file = get_log_file()

    json_formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(json_formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(json_formatter)

    root = logging.getLogger()
    # Remove any pre-existing handlers (uvicorn often installs its own)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _INITIALIZED = True
    structlog.get_logger("steno_server").info(
        "logging_configured",
        log_file=str(log_file),
        level=settings.log_level,
    )


def get_logger(name: str = "steno_server") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; configures on first call."""
    if not _INITIALIZED:
        configure_logging()
    return structlog.get_logger(name)
