"""FastAPI app for steno-server.

Phase 1 surface: only ``/api/health``. Subsequent phases extend this module
with REST endpoints, the WebSocket route, and frontend asset mounting.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from . import __version__
from .logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

_BOOT_TIME: float = 0.0


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str
    version: str
    uptime_seconds: float
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and record boot time at startup."""
    global _BOOT_TIME
    configure_logging()
    _BOOT_TIME = time.monotonic()
    logger.info("server_starting", version=__version__)
    yield
    logger.info("server_stopping", uptime_seconds=round(time.monotonic() - _BOOT_TIME, 2))


app = FastAPI(
    title="Steno Server",
    version=__version__,
    description="LAN/Tailscale transcription service backed by mlx-whisper.",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe.

    ``model_loaded`` is always False in Phase 1; subsequent phases flip it to
    True once a Whisper model has been pulled into memory by the worker.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.monotonic() - _BOOT_TIME, 2),
        model_loaded=False,
    )
