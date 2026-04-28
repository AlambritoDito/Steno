"""Filesystem layout for jobs.

Each job gets its own directory under ``settings.storage_dir``:

    {storage_dir}/{job_id}/
        source.{ext}            uploaded original (any supported format)
        normalized.wav          16 kHz mono PCM 16-bit
        denoised.wav            (only if Phase 2 ran)
        transcript-raw.md       written at end of Phase 1
        transcript-clean.md     written at end of Phase 2

The cleanup task purges directories older than ``settings.job_retention_hours``;
it never touches a directory whose creation time isn't past the cutoff,
which is a safe approximation of "active job" given our single-worker FIFO.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import aiofiles
from fastapi import UploadFile

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)

TranscriptKind = Literal["raw", "clean"]


def get_job_dir(job_id: str, *, create: bool = False) -> Path:
    path = settings.storage_dir / job_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_source_path(job_id: str, extension: str) -> Path:
    """Where to write the original upload. ``extension`` includes the dot."""
    if not extension.startswith("."):
        extension = f".{extension}"
    return get_job_dir(job_id) / f"source{extension}"


def get_normalized_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "normalized.wav"


def get_denoised_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "denoised.wav"


def get_transcript_path(job_id: str, kind: TranscriptKind) -> Path:
    return get_job_dir(job_id) / f"transcript-{kind}.md"


async def save_upload(job_id: str, upload: UploadFile) -> Path:
    """Persist an uploaded file under the job's directory.

    Returns the path to the saved source file. The job directory is created
    if it didn't exist; the caller is responsible for creating the matching
    job row before invoking this.
    """
    if not upload.filename:
        raise ValueError("Upload has no filename")
    extension = Path(upload.filename).suffix.lower() or ".bin"

    job_dir = get_job_dir(job_id, create=True)
    target = job_dir / f"source{extension}"

    async with aiofiles.open(target, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)
    await upload.close()
    return target


def list_job_dirs() -> list[Path]:
    """Return all per-job directories under storage_dir, oldest first."""
    if not settings.storage_dir.exists():
        return []
    dirs = [p for p in settings.storage_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime)
    return dirs


def delete_job_files(job_id: str) -> bool:
    """Remove the job's directory tree. Returns True if anything was deleted."""
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        return False
    shutil.rmtree(job_dir)
    return True


async def cleanup_old_jobs(retention_hours: int | None = None) -> int:
    """Delete job directories older than the retention cutoff.

    Returns the number of directories removed. Never deletes a directory
    that doesn't predate the cutoff, even if it has no metadata.
    """
    hours = retention_hours if retention_hours is not None else settings.job_retention_hours
    cutoff = time.time() - hours * 3600
    deleted = 0
    for job_dir in list_job_dirs():
        try:
            if job_dir.stat().st_mtime < cutoff:
                shutil.rmtree(job_dir)
                deleted += 1
        except OSError as exc:
            logger.warning("cleanup_failed", job_dir=str(job_dir), error=str(exc))
    if deleted:
        logger.info("cleanup_old_jobs", deleted=deleted, retention_hours=hours)
    return deleted


async def run_cleanup_loop(interval_seconds: int = 3600) -> None:
    """Run cleanup_old_jobs() forever, at the given interval.

    Started as an asyncio task in the FastAPI lifespan. Sleeps in 1-minute
    increments so cancellation propagates quickly during shutdown.
    """
    while True:
        try:
            await cleanup_old_jobs()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("cleanup_loop_iteration_failed", error=str(exc), exc_info=True)
        # Sleep in slices so shutdown isn't blocked.
        slept = 0
        while slept < interval_seconds:
            await asyncio.sleep(min(60, interval_seconds - slept))
            slept += 60


def cutoff_iso(retention_hours: int | None = None) -> str:
    """Helpful for logs: ISO timestamp of the current retention cutoff."""
    hours = retention_hours if retention_hours is not None else settings.job_retention_hours
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    return cutoff.isoformat()
