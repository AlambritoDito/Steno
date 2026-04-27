"""SQLite-backed job manager.

Schema lives in ``_SCHEMA``. The DB file path comes from
``settings.storage_dir / "jobs.db"`` so it follows the storage root in tests.

Crash recovery is the load-bearing piece here: ``recover_orphaned_jobs()``
runs at startup and reconciles in-flight states with the fact that the
worker is dead. The rules are granular per-phase so that artifacts from
completed phases stay accessible to the user.
"""

from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from .config import settings
from .logging_setup import get_logger
from .storage import delete_job_files

logger = get_logger(__name__)


class JobStatus(str, enum.Enum):
    """Lifecycle states.

    The state machine flows: queued → phase1_running → phase1_done →
    phase2_running → done. Failures land in ``failed`` (Phase 1 broke) or
    ``phase2_failed`` (Phase 2 broke; raw md still downloadable).
    """

    QUEUED = "queued"
    PHASE1_RUNNING = "phase1_running"
    PHASE1_DONE = "phase1_done"
    PHASE2_RUNNING = "phase2_running"
    DONE = "done"
    FAILED = "failed"
    PHASE2_FAILED = "phase2_failed"


_TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED, JobStatus.PHASE2_FAILED}


@dataclass
class PipelineOptions:
    """User-configurable per-job toggles."""

    enable_denoise: bool = True
    enable_diarization: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "enable_denoise": self.enable_denoise,
                "enable_diarization": self.enable_diarization,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "PipelineOptions":
        if not raw:
            return cls()
        data = json.loads(raw)
        return cls(
            enable_denoise=bool(data.get("enable_denoise", True)),
            enable_diarization=bool(data.get("enable_diarization", True)),
        )


@dataclass
class Job:
    """In-memory shape of a row in the jobs table."""

    id: str
    created_at: datetime
    updated_at: datetime
    status: JobStatus
    language: str
    original_filename: str
    audio_duration_seconds: float | None = None
    phase1_started_at: datetime | None = None
    phase1_completed_at: datetime | None = None
    phase2_started_at: datetime | None = None
    phase2_completed_at: datetime | None = None
    error_message: str | None = None
    phase2_error_message: str | None = None
    options: PipelineOptions = field(default_factory=PipelineOptions)

    def to_dict(self) -> dict[str, Any]:
        def iso(dt: datetime | None) -> str | None:
            return dt.isoformat() if dt else None

        return {
            "id": self.id,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "status": self.status.value,
            "language": self.language,
            "original_filename": self.original_filename,
            "audio_duration_seconds": self.audio_duration_seconds,
            "phase1_started_at": iso(self.phase1_started_at),
            "phase1_completed_at": iso(self.phase1_completed_at),
            "phase2_started_at": iso(self.phase2_started_at),
            "phase2_completed_at": iso(self.phase2_completed_at),
            "error_message": self.error_message,
            "phase2_error_message": self.phase2_error_message,
            "options": {
                "enable_denoise": self.options.enable_denoise,
                "enable_diarization": self.options.enable_diarization,
            },
        }


# ---------------------------------------------------------------------------
# Schema and connection management
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    language TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    audio_duration_seconds REAL,
    phase1_started_at TEXT,
    phase1_completed_at TEXT,
    phase2_started_at TEXT,
    phase2_completed_at TEXT,
    error_message TEXT,
    phase2_error_message TEXT,
    options_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def _db_path() -> Path:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings.storage_dir / "jobs.db"


async def init_db() -> None:
    """Create the schema if it doesn't exist. Idempotent."""
    async with aiosqlite.connect(_db_path()) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_job(row: aiosqlite.Row) -> Job:
    return Job(
        id=row["id"],
        created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
        updated_at=_parse_dt(row["updated_at"]),  # type: ignore[arg-type]
        status=JobStatus(row["status"]),
        language=row["language"],
        original_filename=row["original_filename"],
        audio_duration_seconds=row["audio_duration_seconds"],
        phase1_started_at=_parse_dt(row["phase1_started_at"]),
        phase1_completed_at=_parse_dt(row["phase1_completed_at"]),
        phase2_started_at=_parse_dt(row["phase2_started_at"]),
        phase2_completed_at=_parse_dt(row["phase2_completed_at"]),
        error_message=row["error_message"],
        phase2_error_message=row["phase2_error_message"],
        options=PipelineOptions.from_json(row["options_json"]),
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def create_job(
    *,
    filename: str,
    language: str,
    options: PipelineOptions | None = None,
) -> Job:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    opts = options or PipelineOptions()

    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """
            INSERT INTO jobs (id, created_at, updated_at, status, language,
                              original_filename, options_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, now, now, JobStatus.QUEUED.value, language, filename, opts.to_json()),
        )
        await db.commit()

    return Job(
        id=job_id,
        created_at=datetime.fromisoformat(now),
        updated_at=datetime.fromisoformat(now),
        status=JobStatus.QUEUED,
        language=language,
        original_filename=filename,
        options=opts,
    )


async def get_job(job_id: str) -> Job | None:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
    if not row:
        return None
    return _row_to_job(row)


async def list_jobs(limit: int = 50) -> list[Job]:
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [_row_to_job(r) for r in rows]


async def update_status(
    job_id: str,
    status: JobStatus,
    *,
    audio_duration_seconds: float | None = None,
    phase1_started_at: datetime | None = None,
    phase1_completed_at: datetime | None = None,
    phase2_started_at: datetime | None = None,
    phase2_completed_at: datetime | None = None,
    error_message: str | None = None,
    phase2_error_message: str | None = None,
) -> None:
    """Update mutable fields. Only fields explicitly passed are touched."""
    fields = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status.value, _now_iso()]

    def add(name: str, value: Any) -> None:
        fields.append(f"{name} = ?")
        values.append(value)

    if audio_duration_seconds is not None:
        add("audio_duration_seconds", audio_duration_seconds)
    if phase1_started_at is not None:
        add("phase1_started_at", phase1_started_at.isoformat())
    if phase1_completed_at is not None:
        add("phase1_completed_at", phase1_completed_at.isoformat())
    if phase2_started_at is not None:
        add("phase2_started_at", phase2_started_at.isoformat())
    if phase2_completed_at is not None:
        add("phase2_completed_at", phase2_completed_at.isoformat())
    if error_message is not None:
        add("error_message", error_message)
    if phase2_error_message is not None:
        add("phase2_error_message", phase2_error_message)

    values.append(job_id)
    sql = f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?"  # noqa: S608

    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(sql, values)
        await db.commit()


async def delete_job(job_id: str) -> bool:
    """Remove the job row and its file directory.

    Returns True if a row was deleted.
    """
    async with aiosqlite.connect(_db_path()) as db:
        cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
        deleted = cursor.rowcount > 0

    delete_job_files(job_id)
    return deleted


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


@dataclass
class RecoveryReport:
    """Summary of what recover_orphaned_jobs() did. Useful for tests + logs."""

    failed_phase1: int = 0
    failed_phase2: int = 0
    left_phase1_done: int = 0


async def recover_orphaned_jobs() -> RecoveryReport:
    """Reconcile in-flight statuses at server boot.

    Granular per-phase rules:
      - phase1_running → failed, with error message naming the phase.
      - phase1_done with phase2 expected but never started → leave as
        phase1_done. The raw md is on disk and the user can re-submit
        if they want Phase 2.
      - phase2_running → phase2_failed (NOT failed), so the raw md from
        Phase 1 stays accessible. phase2_error_message is populated.
    """
    report = RecoveryReport()
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, status FROM jobs WHERE status IN (?, ?, ?)",
            (
                JobStatus.PHASE1_RUNNING.value,
                JobStatus.PHASE1_DONE.value,
                JobStatus.PHASE2_RUNNING.value,
            ),
        )
        rows = await cursor.fetchall()

    for row in rows:
        job_id = row["id"]
        status = JobStatus(row["status"])
        if status is JobStatus.PHASE1_RUNNING:
            await update_status(
                job_id,
                JobStatus.FAILED,
                error_message="Server restarted during Phase 1 processing",
            )
            report.failed_phase1 += 1
            logger.warning("recovered_phase1_running", job_id=job_id)
        elif status is JobStatus.PHASE2_RUNNING:
            await update_status(
                job_id,
                JobStatus.PHASE2_FAILED,
                phase2_error_message="Server restarted during Phase 2 processing",
            )
            report.failed_phase2 += 1
            logger.warning("recovered_phase2_running", job_id=job_id)
        elif status is JobStatus.PHASE1_DONE:
            # Intentionally a no-op on the row — the raw md is on disk and
            # the user can re-submit. We only count for telemetry.
            report.left_phase1_done += 1
            logger.warning("recovery_phase1_done_left_alone", job_id=job_id)

    return report


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------


def is_terminal(status: JobStatus) -> bool:
    return status in _TERMINAL_STATUSES
