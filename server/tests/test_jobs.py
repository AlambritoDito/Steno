"""Tests for steno_server.jobs.

Covers basic CRUD, the per-phase crash-recovery rules, and the
PipelineOptions JSON round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from steno_server import jobs
from steno_server.config import settings
from steno_server.jobs import JobStatus, PipelineOptions, RecoveryReport


@pytest.fixture(autouse=True)
async def isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test isolated SQLite DB and storage_dir."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    await jobs.init_db()


# ---------------------------------------------------------------------------
# PipelineOptions
# ---------------------------------------------------------------------------


class TestPipelineOptions:
    def test_default_values(self):
        opts = PipelineOptions()
        assert opts.enable_denoise is True
        assert opts.enable_diarization is True

    def test_round_trip(self):
        opts = PipelineOptions(enable_denoise=False, enable_diarization=True)
        recovered = PipelineOptions.from_json(opts.to_json())
        assert recovered.enable_denoise is False
        assert recovered.enable_diarization is True

    def test_from_json_handles_none(self):
        assert PipelineOptions.from_json(None) == PipelineOptions()

    def test_from_json_handles_partial(self):
        opts = PipelineOptions.from_json('{"enable_denoise": false}')
        assert opts.enable_denoise is False
        assert opts.enable_diarization is True  # default


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    async def test_create_job_returns_queued(self):
        job = await jobs.create_job(filename="meeting.m4a", language="es")
        assert job.status is JobStatus.QUEUED
        assert job.original_filename == "meeting.m4a"
        assert job.language == "es"
        assert job.id  # UUID
        assert job.error_message is None

    async def test_get_job_round_trip(self):
        created = await jobs.create_job(filename="x.wav", language="en")
        loaded = await jobs.get_job(created.id)
        assert loaded is not None
        assert loaded.id == created.id
        assert loaded.original_filename == "x.wav"
        assert loaded.language == "en"

    async def test_get_unknown_returns_none(self):
        assert await jobs.get_job("does-not-exist") is None

    async def test_list_returns_newest_first(self):
        a = await jobs.create_job(filename="a.wav", language="es")
        b = await jobs.create_job(filename="b.wav", language="es")
        c = await jobs.create_job(filename="c.wav", language="es")
        listed = await jobs.list_jobs()
        assert [j.id for j in listed[:3]] == [c.id, b.id, a.id]

    async def test_update_status_transitions(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(
            job.id,
            JobStatus.PHASE1_RUNNING,
            phase1_started_at=datetime.now(UTC),
        )
        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.PHASE1_RUNNING
        assert loaded.phase1_started_at is not None

    async def test_update_status_only_touches_provided_fields(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        first = datetime.now(UTC)
        await jobs.update_status(
            job.id,
            JobStatus.PHASE1_RUNNING,
            phase1_started_at=first,
        )
        # Now flip status without passing started_at; it must persist.
        await jobs.update_status(job.id, JobStatus.PHASE1_DONE)
        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.PHASE1_DONE
        assert loaded.phase1_started_at is not None
        assert loaded.phase1_started_at == first

    async def test_phase2_failed_carries_separate_error_message(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(
            job.id,
            JobStatus.PHASE2_FAILED,
            phase2_error_message="diarizer fell over",
        )
        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.PHASE2_FAILED
        assert loaded.phase2_error_message == "diarizer fell over"
        assert loaded.error_message is None  # the regular field stays clean

    async def test_delete_removes_row(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        assert await jobs.delete_job(job.id) is True
        assert await jobs.get_job(job.id) is None

    async def test_delete_unknown_returns_false(self):
        assert await jobs.delete_job("ghost") is False


# ---------------------------------------------------------------------------
# Crash recovery (granular per-phase rules)
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    async def test_no_orphans_yields_zeros(self):
        await jobs.create_job(filename="ok.wav", language="es")  # queued — not orphaned
        report = await jobs.recover_orphaned_jobs()
        assert report == RecoveryReport()

    async def test_phase1_running_marked_failed(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(job.id, JobStatus.PHASE1_RUNNING)

        report = await jobs.recover_orphaned_jobs()
        assert report.failed_phase1 == 1
        assert report.failed_phase2 == 0
        assert report.left_phase1_done == 0

        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.FAILED
        assert "Phase 1" in (loaded.error_message or "")

    async def test_phase1_done_left_alone(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(job.id, JobStatus.PHASE1_DONE)

        report = await jobs.recover_orphaned_jobs()
        assert report.failed_phase1 == 0
        assert report.failed_phase2 == 0
        assert report.left_phase1_done == 1

        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.PHASE1_DONE
        assert loaded.error_message is None
        assert loaded.phase2_error_message is None

    async def test_phase2_running_marked_phase2_failed(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(job.id, JobStatus.PHASE2_RUNNING)

        report = await jobs.recover_orphaned_jobs()
        assert report.failed_phase1 == 0
        assert report.failed_phase2 == 1
        assert report.left_phase1_done == 0

        loaded = await jobs.get_job(job.id)
        # Critically: NOT failed (so raw md stays accessible).
        assert loaded.status is JobStatus.PHASE2_FAILED
        assert "Phase 2" in (loaded.phase2_error_message or "")
        assert loaded.error_message is None  # separate column

    async def test_done_jobs_unaffected(self):
        job = await jobs.create_job(filename="x.wav", language="es")
        await jobs.update_status(job.id, JobStatus.DONE)

        report = await jobs.recover_orphaned_jobs()
        assert report == RecoveryReport()

        loaded = await jobs.get_job(job.id)
        assert loaded.status is JobStatus.DONE

    async def test_mixed_orphans(self):
        a = await jobs.create_job(filename="a.wav", language="es")
        b = await jobs.create_job(filename="b.wav", language="es")
        c = await jobs.create_job(filename="c.wav", language="es")
        d = await jobs.create_job(filename="d.wav", language="es")
        await jobs.update_status(a.id, JobStatus.PHASE1_RUNNING)
        await jobs.update_status(b.id, JobStatus.PHASE1_DONE)
        await jobs.update_status(c.id, JobStatus.PHASE2_RUNNING)
        # d stays QUEUED, which is not an orphan state.

        report = await jobs.recover_orphaned_jobs()
        assert report.failed_phase1 == 1
        assert report.failed_phase2 == 1
        assert report.left_phase1_done == 1

        assert (await jobs.get_job(a.id)).status is JobStatus.FAILED
        assert (await jobs.get_job(b.id)).status is JobStatus.PHASE1_DONE
        assert (await jobs.get_job(c.id)).status is JobStatus.PHASE2_FAILED
        assert (await jobs.get_job(d.id)).status is JobStatus.QUEUED


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


def test_is_terminal_classification():
    for s in (JobStatus.DONE, JobStatus.FAILED, JobStatus.PHASE2_FAILED):
        assert jobs.is_terminal(s) is True
    for s in (
        JobStatus.QUEUED,
        JobStatus.PHASE1_RUNNING,
        JobStatus.PHASE1_DONE,
        JobStatus.PHASE2_RUNNING,
    ):
        assert jobs.is_terminal(s) is False
