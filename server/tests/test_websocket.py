"""WebSocket tests — live progress and reconnect-with-state-replay.

The three reconnect cases follow the plan:
  - queued job  → reconnect receives queue_position event
  - running job → reconnect receives status + replay of all chunks emitted
  - done job    → reconnect receives final state with transcript URLs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from steno_server import jobs as jobs_mod
from steno_server import worker
from steno_server.config import settings
from steno_server.jobs import JobStatus
from steno_server.worker import JobState


def _seed_state(job_id: str, **kwargs) -> JobState:
    """Insert a synthetic JobState directly into the queue's in-memory map."""
    state = JobState(job_id=job_id, **kwargs)
    worker.queue._states[job_id] = state  # noqa: SLF001
    return state


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    # Reset the worker's in-memory state between tests.
    worker.queue._states.clear()  # noqa: SLF001
    worker.queue._pending_order.clear()  # noqa: SLF001
    yield


@pytest.fixture
def client(isolated):
    from steno_server.server import app

    with TestClient(app) as test_client:
        yield test_client


async def _create_job_in_db(filename: str = "x.wav", language: str = "es"):
    return await jobs_mod.create_job(filename=filename, language=language)


def test_unknown_job_id_closes_with_error(client: TestClient):
    with client.websocket_connect("/ws/jobs/does-not-exist") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        # Server sends close after the error; the test client surfaces it as
        # a disconnect on the next read attempt.


class TestReconnectQueued:
    """Reconnecting to a queued job receives a queue_position event."""

    async def test_returns_queue_position(self, client: TestClient):
        job = await _create_job_in_db()
        _seed_state(job.id, status=JobStatus.QUEUED, queue_position=2)

        with client.websocket_connect(f"/ws/jobs/{job.id}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "queue_position"
            assert msg["job_id"] == job.id
            assert msg["position"] == 2


class TestReconnectRunning:
    """Reconnecting to a phase1_running job receives status + chunk replay."""

    async def test_returns_replay(self, client: TestClient):
        job = await _create_job_in_db()
        chunks = [
            {
                "type": "phase1_chunk",
                "job_id": job.id,
                "text": "hola",
                "start_s": 0.0,
                "end_s": 1.0,
                "is_partial": False,
            },
            {
                "type": "phase1_chunk",
                "job_id": job.id,
                "text": "mundo",
                "start_s": 1.0,
                "end_s": 2.0,
                "is_partial": False,
            },
        ]
        _seed_state(
            job.id,
            status=JobStatus.PHASE1_RUNNING,
            chunks=list(chunks),
            queue_position=0,
        )

        with client.websocket_connect(f"/ws/jobs/{job.id}") as ws:
            msgs: list[dict[str, Any]] = []
            for _ in range(3):  # phase1_started + 2 chunks
                msgs.append(ws.receive_json())
        assert msgs[0]["type"] == "phase1_started"
        assert msgs[1]["type"] == "phase1_chunk"
        assert msgs[1]["text"] == "hola"
        assert msgs[2]["text"] == "mundo"


class TestReconnectCompleted:
    """Reconnecting to a finished job receives the terminal state."""

    async def test_done_replays_phase1_and_phase2_completion(self, client: TestClient):
        job = await _create_job_in_db()
        await jobs_mod.update_status(job.id, JobStatus.DONE)
        _seed_state(
            job.id,
            status=JobStatus.DONE,
            transcript_raw_url=f"/api/jobs/{job.id}/transcript-raw.md",
            transcript_clean_url=f"/api/jobs/{job.id}/transcript-clean.md",
        )

        with client.websocket_connect(f"/ws/jobs/{job.id}") as ws:
            phase1 = ws.receive_json()
            phase2 = ws.receive_json()
        assert phase1["type"] == "phase1_completed"
        assert phase2["type"] == "phase2_completed"
        assert phase2["transcript_url"].endswith("transcript-clean.md")

    async def test_failed_replays_error(self, client: TestClient):
        job = await _create_job_in_db()
        await jobs_mod.update_status(
            job.id,
            JobStatus.FAILED,
            error_message="something exploded",
        )
        _seed_state(
            job.id,
            status=JobStatus.FAILED,
            error={
                "type": "error",
                "job_id": job.id,
                "phase": "phase1",
                "message": "something exploded",
            },
        )

        with client.websocket_connect(f"/ws/jobs/{job.id}") as ws:
            msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["message"] == "something exploded"

    async def test_phase2_failed_replays_phase1_completed_then_error(self, client: TestClient):
        """Critical: phase2_failed clients must still see the raw transcript URL."""
        job = await _create_job_in_db()
        await jobs_mod.update_status(
            job.id,
            JobStatus.PHASE2_FAILED,
            phase2_error_message="diarizer fell over",
        )
        _seed_state(
            job.id,
            status=JobStatus.PHASE2_FAILED,
            transcript_raw_url=f"/api/jobs/{job.id}/transcript-raw.md",
            error={
                "type": "error",
                "job_id": job.id,
                "phase": "phase2",
                "message": "diarizer fell over",
            },
        )

        with client.websocket_connect(f"/ws/jobs/{job.id}") as ws:
            phase1 = ws.receive_json()
            err = ws.receive_json()
        assert phase1["type"] == "phase1_completed"
        assert phase1["transcript_url"].endswith("transcript-raw.md")
        assert err["type"] == "error"
        assert err["phase"] == "phase2"
