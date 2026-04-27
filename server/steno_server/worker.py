"""Single-worker FIFO queue plus per-job WebSocket state.

mlx-whisper is not thread-safe and we don't want to OOM the GPU, so the
server runs at most one transcription at a time. New uploads enqueue
behind any in-flight job; the queue worker pulls them in order.

Per-job WS state:
    For each known job_id, we keep a JobState that records the latest
    status, the queue position (if queued), the ordered list of phase1
    chunks emitted so far (for replay on reconnect), and the set of live
    WebSocket connections.

    On WS connect, the server immediately sends the current state to the
    client (via reconnect_payload()) before any new live events flow.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import WebSocket

from .config import settings
from .jobs import JobStatus, get_job, update_status
from .logging_setup import get_logger
from .pipeline import run_phase_1
from .transcriber import Transcriber

logger = get_logger(__name__)


@dataclass
class QueuedJob:
    """A job ticket in the FIFO queue."""

    job_id: str
    source_path: Path
    language: str


@dataclass
class JobState:
    """All the state we need to replay a job's lifetime to a (re)connecting WS."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    queue_position: int | None = None  # None once running
    chunks: list[dict[str, Any]] = field(default_factory=list)
    last_event: dict[str, Any] | None = None
    transcript_raw_url: str | None = None
    transcript_clean_url: str | None = None
    error: dict[str, Any] | None = None
    listeners: set[WebSocket] = field(default_factory=set)


class JobQueue:
    """FIFO transcription queue with one worker.

    Methods are async and safe to call from the FastAPI request handlers
    and from the worker loop simultaneously. The queue itself is an
    asyncio.Queue; queue_positions are reconstructed by snapshotting the
    contents at the moment of inspection (see _recompute_positions).
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedJob] = asyncio.Queue()
        self._states: dict[str, JobState] = {}
        # Tracks pending (queued/running) job_ids in submission order so we
        # can recompute positions when a job completes.
        self._pending_order: list[str] = []
        self._running_id: str | None = None
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._transcriber = Transcriber()
        self._stopping = asyncio.Event()

    # -- Public API ----------------------------------------------------------

    async def enqueue(self, job: QueuedJob) -> int:
        """Put a job on the queue and return its position (0 = running)."""
        async with self._lock:
            self._states.setdefault(job.job_id, JobState(job_id=job.job_id))
            self._pending_order.append(job.job_id)
            await self._queue.put(job)
            position = self._position_for(job.job_id)
            self._states[job.job_id].queue_position = position
        await self._broadcast_to_listeners(
            job.job_id,
            {"type": "queue_position", "job_id": job.job_id, "position": position},
        )
        return position

    def get_state(self, job_id: str) -> JobState | None:
        return self._states.get(job_id)

    async def register_listener(self, job_id: str, websocket: WebSocket) -> None:
        """Add a websocket listener for a job. Caller has already accepted."""
        state = self._states.get(job_id)
        if state is None:
            # The job exists in the DB but not yet in our in-memory state
            # (e.g. server boot, or a job from before this process started).
            # Hydrate a minimal state from the DB.
            await self._hydrate_state(job_id)
            state = self._states.get(job_id)
        if state is None:
            return
        state.listeners.add(websocket)

    def unregister_listener(self, job_id: str, websocket: WebSocket) -> None:
        state = self._states.get(job_id)
        if state and websocket in state.listeners:
            state.listeners.discard(websocket)

    def reconnect_payload(self, job_id: str) -> list[dict[str, Any]]:
        """Build the list of events a reconnecting client should receive
        before live updates resume.

        Returns an empty list if we have no record of the job.
        """
        state = self._states.get(job_id)
        if state is None:
            return []

        out: list[dict[str, Any]] = []
        if state.status is JobStatus.QUEUED:
            position = state.queue_position if state.queue_position is not None else 0
            out.append(
                {"type": "queue_position", "job_id": job_id, "position": position}
            )
            return out

        if state.status is JobStatus.PHASE1_RUNNING:
            out.append({"type": "phase1_started", "job_id": job_id})
            out.extend(state.chunks)
            return out

        if state.status is JobStatus.PHASE1_DONE:
            out.append({"type": "phase1_started", "job_id": job_id})
            out.extend(state.chunks)
            out.append(
                {
                    "type": "phase1_completed",
                    "job_id": job_id,
                    "transcript_url": state.transcript_raw_url
                    or f"/api/jobs/{job_id}/transcript-raw.md",
                }
            )
            return out

        if state.status is JobStatus.DONE:
            out.append(
                {
                    "type": "phase1_completed",
                    "job_id": job_id,
                    "transcript_url": state.transcript_raw_url
                    or f"/api/jobs/{job_id}/transcript-raw.md",
                }
            )
            out.append(
                {
                    "type": "phase2_completed",
                    "job_id": job_id,
                    "transcript_url": state.transcript_clean_url
                    or f"/api/jobs/{job_id}/transcript-clean.md",
                }
            )
            return out

        if state.status is JobStatus.FAILED:
            out.append(
                state.error
                or {"type": "error", "job_id": job_id, "message": "Job failed"}
            )
            return out

        if state.status is JobStatus.PHASE2_FAILED:
            out.append(
                {
                    "type": "phase1_completed",
                    "job_id": job_id,
                    "transcript_url": state.transcript_raw_url
                    or f"/api/jobs/{job_id}/transcript-raw.md",
                }
            )
            out.append(
                state.error
                or {
                    "type": "error",
                    "job_id": job_id,
                    "message": "Phase 2 failed",
                    "phase": "phase2",
                }
            )
            return out

        return out

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._stopping.clear()
            self._worker_task = asyncio.create_task(self._worker_loop(), name="job-worker")
            logger.info("job_queue_worker_started")

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._stopping.set()
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            logger.info("job_queue_worker_stopped")

    # -- Internals -----------------------------------------------------------

    def _position_for(self, job_id: str) -> int:
        """0 = running, 1 = next, ..."""
        if job_id not in self._pending_order:
            return -1
        return self._pending_order.index(job_id)

    async def _hydrate_state(self, job_id: str) -> None:
        job = await get_job(job_id)
        if job is None:
            return
        state = self._states.get(job_id)
        if state is None:
            state = JobState(job_id=job_id, status=job.status)
            self._states[job_id] = state
        else:
            state.status = job.status
        # Populate transcript URLs based on terminal status — used by the
        # reconnect payload.
        if job.status in (JobStatus.PHASE1_DONE, JobStatus.DONE, JobStatus.PHASE2_FAILED):
            state.transcript_raw_url = f"/api/jobs/{job_id}/transcript-raw.md"
        if job.status is JobStatus.DONE:
            state.transcript_clean_url = f"/api/jobs/{job_id}/transcript-clean.md"
        if job.status is JobStatus.PHASE2_FAILED and job.phase2_error_message:
            state.error = {
                "type": "error",
                "job_id": job_id,
                "phase": "phase2",
                "message": job.phase2_error_message,
            }
        if job.status is JobStatus.FAILED and job.error_message:
            state.error = {
                "type": "error",
                "job_id": job_id,
                "phase": "phase1",
                "message": job.error_message,
            }

    async def _broadcast_to_listeners(
        self, job_id: str, payload: dict[str, Any]
    ) -> None:
        state = self._states.get(job_id)
        if not state:
            return
        dead: list[WebSocket] = []
        for ws in list(state.listeners):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 — connection died, drop it
                dead.append(ws)
        for ws in dead:
            state.listeners.discard(ws)

    async def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            queued = await self._queue.get()
            await self._run_job(queued)

    async def _run_job(self, queued: QueuedJob) -> None:
        async with self._lock:
            self._running_id = queued.job_id
            state = self._states.setdefault(queued.job_id, JobState(job_id=queued.job_id))
            state.status = JobStatus.PHASE1_RUNNING
            state.queue_position = 0
            # Recompute positions for the rest of the queue (everyone moves up).
            await self._broadcast_position_updates_locked()

        async def progress(payload: dict[str, Any]) -> None:
            kind = payload.get("type")
            st = self._states[queued.job_id]
            if kind == "phase1_chunk":
                st.chunks.append(payload)
            if kind == "phase1_completed":
                st.status = JobStatus.PHASE1_DONE
                st.transcript_raw_url = payload.get("transcript_url")
            if kind == "error":
                st.error = payload
            st.last_event = payload
            await self._broadcast_to_listeners(queued.job_id, payload)

        try:
            await run_phase_1(
                job_id=queued.job_id,
                source_path=queued.source_path,
                language=queued.language,
                transcriber=self._transcriber,
                progress_callback=progress,
            )
        except Exception:  # noqa: BLE001
            # run_phase_1 already logged + updated DB + emitted error event.
            pass
        finally:
            async with self._lock:
                if queued.job_id in self._pending_order:
                    self._pending_order.remove(queued.job_id)
                self._running_id = None
                await self._broadcast_position_updates_locked()
            self._queue.task_done()

    async def _broadcast_position_updates_locked(self) -> None:
        """Send updated queue_position events to every queued listener.

        Caller must hold _lock.
        """
        for idx, job_id in enumerate(self._pending_order):
            state = self._states.get(job_id)
            if not state or state.status is not JobStatus.QUEUED:
                continue
            if state.queue_position == idx:
                continue
            state.queue_position = idx
            # Don't await broadcasts under the lock to avoid deadlock; schedule.
            asyncio.create_task(
                self._broadcast_to_listeners(
                    job_id,
                    {"type": "queue_position", "job_id": job_id, "position": idx},
                )
            )


# Module-level singleton; the FastAPI lifespan wires start()/stop().
queue: JobQueue = JobQueue()


__all__ = ["queue", "JobQueue", "QueuedJob", "JobState"]
