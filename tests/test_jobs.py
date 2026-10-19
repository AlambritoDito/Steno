"""Behavioral regression tests for durable jobs and recording ownership."""
import asyncio
from pathlib import Path
import threading
import wave

import numpy as np
import pytest

from steno.audio import WavWriter
from steno.config import Config
from steno.jobs import BusyError, JobManager
from steno.session import Session
from steno.storage import Store


class FakeTranscriber:
    fail = False
    def __init__(self, **kwargs):
        self.language = None
    def set_language(self, language):
        self.language = language
    def transcribe_segments(self, audio):
        if self.fail:
            raise RuntimeError("CUDA allocation failed")
        return [{"start": 0, "end": len(audio) / Config.SAMPLE_RATE, "text": "kept"}]
    def transcribe_file(self, path):
        if self.fail:
            raise RuntimeError("bad input")
        assert Path(path).is_file()
        return [{"start": 3661, "end": 3662, "text": "file text"}]
    def unload(self):
        pass
    def get_capabilities(self):
        return {"backend": "fake", "device": "cpu", "validated": True}


class FakeCapture:
    def __init__(self):
        self.recording = False
        self.writer = None
        self.error = None
    def start(self, device, callback, writer):
        self.writer = writer
        self.recording = True
    def stop(self):
        self.recording = False
    def is_recording(self):
        return self.recording
    def feed(self, seconds):
        self.writer.write(np.ones(int(seconds * Config.SAMPLE_RATE), dtype=np.float32) * 0.2)


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr("steno.jobs.Transcriber", FakeTranscriber)
    monkeypatch.setattr(FakeTranscriber, "fail", False)


def file_take(session):
    take = session.begin_take(source="upload")
    writer = WavWriter(Path(take["path"]))
    writer.write(np.ones(16000, dtype=np.float32) * .1)
    writer.close()
    return take


@pytest.mark.asyncio
async def test_stop_preserves_partial_audio_and_resume_creates_separate_take():
    session = Session("lecture")
    capture = FakeCapture()
    jobs = JobManager(capture)
    first = await jobs.start_recording(session, "owner", "tiny")
    capture.feed(2)
    await jobs.stop_recording("owner")
    await jobs.task
    one = session.takes()[0]
    before = Path(one["path"]).read_bytes()
    assert Store().get_job(first["id"])["status"] == "completed"
    assert one["processed_frames"] == 32000
    assert len(session.to_dict()["entries"]) == 1
    await jobs.start_recording(session, "owner", "tiny")
    capture.feed(1)
    await jobs.stop_recording("owner")
    await jobs.task
    assert len(session.takes()) == 2
    assert Path(one["path"]).read_bytes() == before


@pytest.mark.asyncio
async def test_other_client_cannot_stop_or_replace_recording():
    session = Session("one")
    jobs = JobManager(FakeCapture())
    await jobs.start_recording(session, "owner", "tiny", mode="light")
    with pytest.raises(BusyError):
        await jobs.stop_recording("visitor")
    with pytest.raises(BusyError):
        await jobs.start_recording(Session("two"), "visitor", "tiny")
    assert jobs.capture.is_recording()
    await jobs.stop_recording("owner")
    await jobs.task


@pytest.mark.asyncio
async def test_recover_queued_recording_uses_cursor_without_duplicate_segments():
    session = Session("recovery")
    take = file_take(session)
    job = Store().create_job(session.session_id, "recording", {"model": "tiny", "take_id": take["id"], "mode": "full"})
    # Simulate death after persisting text but before committing the cursor.
    session.add_transcript("kept", take_id=take["id"], start=0, end=1, entry_key=f"{job['id']}:0:0")
    Store().update_job(job["id"], status="running")
    jobs = JobManager(FakeCapture())
    await jobs.recover()
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "completed"
    assert len(session.to_dict()["entries"]) == 1


@pytest.mark.asyncio
async def test_failed_retranscription_retains_original_and_can_retry(monkeypatch):
    session = Session("retry")
    take = file_take(session)
    session.add_transcript("original", take_id=take["id"], start=0, end=1)
    jobs = JobManager()
    monkeypatch.setattr(FakeTranscriber, "fail", True)
    job = jobs.submit(session.session_id, "retranscribe", {"model": "tiny", "take_id": take["id"]})
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "failed"
    assert "original" in session.to_markdown()
    assert Path(take["path"]).is_file()
    monkeypatch.setattr(FakeTranscriber, "fail", False)
    jobs.retry(job["id"], "cpu")
    await jobs.task
    done = Store().get_job(job["id"])
    assert done["status"] == "completed"
    assert done["payload"]["device"] == "cpu"
    assert "01:01:01" in session.to_markdown()
    assert "original" not in session.to_markdown()


@pytest.mark.asyncio
async def test_cancel_running_does_not_commit_late_inference(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(self, path):
        entered.set()
        assert release.wait(5)
        return [{"start": 0, "end": 1, "text": "must not commit"}]
    monkeypatch.setattr(FakeTranscriber, "transcribe_file", slow)
    session = Session("cancel")
    take = file_take(session)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    assert await asyncio.to_thread(entered.wait, 5)
    await jobs.cancel(job["id"])
    release.set()
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "cancelled"
    assert "must not commit" not in session.to_markdown()
    assert session.has_audio()


@pytest.mark.asyncio
async def test_shutdown_preserves_queued_work_for_next_start():
    session = Session("restart")
    take = file_take(session)
    jobs = JobManager()
    jobs._closing = True
    job = jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    assert Store().get_job(job["id"])["status"] == "queued"
    next_jobs = JobManager()
    await next_jobs.recover()
    await next_jobs.task
    assert Store().get_job(job["id"])["status"] == "completed"


@pytest.mark.asyncio
async def test_cancel_recording_does_not_revert_completed_worker_to_cancelling(monkeypatch):
    session = Session("cancel race")
    jobs = JobManager(FakeCapture())
    job = await jobs.start_recording(session, "owner", "tiny", mode="light")
    jobs.capture.feed(1)
    original_stop = jobs.stop_recording

    async def stop_and_allow_worker_to_finish(owner=None):
        await original_stop(owner)
        if asyncio.current_task() is not jobs.task:
            await jobs.task

    monkeypatch.setattr(jobs, "stop_recording", stop_and_allow_worker_to_finish)
    result = await jobs.cancel(job["id"])
    assert result["status"] == "cancelled"
    assert not jobs.busy()
    assert session.takes()[0]["status"] == "cancelled"
    assert session.takes()[0]["duration_seconds"] == 1


@pytest.mark.asyncio
async def test_cancel_before_recording_worker_starts_finalizes_take(monkeypatch):
    session = Session("cancel queued recording")
    jobs = JobManager(FakeCapture())
    monkeypatch.setattr(jobs, "kick", lambda: None)
    job = await jobs.start_recording(session, "owner", "tiny")
    jobs.capture.feed(0.25)
    result = await jobs.cancel(job["id"])
    assert result["status"] == "cancelled"
    assert session.takes()[0]["status"] == "cancelled"
    assert not jobs.capture.is_recording()
    assert not jobs.session_busy(session.session_id)


@pytest.mark.asyncio
async def test_double_cancel_does_not_make_running_inference_deletable(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(self, path):
        entered.set()
        assert release.wait(5)
        return []
    monkeypatch.setattr(FakeTranscriber, "transcribe_file", slow)
    session = Session("double cancel")
    take = file_take(session)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        await jobs.cancel(job["id"])
        result = await jobs.cancel(job["id"])
        assert result["status"] == "cancelling"
        assert jobs.session_busy(session.session_id)
    finally:
        release.set()
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_session_deleted_while_unloading_does_not_stop_worker_queue(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def unload(self):
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(FakeTranscriber, "unload", unload)
    session = Session("delete during cleanup")
    take = file_take(session)
    second = Session("next job")
    second_take = file_take(second)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    next_job = jobs.submit(second.session_id, "file", {"model": "tiny", "take_id": second_take["id"]})
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert Store().get_job(job["id"])["status"] == "completed"
        session.delete()
    finally:
        release.set()
    await jobs.task
    assert Store().get_job(next_job["id"])["status"] == "completed"


@pytest.mark.asyncio
async def test_microphone_start_reserves_capture_before_native_start_returns():
    entered, release = threading.Event(), threading.Event()
    class SlowCapture(FakeCapture):
        def start(self, *args):
            entered.set()
            assert release.wait(5)
            super().start(*args)
    session = Session("starting")
    jobs = JobManager(SlowCapture())
    start = asyncio.create_task(jobs.start_recording(session, "owner", "tiny", mode="light"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert jobs.busy()
        assert jobs.session_busy(session.session_id)
        with pytest.raises(BusyError):
            jobs.submit(session.session_id, "file", {})
    finally:
        release.set()
    await start
    await jobs.stop_recording()
    await jobs.task


@pytest.mark.asyncio
async def test_cancel_during_segment_notification_keeps_cursor_retryable(monkeypatch):
    def two_segments(self, audio):
        return [{"start": 0, "end": .5, "text": "first"},
                {"start": .5, "end": 1, "text": "late"}]
    monkeypatch.setattr(FakeTranscriber, "transcribe_segments", two_segments)
    session = Session("cancel midchunk")
    jobs = JobManager(FakeCapture())
    job = await jobs.start_recording(session, "owner", "tiny")
    jobs.capture.feed(1)
    async def cancel_on_first(event):
        if event["type"] == "transcript":
            await jobs.cancel(job["id"])
    jobs.subscribe(session.session_id, cancel_on_first)
    await jobs.stop_recording()
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "cancelled"
    assert "first" in session.to_markdown()
    assert "late" not in session.to_markdown()
    assert session.takes()[0]["processed_frames"] == 0


@pytest.mark.asyncio
async def test_streamed_file_cancellation_closes_decoder_and_keeps_original(monkeypatch):
    closed = []
    def stream(self, path, *, start_frame=0):
        try:
            yield 30.0, [{"start": 0, "end": 1, "text": "replacement"}]
            raise AssertionError("Cancelled job must not decode the next window")
        finally:
            closed.append(True)
    monkeypatch.setattr(FakeTranscriber, "iter_transcribe_file", stream, raising=False)
    session = Session("cancel streamed file")
    take = file_take(session)
    session.add_transcript("original", take_id=take["id"], start=0, end=1)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "retranscribe", {"model": "tiny", "take_id": take["id"]})
    async def cancel_on_progress(event):
        if event["type"] == "job_progress":
            await jobs.cancel(job["id"])
    jobs.subscribe(session.session_id, cancel_on_progress)
    await jobs.task
    assert closed == [True]
    assert "original" in session.to_markdown()
    assert "replacement" not in session.to_markdown()
    assert Store().get_job(job["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_failed_file_resumes_durable_windows_and_preserves_original(monkeypatch):
    starts = []
    def stream(self, path, *, start_frame=0):
        starts.append(start_frame)
        if start_frame == 0:
            yield 30.0, [{"start": 1, "end": 2, "text": "first window"}]
            raise RuntimeError("GPU failed after first checkpoint")
        yield 40.0, [{"start": 32, "end": 33, "text": "second window"}]
    monkeypatch.setattr(FakeTranscriber, "iter_transcribe_file", stream, raising=False)
    session = Session("resume windows")
    take = file_take(session)
    session.add_transcript("original", take_id=take["id"], start=0, end=1)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "retranscribe", {"model": "tiny", "take_id": take["id"]})
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "failed"
    assert Store().file_checkpoint(job["id"], take["id"])["processed_frames"] == 30 * Config.SAMPLE_RATE
    assert "original" in session.to_markdown()
    assert "first window" not in session.to_markdown()
    # Use a new manager to exercise persisted state, including a CPU retry.
    restarted = JobManager()
    restarted.retry(job["id"], "cpu")
    await restarted.task
    assert starts == [0, 30 * Config.SAMPLE_RATE]
    assert Store().get_job(job["id"])["status"] == "completed"
    assert "original" not in session.to_markdown()
    assert "first window" in session.to_markdown()
    assert "second window" in session.to_markdown()
    assert session.takes()[0]["duration_seconds"] == 40
    with Store().connection() as db:
        assert db.execute("SELECT count(*) FROM job_checkpoint_batches").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_cancelled_file_retry_resumes_staged_text_without_duplicate_inference(monkeypatch):
    starts = []
    def stream(self, path, *, start_frame=0):
        starts.append(start_frame)
        if not start_frame:
            yield 30.0, [{"start": 1, "end": 2, "text": "saved before cancellation"}]
        yield 35.0, []  # Silence must still advance the durable cursor.
    monkeypatch.setattr(FakeTranscriber, "iter_transcribe_file", stream, raising=False)
    session = Session("cancel resume")
    take = file_take(session)
    jobs = JobManager()
    job = jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    async def cancel_on_progress(event):
        if event["type"] == "job_progress":
            await jobs.cancel(job["id"])
    jobs.subscribe(session.session_id, cancel_on_progress)
    await jobs.task
    assert Store().get_job(job["id"])["status"] == "cancelled"
    assert "saved before cancellation" not in session.to_markdown()
    jobs.unsubscribe(session.session_id, cancel_on_progress)
    jobs.retry(job["id"])
    await jobs.task
    assert starts == [0, 30 * Config.SAMPLE_RATE]
    assert session.to_markdown().count("saved before cancellation") == 1
    assert session.takes()[0]["duration_seconds"] == 35


@pytest.mark.asyncio
async def test_multitake_recovery_skips_committed_takes_and_completed_windows(monkeypatch):
    session = Session("recover all")
    first, second = file_take(session), file_take(session)
    calls = []
    def stream(self, path, *, start_frame=0):
        calls.append((Path(path).stem, start_frame))
        yield 20.0, [{"start": 1, "end": 2, "text": "recovered second take"}]
    monkeypatch.setattr(FakeTranscriber, "iter_transcribe_file", stream, raising=False)
    store = Store()
    job = store.create_job(session.session_id, "retranscribe", {"model": "tiny", "take_ids": [first["id"], second["id"]]})
    store.file_checkpoint(job["id"], first["id"])
    store.save_file_batch(job["id"], first["id"], 16000, [{"start": 0, "end": 1, "text": "committed first take"}])
    store.commit_file_checkpoint(job["id"], first["id"])
    store.file_checkpoint(job["id"], second["id"])
    store.save_file_batch(job["id"], second["id"], 16000, [])
    store.update_job(job["id"], status="running")  # Process died before final job status.
    restarted = JobManager()
    await restarted.recover()
    await restarted.task
    assert calls == [(second["id"], 16000)]
    assert Store().get_job(job["id"])["result"]["count"] == 2
    assert session.to_markdown().count("committed first take") == 1
    assert "recovered second take" in session.to_markdown()


def test_checkpoint_commit_rollback_preserves_original_and_staged_text():
    session = Session("atomic publication")
    take = file_take(session)
    session.add_transcript("original", take_id=take["id"], start=0, end=1)
    store = Store()
    job = store.create_job(session.session_id, "retranscribe", {"model": "tiny", "take_id": take["id"]})
    store.file_checkpoint(job["id"], take["id"])
    store.save_file_batch(job["id"], take["id"], 16000, [{"start": 0, "end": 1, "text": "replacement"}])
    # Force a storage failure after the transaction deletes the old transcript.
    with store.connection() as db:
        db.execute("CREATE TRIGGER fail_publication BEFORE INSERT ON entries "
                   "BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END")
    with pytest.raises(Exception, match="simulated disk failure"):
        store.commit_file_checkpoint(job["id"], take["id"])
    assert "original" in session.to_markdown()
    assert not store.file_checkpoint(job["id"], take["id"])["completed"]
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM job_checkpoint_batches").fetchone()[0] == 1
        db.execute("DROP TRIGGER fail_publication")
    assert store.commit_file_checkpoint(job["id"], take["id"]) == 1
    assert "replacement" in session.to_markdown()


def test_checkpoint_rejects_foreign_take_and_invalid_batch_without_advancing_cursor():
    session, other = Session("owner"), Session("unrelated")
    take, foreign = file_take(session), file_take(other)
    store = Store()
    job = store.create_job(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    with pytest.raises(ValueError, match="does not belong"):
        store.file_checkpoint(job["id"], foreign["id"])
    store.file_checkpoint(job["id"], take["id"])
    with pytest.raises(ValueError, match="Invalid checkpoint segment"):
        store.save_file_batch(job["id"], take["id"], 16000, [{"start": 0, "end": float("nan"), "text": "bad"}])
    assert store.file_checkpoint(job["id"], take["id"])["processed_frames"] == 0


@pytest.mark.asyncio
async def test_job_logs_lifecycle_metadata_without_error_text_or_user_content(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="steno.jobs")
    session = Session("private-session-name")
    take = file_take(session)
    def fail(self, path):
        raise RuntimeError("private-error-text")
    monkeypatch.setattr(FakeTranscriber, "transcribe_file", fail)
    jobs = JobManager()
    jobs.submit(session.session_id, "file", {"model": "tiny", "take_id": take["id"]})
    await jobs.task
    records = "\n".join(record.getMessage() for record in caplog.records if record.name == "steno.jobs")
    assert "event=job_queued" in records and "event=job_started" in records
    assert "event=job_failed" in records and "error_type=RuntimeError" in records
    assert "private-session-name" not in records and "private-error-text" not in records
