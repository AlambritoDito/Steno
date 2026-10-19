"""One inference worker, durable jobs, and exclusive ownership of microphone capture."""

import asyncio
import logging
import wave
from pathlib import Path
from uuid import uuid4

from steno.audio import AudioCapture, WavWriter, read_audio_range
from steno.config import Config
from steno.session import Session
from steno.storage import Store
from steno.transcriber import Transcriber

logger = logging.getLogger("steno.jobs")
TERMINAL = {"completed", "failed", "cancelled"}


class BusyError(RuntimeError):
    pass


class JobManager:
    def __init__(self, capture=None):
        self.capture = capture or AudioCapture()
        self.recording = None
        self.task = None
        self.listeners = {}
        self._closing = False
        self._starting_session = None
        self._active_job_id = None
        self._uploads = {}
        self._start_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()

    def busy(self):
        return bool(self.recording or self._starting_session or self._uploads) or any(j["status"] not in TERMINAL for j in Store().list_jobs())

    def session_busy(self, session_id):
        return (self._starting_session == session_id
                or session_id in self._uploads.values()
                or bool(self.recording and self.recording["session_id"] == session_id)
                or any(j["status"] not in TERMINAL for j in Store().list_jobs(session_id)))

    def subscribe(self, session_id, callback):
        self.listeners.setdefault(session_id, set()).add(callback)

    def unsubscribe(self, session_id, callback):
        self.listeners.get(session_id, set()).discard(callback)

    async def emit(self, session_id, event):
        for callback in tuple(self.listeners.get(session_id, ())):
            try:
                await asyncio.wait_for(callback(event), timeout=2)
            except Exception:
                self.unsubscribe(session_id, callback)

    def status(self, session_id):
        jobs = Store().list_jobs(session_id)
        pending = next((j for j in jobs if j["status"] not in TERMINAL), None)
        recording = bool(self.recording and self.recording["session_id"] == session_id
                         and self.capture.is_recording())
        try:
            has_audio = Session.load(session_id).has_audio()
        except FileNotFoundError:
            # A completed session can be deleted while the native model unloads.
            has_audio = False
        return {"type": "status", "recording": recording,
                "processing": bool(pending) and not recording,
                "job_id": pending["id"] if pending else None,
                "model_loaded": False, "has_audio": has_audio}

    async def recover(self):
        from steno.session import recover_interrupted_takes
        await asyncio.to_thread(recover_interrupted_takes)
        store = Store()
        for job in store.list_jobs():
            if job["status"] in {"running", "cancelling"}:
                store.update_job(job["id"], status="cancelled" if job["status"] == "cancelling" else "queued")
        self.kick()

    def reserve_upload(self, session_id):
        """Protect the session and capture ownership while a file is streaming."""
        if self._closing or self.recording or self._starting_session:
            raise BusyError("Stop the recording before uploading audio")
        token = uuid4().hex
        self._uploads[token] = session_id
        return token

    def release_upload(self, token):
        self._uploads.pop(token, None)

    def submit(self, session_id, kind, payload, *, upload_token=None):
        if self.recording or self._starting_session:
            raise BusyError("Stop the recording before starting another transcription")
        if self._uploads and self._uploads.get(upload_token) != session_id:
            raise BusyError("Wait for the audio upload to finish")
        job = Store().create_job(session_id, kind, payload)
        self.kick()
        return job

    def kick(self):
        if not self._closing and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._run(), name="steno-inference")

    async def start_recording(self, session, owner, model, device_index=None, language=None, mode="full", device="auto"):
        async with self._start_lock:
            if self._closing or self.busy():
                raise BusyError("Wait for the current recording or transcription to finish")
            self._starting_session = session.session_id
            take = None
            writer = None
            try:
                take = session.begin_take(mode=mode, source="microphone")
                writer = WavWriter(Path(take["path"]), Config.SAMPLE_RATE)
                await asyncio.to_thread(self.capture.start, device_index, None, writer)
                payload = {"take_id": take["id"], "model": model, "language": language,
                           "mode": mode, "device": device}
                job = Store().create_job(session.session_id, "recording", payload)
                self.recording = {"session_id": session.session_id, "owner": owner,
                                  "job_id": job["id"], "take_id": take["id"], "writer": writer}
            except Exception:
                try:
                    await asyncio.to_thread(self.capture.stop)
                except Exception:
                    logger.exception("Failed to stop capture after microphone startup failure")
                if writer is not None:
                    try:
                        await asyncio.to_thread(writer.close)
                    except Exception:
                        logger.exception("Failed to close archive after microphone startup failure")
                if take is not None:
                    session.finish_take(take["id"], status="failed")
                raise
            finally:
                self._starting_session = None
            self.kick()
            await self.emit(session.session_id, self.status(session.session_id))
            return job

    async def stop_recording(self, owner=None):
        async with self._stop_lock:
            record = self.recording
            if not record:
                return
            if owner is not None and record["owner"] != owner:
                raise BusyError("This connection does not own the recording")
            await asyncio.to_thread(self.capture.stop)
            await asyncio.to_thread(record["writer"].close)
            frames = record["writer"].frames
            job = Store().get_job(record["job_id"])
            take_status = "cancelled" if job and job["status"] in {"cancelled", "cancelling"} else "processing"
            Store().update_take(record["take_id"], frames=frames,
                                duration_seconds=frames / Config.SAMPLE_RATE,
                                status="failed" if self.capture.error else take_status)
            if self.capture.error:
                Store().update_job(record["job_id"], error=self.capture.error)
            self.recording = None
            await self.emit(record["session_id"], self.status(record["session_id"]))

    async def cancel(self, job_id):
        job = Store().get_job(job_id)
        if not job:
            raise FileNotFoundError(job_id)
        if job["status"] in TERMINAL:
            return job
        recording = bool(self.recording and self.recording["job_id"] == job_id)
        # Reserve cancellation before stop yields to the worker. Otherwise a
        # completed job can be changed back to "cancelling" by this request.
        Store().update_job(job_id, status="cancelling" if recording or job["status"] in {"running", "cancelling"} else "cancelled")
        if recording:
            await self.stop_recording()
        current = Store().get_job(job_id)
        if current and current["status"] == "cancelling" and self._active_job_id != job_id:
            current = Store().update_job(job_id, status="cancelled")
        return current

    def retry(self, job_id, device=None):
        job = Store().get_job(job_id)
        if not job:
            raise FileNotFoundError(job_id)
        if job["status"] not in {"failed", "cancelled"}:
            raise BusyError("Only failed or cancelled jobs can be retried")
        if self.recording or self._starting_session or self._uploads:
            raise BusyError("Stop recording before retrying")
        payload = dict(job["payload"])
        if device:
            payload["device"] = device
        result = Store().update_job(job_id, status="queued", error=None, payload=payload, result=None)
        self.kick()
        return result

    def _cancelled(self, job_id):
        job = Store().get_job(job_id)
        return job is None or job["status"] in {"cancelled", "cancelling"}

    async def _run(self):
        while not self._closing:
            pending = Store().list_jobs(status="queued")
            if not pending:
                break
            job = pending[0]
            self._active_job_id = job["id"]
            Store().update_job(job["id"], status="running")
            await self.emit(job["session_id"], self.status(job["session_id"]))
            transcriber = None
            try:
                payload = job["payload"]
                transcriber = Transcriber(model_name=payload["model"], device=payload.get("device", "auto"))
                transcriber.set_language(payload.get("language"))
                if job["kind"] == "recording":
                    result = await self._recorded(job, transcriber)
                else:
                    result = await self._file(job, transcriber)
                if self._closing:
                    Store().update_job(job["id"], status="queued")
                    break
                result["capabilities"] = transcriber.get_capabilities()
                cancelled = self._cancelled(job["id"])
                error = Store().get_job(job["id"]).get("error")
                Store().update_job(job["id"], status="cancelled" if cancelled else ("failed" if error else "completed"), result=result)
            except Exception as exc:
                logger.exception("Job %s failed", job["id"])
                if self.recording and self.recording["job_id"] == job["id"]:
                    await self.stop_recording()
                cancelled = self._cancelled(job["id"])
                if Store().get_job(job["id"]):
                    Store().update_job(job["id"], status="cancelled" if cancelled else "failed",
                                       error=None if cancelled else str(exc))
                if not cancelled:
                    await self.emit(job["session_id"], {"type": "error", "message": str(exc), "job_id": job["id"]})
            finally:
                if transcriber is not None:
                    # Dispose through the backend facade; do not leave multiple models resident.
                    unload = getattr(transcriber, "unload", None)
                    if unload:
                        try:
                            await asyncio.to_thread(unload)
                        except Exception:
                            logger.exception("Native model cleanup failed for job %s", job["id"])
                self._active_job_id = None
                await self.emit(job["session_id"], self.status(job["session_id"]))

    async def _recorded(self, job, transcriber):
        session = Session.load(job["session_id"])
        take_id = job["payload"]["take_id"]
        take = next(t for t in Store().takes(session.session_id) if t["id"] == take_id)
        path = Path(take["path"])
        cursor = take["processed_frames"]
        sample_rate = take["sample_rate"]
        step = int(sample_rate * Config.CHUNK_DURATION)
        overlap = int(sample_rate * Config.OVERLAP_DURATION)
        count = 0
        frames = take["frames"]
        while not self._closing and not self._cancelled(job["id"]):
            active = bool(self.recording and self.recording["job_id"] == job["id"])
            if active and not self.capture.is_recording():
                await self.stop_recording()
                active = False
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
            Store().update_take(take_id, frames=frames)
            await self.emit(session.session_id, {"type": "elapsed", "seconds": frames // sample_rate})
            if frames <= cursor or (active and frames - cursor < step):
                if not active:
                    break
                await asyncio.sleep(0.2)
                continue
            end = min(cursor + step, frames)
            start = max(0, cursor - overlap)
            if job["payload"].get("mode") != "light":
                chunk = await asyncio.to_thread(read_audio_range, path, start, end)
                segments = await asyncio.to_thread(transcriber.transcribe_segments, chunk)
                if self._cancelled(job["id"]):
                    break
                for index, segment in enumerate(segments):
                    if self._cancelled(job["id"]):
                        break
                    text = segment.get("text", "").strip()
                    seg_end = min(end / sample_rate, start / sample_rate + segment.get("end", len(chunk) / sample_rate))
                    if not text or seg_end <= cursor / sample_rate:
                        continue
                    seg_start = max(cursor / sample_rate, start / sample_rate + segment.get("start", 0))
                    session.add_transcript(text, take_id=take_id, start=seg_start, end=seg_end,
                                           entry_key=f"{job['id']}:{cursor}:{index}")
                    count += 1
                    await self.emit(session.session_id, {"type": "transcript", "text": text,
                                    "timestamp": f"{int(seg_start)//60:02d}:{int(seg_start)%60:02d}",
                                    "start": seg_start, "end": seg_end, "is_final": True})
                if self._cancelled(job["id"]):
                    break
            cursor = end
            Store().update_take(take_id, processed_frames=cursor)
        if not self._closing:
            if self._cancelled(job["id"]) and self.recording and self.recording["job_id"] == job["id"]:
                await self.stop_recording()
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
            current = Store().get_job(job["id"])
            state = "cancelled" if self._cancelled(job["id"]) else ("failed" if current and current["error"] else "complete")
            session.finish_take(take_id, status=state, duration_seconds=frames / sample_rate)
        return {"count": count, "take_id": take_id}

    async def _file(self, job, transcriber):
        session = Session.load(job["session_id"])
        payload = job["payload"]
        take_ids = payload.get("take_ids") or [payload["take_id"]]
        store = Store()
        count = 0
        completed_takes = []
        for take_id in take_ids:
            if self._closing or self._cancelled(job["id"]):
                break
            path = session.audio_path(take_id)
            checkpoint = store.file_checkpoint(job["id"], take_id)
            if checkpoint["completed"]:
                count += checkpoint["segment_count"]
                completed_takes.append(take_id)
                continue
            stream = getattr(transcriber, "iter_transcribe_file", None)
            if stream is not None:
                iterator = stream(str(path), start_frame=checkpoint["processed_frames"])
                try:
                    while not self._closing and not self._cancelled(job["id"]):
                        batch = await asyncio.to_thread(next, iterator, None)
                        if batch is None:
                            break
                        duration, batch_segments = batch
                        store.save_file_batch(job["id"], take_id,
                                              round(duration * Config.SAMPLE_RATE), batch_segments)
                        await self.emit(session.session_id, {"type": "job_progress", "job_id": job["id"],
                                                             "take_id": take_id, "processed_seconds": duration})
                finally:
                    await asyncio.to_thread(iterator.close)
            else:
                # Compatible with external adapters that implement the original API.
                segments = await asyncio.to_thread(transcriber.transcribe_file, str(path))
                duration = max((s.get("end", 0) for s in segments), default=0)
                if not self._closing and not self._cancelled(job["id"]):
                    store.save_file_batch(job["id"], take_id, round(duration * Config.SAMPLE_RATE), segments)
            if self._closing or self._cancelled(job["id"]):
                break
            count += store.commit_file_checkpoint(job["id"], take_id)
            completed_takes.append(take_id)
        return {"count": count, "take_ids": completed_takes}

    async def shutdown(self):
        self._closing = True
        # Wait for a native microphone start to finish before stopping it.
        async with self._start_lock:
            await self.stop_recording()
        if self.task:
            await self.task
