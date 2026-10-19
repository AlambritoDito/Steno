"""Desktop HTTP API. Jobs own inference, sessions own storage, and the UI owns no audio."""

import asyncio
import logging
import os
import platform
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from packaging.version import InvalidVersion, Version

from steno.audio import AudioCapture, AudioCaptureError, PORTAUDIO_AVAILABLE, _portaudio_error
from steno.config import Config, WHISPER_MODELS, SUPPORTED_AUDIO_EXTENSIONS, _detect_hardware, recommend_model, default_backend
from steno.i18n import load_locale, get_supported_languages
from steno.jobs import JobManager, BusyError
from steno.security import LocalSecurity
from steno.session import Session, list_sessions, safe_path, validate_id, import_legacy_sessions
from steno.storage import Store
from steno.transcriber import Transcriber

logger = logging.getLogger("steno")
APP_VERSION = "1.0.0-beta.1"
_sessions = {}  # Compatibility for integrations; persisted Session objects are short-lived.
_audio_capture = AudioCapture()
_download_progress = {}
_model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(import_legacy_sessions)
    settings = Config.load_settings()
    model = settings.get("model_key", "large-v3-turbo")
    model = WHISPER_MODELS.get(model, WHISPER_MODELS["large-v3-turbo"])["repo"]
    app.state.transcriber = Transcriber(model_name=model)
    app.state.setup_complete = settings.get("setup_complete", False) and Config.model_is_downloaded(model)
    app.state.jobs = JobManager(_audio_capture)
    app.state.model_busy = False
    await app.state.jobs.recover()
    try:
        yield
    finally:
        await app.state.jobs.shutdown()


app = FastAPI(title="Steno", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(LocalSecurity)
app.mount("/static", StaticFiles(directory=str(Config.static_path())), name="static")


def manager():
    if not hasattr(app.state, "jobs"):
        app.state.jobs = JobManager(_audio_capture)
    return app.state.jobs


def session_or_404(session_id):
    try:
        return Session.load(session_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Session not found")


def require_idle():
    if manager().busy() or getattr(app.state, "model_busy", False):
        raise HTTPException(409, "Wait for current recording, transcription or model operation")


def model_repo(model_key):
    if not isinstance(model_key, str) or model_key not in WHISPER_MODELS:
        raise HTTPException(422, "Invalid model key")
    return WHISPER_MODELS[model_key]["repo"]


@asynccontextmanager
async def model_operation():
    """Reserve the idle runtime before the first await in a model operation."""
    require_idle()
    if _model_lock.locked():
        raise HTTPException(409, "Wait for the current model operation")
    app.state.model_busy = True
    try:
        async with _model_lock:
            yield
    finally:
        app.state.model_busy = False


async def activate_candidate(repo, settings):
    """Only publish a model and settings after its native warm-up succeeds."""
    previous = app.state.transcriber
    was_loaded = previous.is_loaded()
    candidate = Transcriber(model_name=repo)
    try:
        await asyncio.to_thread(candidate.warmup)
        Config.save_settings(settings)
    except BaseException:
        try:
            await asyncio.to_thread(candidate.unload)
            if was_loaded and not previous.is_loaded():
                await asyncio.to_thread(previous.warmup)
        except Exception:
            logger.exception("Could not restore the previously loaded model")
        raise
    app.state.transcriber = candidate
    app.state.setup_complete = settings.get("setup_complete", app.state.setup_complete)


@app.exception_handler(ValueError)
async def invalid_value(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(BusyError)
async def busy_error(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse((Config.static_path() / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/status")
async def get_status():
    transcriber = app.state.transcriber
    return {"model_loaded": transcriber.is_loaded(), "model_info": transcriber.get_model_info(),
            "capabilities": await asyncio.to_thread(transcriber.get_capabilities),
            "active_sessions": [manager().recording["session_id"]] if manager().recording else [],
            "recording": _audio_capture.is_recording(),
            "busy": manager().busy() or getattr(app.state, "model_busy", False),
            "setup_complete": app.state.setup_complete, "portaudio_available": PORTAUDIO_AVAILABLE,
            "portaudio_error": _portaudio_error if not PORTAUDIO_AVAILABLE else None}


@app.get("/api/version")
async def get_version():
    return {"version": APP_VERSION}


def _compare_versions(a, b):
    try:
        return (Version(a) > Version(b)) - (Version(a) < Version(b))
    except InvalidVersion:
        return 0


@app.get("/api/update-check")
async def check_for_update():
    def check():
        import json
        import urllib.request
        request = urllib.request.Request(
            "https://api.github.com/repos/AlambritoDito/Steno/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Steno"})
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
        latest = data.get("tag_name", "").lstrip("v")
        return {"update_available": _compare_versions(latest, APP_VERSION) > 0,
                "current_version": APP_VERSION, "latest_version": latest,
                "release_url": "https://github.com/AlambritoDito/Steno/releases",
                "release_notes": (data.get("body") or "")[:500]}
    try:
        return await asyncio.to_thread(check)
    except Exception as exc:
        return {"update_available": False, "current_version": APP_VERSION, "error": str(exc)}


@app.get("/api/devices")
async def get_devices():
    try:
        return await asyncio.to_thread(AudioCapture.list_devices)
    except AudioCaptureError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/hardware")
async def get_hardware():
    hardware = await asyncio.to_thread(_detect_hardware)
    recommended = recommend_model(hardware["ram_gb"])
    return {**hardware, "recommended_model": recommended,
            "capabilities": await asyncio.to_thread(app.state.transcriber.get_capabilities),
            "models": [{"key": key, **info, "recommended": key == recommended,
                        "compatible": hardware["ram_gb"] is None or hardware["ram_gb"] >= info["min_ram_gb"]}
                       for key, info in WHISPER_MODELS.items()]}


async def perform_download(key, activate):
    repo = model_repo(key)
    async with model_operation():
        _download_progress.clear()
        try:
            await asyncio.to_thread(Transcriber.download_model_by_repo, repo, _download_progress)
            if not Config.model_is_downloaded(repo):
                raise ValueError("Downloaded model is incomplete; retry download")
            settings = Config.load_settings()
            settings["downloaded_models"] = sorted(set(settings.get("downloaded_models", [])) | {key})
            if activate:
                settings.update(model_key=key, model_repo=repo, setup_complete=True)
                await activate_candidate(repo, settings)
            else:
                Config.save_settings(settings)
            return {"status": "ok", "model_key": key, "model_repo": repo}
        except Exception as exc:
            _download_progress["status"] = "cancelled" if _download_progress.get("cancel_requested") else "failed"
            logger.exception("Model setup failed")
            raise HTTPException(409 if _download_progress.get("cancel_requested") else 503, str(exc))


@app.post("/api/setup/select-model")
async def select_model(body: dict):
    return await perform_download(body.get("model_key"), True)


@app.post("/api/models/download")
async def download_model(body: dict):
    return await perform_download(body.get("model_key"), False)


@app.post("/api/models/download/cancel")
async def cancel_download():
    _download_progress["cancel_requested"] = True
    return {"status": "cancelling"}


@app.get("/api/download-progress")
async def get_download_progress():
    total = _download_progress.get("bytes_total", 0)
    downloaded = _download_progress.get("bytes_downloaded", 0)
    return {"status": _download_progress.get("status", "idle"), "bytes_downloaded": downloaded,
            "bytes_total": total, "percent": min(100, round(downloaded / total * 100, 1)) if total else 0}


@app.get("/api/models")
async def get_models():
    active = app.state.transcriber._model_name
    return {"active_repo": active, "models": [
        {"key": key, **info, "downloaded": Config.model_is_downloaded(info["repo"]),
         "disk_mb": Config.model_cache_size_mb(info["repo"]), "active": info["repo"] == active}
        for key, info in WHISPER_MODELS.items()]}


@app.post("/api/models/active")
async def set_active_model(body: dict):
    key = body.get("model_key")
    repo = model_repo(key)
    async with model_operation():
        if not Config.model_is_downloaded(repo):
            raise HTTPException(409, "Download this model first")
        settings = Config.load_settings()
        settings.update(model_key=key, model_repo=repo, setup_complete=True)
        try:
            await activate_candidate(repo, settings)
        except Exception as exc:
            raise HTTPException(503, str(exc))
    return {"status": "ok", "model_key": key, "model_repo": repo}


@app.delete("/api/models/{model_key}")
async def delete_model(model_key: str):
    repo = model_repo(model_key)
    async with model_operation():
        if repo == app.state.transcriber._model_name:
            raise HTTPException(409, "Cannot delete the active model")
        deleted = await asyncio.to_thread(Config.delete_model_cache, repo)
    return {"status": "ok", "deleted": deleted}


@app.get("/api/mic-test")
async def mic_test(device_index: int | None = None):
    async with model_operation():
        try:
            from steno.audio import sd
            import numpy as np
            if not PORTAUDIO_AVAILABLE:
                raise AudioCaptureError("PortAudio is unavailable")
            audio = await asyncio.to_thread(sd.rec, int(0.5 * Config.SAMPLE_RATE),
                                           samplerate=Config.SAMPLE_RATE, device=device_index,
                                           channels=1, dtype="float32", blocking=True)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return {"status": "ok", "rms": rms, "has_audio": rms > 0.001}
        except Exception as exc:
            raise HTTPException(503, str(exc))


@app.get("/api/sessions")
async def get_sessions():
    return await asyncio.to_thread(list_sessions)


@app.post("/api/sessions/new", status_code=201)
async def create_session(body: dict):
    name = body.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise HTTPException(422, "Session name must contain 1–200 characters")
    session = Session(name=name.strip())
    return {"session_id": session.session_id, "name": session.name, "created_at": session.created_at.isoformat()}


@app.get("/api/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    return session_or_404(session_id).to_dict()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    return Response(session_or_404(session_id).to_markdown(), media_type="text/markdown")


@app.put("/api/sessions/{session_id}/note")
@app.post("/api/sessions/{session_id}/note")
async def add_note(session_id: str, body: dict):
    markdown = body.get("markdown", "")
    if not isinstance(markdown, str) or len(markdown) > 2_000_000:
        raise HTTPException(422, "Invalid note")
    session_or_404(session_id).add_note(markdown)
    return {"status": "ok"}


@app.post("/api/sessions/{session_id}/image")
async def add_image(session_id: str, file: UploadFile = File(...), caption: str = Form("")):
    session = session_or_404(session_id)
    image = await file.read(20 * 1024 * 1024 + 1)
    if len(image) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 20 MB")
    mime = ("image/png" if image.startswith(b"\x89PNG\r\n\x1a\n") else
            "image/jpeg" if image.startswith(b"\xff\xd8\xff") else
            "image/gif" if image.startswith((b"GIF87a", b"GIF89a")) else
            "image/webp" if image[:4] == b"RIFF" and image[8:12] == b"WEBP" else None)
    if not mime:
        raise HTTPException(422, "Use PNG, JPEG, GIF or WebP images")
    return {"status": "ok", **session.add_image(image, mime, caption[:1000])}


@app.get("/api/sessions/{session_id}/images/{filename}")
async def get_session_image(session_id: str, filename: str):
    session_or_404(session_id)
    try:
        path = safe_path(Config.images_path(session_id), filename)
        if not path.is_file():
            raise ValueError("Not found")
    except ValueError:
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.post("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "md"):
    session = session_or_404(session_id)
    if format not in {"md", "zip"}:
        raise HTTPException(422, "Use md or zip")
    content = await asyncio.to_thread(session.export_zip if format == "zip" else lambda: session.to_markdown(portable=True))
    return Response(content, media_type="application/zip" if format == "zip" else "text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="{session_id}.{format}"'})


@app.api_route("/api/sessions/{session_id}/audio", methods=["GET", "HEAD"])
async def get_session_audio(session_id: str, take_id: str | None = None):
    try:
        path = session_or_404(session_id).audio_path(take_id)
        if not path.is_file():
            raise FileNotFoundError()
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Audio not found")
    return FileResponse(path)


def validate_device(device):
    allowed = {"auto", "metal"} if default_backend() == "mlx" else {"auto", "cpu", "cuda"}
    if not isinstance(device, str) or device not in allowed:
        raise HTTPException(422, "Invalid device for this platform")
    return device


def job_payload(body=None):
    body = body or {}
    device = validate_device(body.get("device", "auto"))
    language = body.get("language")
    if language == "":
        language = None
    if language is not None and (not isinstance(language, str) or not language.isalpha() or len(language) > 3):
        raise HTTPException(422, "Invalid language")
    return {"model": model_repo(body["model_key"]) if body.get("model_key") else app.state.transcriber._model_name,
            "device": device, "language": language}


@app.post("/api/sessions/{session_id}/transcribe-file", status_code=202)
async def transcribe_file(session_id: str, file: UploadFile = File(...), language: str | None = Form(None), device: str = Form("auto")):
    session = session_or_404(session_id)
    if manager().recording or getattr(app.state, "model_busy", False):
        raise HTTPException(409, "Stop recording or wait for model setup first")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(422, "Unsupported audio format")
    payload = job_payload({"language": language, "device": device})
    jobs = manager()
    reservation = jobs.reserve_upload(session_id)
    take = path = None
    size = 0
    try:
        take = session.begin_take(source="upload", suffix=suffix)
        path = Path(take["path"])
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise HTTPException(413, "Audio exceeds 1 GB")
                await asyncio.to_thread(output.write, chunk)
            output.flush()
            os.fsync(output.fileno())
        if not size:
            raise HTTPException(422, "Empty audio file")
        session.update_take(take["id"], status="queued")
        if getattr(app.state, "model_busy", False):
            raise HTTPException(409, "Wait for model setup")
        payload = {**payload, "take_id": take["id"]}
        job = jobs.submit(session_id, "file", payload, upload_token=reservation)
        return {"job_id": job["id"], "status": job["status"]}
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        if take is not None:
            session.finish_take(take["id"], status="failed")
        raise
    finally:
        jobs.release_upload(reservation)
        await file.close()


@app.post("/api/sessions/{session_id}/retranscribe", status_code=202)
async def retranscribe_session(session_id: str, body: dict):
    session = session_or_404(session_id)
    if getattr(app.state, "model_busy", False):
        raise HTTPException(409, "Wait for model setup")
    takes = session.takes()
    ids = [t["id"] for t in takes if Path(t["path"]).is_file()]
    if body.get("take_id"):
        if body["take_id"] not in ids:
            raise HTTPException(404, "Take not found")
        ids = [body["take_id"]]
    if not ids:
        raise HTTPException(404, "No saved audio")
    job = manager().submit(session_id, "retranscribe", {**job_payload(body), "take_ids": ids})
    return {"job_id": job["id"], "status": job["status"]}


@app.get("/api/jobs")
async def get_jobs(session_id: str | None = None):
    return Store().list_jobs(session_id)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = Store().get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    await get_job(job_id)
    return await manager().cancel(job_id)


@app.post("/api/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: str, body: dict):
    await get_job(job_id)
    if getattr(app.state, "model_busy", False):
        raise HTTPException(409, "Wait for model setup")
    device = body.get("device")
    if device is not None:
        validate_device(device)
    return manager().retry(job_id, device)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if manager().session_busy(session_id):
        raise HTTPException(409, "Cancel or finish session jobs before deleting")
    session_or_404(session_id).delete()
    _sessions.pop(session_id, None)
    return {"status": "ok"}


@app.post("/api/reset")
@app.post("/api/debug/reset")
async def reset_app():
    require_idle()
    sessions = list_sessions()
    for item in sessions:
        Session.load(item["id"]).delete()
    _sessions.clear()
    Config.save_settings({})
    app.state.setup_complete = False
    return {"status": "ok", "deleted_sessions": len(sessions)}


@app.get("/api/diagnostics")
async def diagnostics():
    return {"app_version": APP_VERSION, "platform": platform.platform(),
            "python_version": sys.version, "is_frozen": Config.is_frozen(),
            "portaudio_available": PORTAUDIO_AVAILABLE,
            "transcriber": app.state.transcriber.get_model_info(),
            "data_dir": str(Config.data_dir()), "sessions_path": str(Config.sessions_path())}


@app.get("/api/debug/info")
async def debug_info():
    if Config.is_frozen() and not os.environ.get("STENO_DEBUG"):
        raise HTTPException(403, "Debug endpoint disabled in production")
    return {**await diagnostics(), "settings": Config.load_settings(),
            "active_sessions": list(_sessions), "session_files": len(list_sessions()),
            "audio": {"recording": _audio_capture.is_recording(), "sample_rate": Config.SAMPLE_RATE}}


@app.get("/api/i18n/{lang}")
async def get_locale(lang: str):
    return load_locale(lang if lang in Config.SUPPORTED_LANGUAGES else Config.DEFAULT_LANGUAGE)


@app.get("/api/languages")
async def get_languages():
    return get_supported_languages()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        session = session_or_404(session_id)
    except HTTPException:
        await websocket.close(code=4004, reason="Session not found")
        return
    await websocket.accept()
    owner = uuid4().hex
    jobs = manager()
    send_lock = asyncio.Lock()

    async def send(message):
        async with send_lock:
            await websocket.send_json(message)

    jobs.subscribe(session_id, send)
    await send(jobs.status(session_id))
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                await send({"type": "error", "message": "Invalid message"})
                continue
            try:
                kind = data.get("type")
                if kind == "ping":
                    await send({"type": "pong"})
                elif kind == "start":
                    if getattr(app.state, "model_busy", False):
                        raise BusyError("Wait for model setup")
                    payload = job_payload(data)
                    mode = data.get("mode", "full")
                    if mode not in {"full", "light"}:
                        raise ValueError("Invalid recording mode")
                    index = data.get("device_index")
                    if index is not None and (not isinstance(index, int) or index < 0):
                        raise ValueError("Invalid microphone")
                    await jobs.start_recording(session, owner, payload["model"], index,
                                               payload["language"], mode, payload["device"])
                elif kind == "stop":
                    await jobs.stop_recording(owner)
                    await send(jobs.status(session_id))
                elif kind == "note":
                    text = data.get("text", "")
                    elapsed = data.get("elapsed_seconds")
                    if not isinstance(text, str) or len(text) > 100_000:
                        raise ValueError("Invalid note")
                    if elapsed is not None and (not isinstance(elapsed, int) or elapsed < 0):
                        raise ValueError("Invalid note timestamp")
                    session.add_note(text, elapsed_seconds=elapsed)
                    await send({"type": "note_saved", "elapsed_seconds": elapsed})
            except (BusyError, AudioCaptureError, ValueError, HTTPException) as exc:
                await send({"type": "error", "message": getattr(exc, "detail", str(exc))})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        jobs.unsubscribe(session_id, send)
        if jobs.recording and jobs.recording["owner"] == owner:
            await jobs.stop_recording(owner)
