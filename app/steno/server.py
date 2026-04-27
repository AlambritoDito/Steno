"""FastAPI application for Steno."""

import asyncio
import json
import logging
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("steno")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

from steno.audio import AudioCapture, AudioCaptureError, WavWriter, PORTAUDIO_AVAILABLE, _portaudio_error
from steno.config import Config, WHISPER_MODELS, SUPPORTED_AUDIO_EXTENSIONS, _detect_hardware, recommend_model
from steno.i18n import load_locale, get_supported_languages
from steno.session import Session, list_sessions
from steno.transcriber import Transcriber

# In-memory state
_sessions: dict[str, Session] = {}
_audio_capture = AudioCapture()
_download_progress: dict = {}  # Shared download progress state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the transcriber with saved model preference."""
    settings = Config.load_settings()
    model_name = settings.get("model_repo", Config.MODEL_NAME)
    app.state.transcriber = Transcriber(model_name=model_name)
    app.state.setup_complete = settings.get("setup_complete", False)
    yield


APP_VERSION = "0.2.0"

app = FastAPI(title="Steno", version=APP_VERSION, lifespan=lifespan)


# --- Static file serving ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main UI."""
    index_path = Config.static_path() / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

# NOTE: Mount after the "/" route so explicit routes take priority over the catch-all static handler.
app.mount("/static", StaticFiles(directory=str(Config.static_path())), name="static")


# --- API Routes ---

@app.get("/api/status")
async def get_status():
    """Server status: model loaded, active session, devices."""
    transcriber: Transcriber = app.state.transcriber
    active_sessions = list(_sessions.keys())
    return {
        "model_loaded": transcriber.is_loaded(),
        "model_info": transcriber.get_model_info(),
        "active_sessions": active_sessions,
        "recording": _audio_capture.is_recording(),
        "setup_complete": app.state.setup_complete,
        "portaudio_available": PORTAUDIO_AVAILABLE,
        "portaudio_error": _portaudio_error if not PORTAUDIO_AVAILABLE else None,
    }


@app.get("/api/version")
async def get_version():
    """Return the current app version."""
    return {"version": APP_VERSION}


@app.get("/api/diagnostics")
async def get_diagnostics():
    """Return diagnostic info for debugging packaged-app issues."""
    import sys
    import os
    diag = {
        "frozen": getattr(sys, "frozen", False),
        "platform": sys.platform,
        "python_version": sys.version,
        "executable": sys.executable,
        "portaudio_available": PORTAUDIO_AVAILABLE,
        "steno_electron": bool(os.environ.get("STENO_ELECTRON")),
        "home": str(Path.home()),
        "data_dir": str(Config.data_dir()),
        "sessions_path": str(Config.sessions_path()),
        "ssl_cert_file": os.environ.get("SSL_CERT_FILE", "(not set)"),
        "models_dir": str(Config.models_dir()),
        "models_dir_exists": Config.models_dir().exists(),
    }
    # Check if sounddevice can list devices
    try:
        devices = AudioCapture.list_devices()
        diag["audio_devices"] = len(devices)
        diag["audio_error"] = None
    except Exception as e:
        diag["audio_devices"] = 0
        diag["audio_error"] = str(e)
    return diag


@app.get("/api/update-check")
async def check_for_update():
    """Check GitHub Releases for a newer version."""
    import json as _json
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/AlambritoDito/Steno/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "Steno"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())

        latest_tag = data.get("tag_name", "").lstrip("v")
        update_available = _compare_versions(latest_tag, APP_VERSION) > 0

        return {
            "update_available": update_available,
            "current_version": APP_VERSION,
            "latest_version": latest_tag,
            "release_url": data.get("html_url", ""),
            "release_notes": (data.get("body", "") or "")[:500],
        }
    except Exception as e:
        logger.warning("Update check failed: %s", e)
        return {"update_available": False, "current_version": APP_VERSION, "error": str(e)}


def _compare_versions(a: str, b: str) -> int:
    """Compare two version strings. Returns >0 if a > b.

    Handles pre-release suffixes: 0.2.0 > 0.1.0-alpha.1
    """
    def parse(v: str) -> tuple:
        # Split on hyphen: "0.1.0-alpha.1" -> ("0.1.0", "alpha.1")
        parts = v.split("-", 1)
        nums = tuple(int(x) for x in parts[0].split(".") if x.isdigit())
        pre = parts[1] if len(parts) > 1 else ""
        return nums, pre

    a_nums, a_pre = parse(a)
    b_nums, b_pre = parse(b)

    if a_nums != b_nums:
        return 1 if a_nums > b_nums else -1
    # Same numeric version: release (no pre) > pre-release
    if not a_pre and b_pre:
        return 1
    if a_pre and not b_pre:
        return -1
    return 0


@app.get("/api/devices")
async def get_devices():
    """List available microphones."""
    return AudioCapture.list_devices()


@app.get("/api/hardware")
async def get_hardware():
    """Detect hardware and return model recommendations."""
    hw = await asyncio.to_thread(_detect_hardware)
    recommended = recommend_model(hw["ram_gb"])

    models = []
    for key, info in WHISPER_MODELS.items():
        models.append({
            "key": key,
            "repo": info["repo"],
            "size_mb": info["size_mb"],
            "quality": info["quality"],
            "speed": info["speed"],
            "min_ram_gb": info["min_ram_gb"],
            "recommended": key == recommended,
            "compatible": hw["ram_gb"] >= info["min_ram_gb"],
        })

    return {
        "chip": hw["chip"],
        "ram_gb": hw["ram_gb"],
        "recommended_model": recommended,
        "models": models,
    }


@app.post("/api/setup/select-model")
async def select_model(body: dict):
    """Select and download a model. Streams progress via the response."""
    model_key = body.get("model_key")
    if model_key not in WHISPER_MODELS:
        raise HTTPException(status_code=422, detail="Invalid model key")

    model_info = WHISPER_MODELS[model_key]
    model_repo = model_info["repo"]

    # Update transcriber with selected model
    transcriber: Transcriber = app.state.transcriber
    transcriber.set_model(model_repo)

    # Download/cache the model with progress tracking
    _download_progress.clear()
    try:
        await asyncio.to_thread(transcriber.download_model, _download_progress)
    except Exception as e:
        logger.error("Model download failed: %s — %s", model_repo, e)
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    # Save settings and track as downloaded
    settings = Config.load_settings()
    settings["model_key"] = model_key
    settings["model_repo"] = model_repo
    settings["setup_complete"] = True
    downloaded = settings.get("downloaded_models", [])
    if model_key not in downloaded:
        downloaded.append(model_key)
        settings["downloaded_models"] = downloaded
    Config.save_settings(settings)
    app.state.setup_complete = True

    return {
        "status": "ok",
        "model_key": model_key,
        "model_repo": model_repo,
    }


@app.get("/api/models")
async def get_models():
    """List all models with download status.

    Auto-detects models present in the HuggingFace cache even if they
    aren't tracked in settings (self-healing after resets / first runs).
    """
    settings = Config.load_settings()
    downloaded = set(settings.get("downloaded_models", []))
    active_repo = app.state.transcriber._model_name
    settings_dirty = False

    models = []
    for key, info in WHISPER_MODELS.items():
        # Check both settings list AND physical HF cache
        on_disk = Config.model_cache_path(info["repo"]) is not None
        is_downloaded = key in downloaded or on_disk

        # Self-heal: sync settings if model is on disk but not tracked
        if on_disk and key not in downloaded:
            downloaded.add(key)
            settings_dirty = True

        disk_mb = 0.0
        if is_downloaded:
            disk_mb = Config.model_cache_size_mb(info["repo"])
        models.append({
            "key": key,
            "repo": info["repo"],
            "size_mb": info["size_mb"],
            "disk_mb": disk_mb,
            "quality": info["quality"],
            "speed": info["speed"],
            "downloaded": is_downloaded,
            "active": info["repo"] == active_repo,
        })

    # Persist any self-heal updates
    if settings_dirty:
        settings["downloaded_models"] = sorted(downloaded)
        Config.save_settings(settings)

    return {"models": models, "active_repo": active_repo}


@app.post("/api/models/download")
async def download_model(body: dict):
    """Download a model without setting it as active."""
    model_key = body.get("model_key")
    if model_key not in WHISPER_MODELS:
        raise HTTPException(status_code=422, detail="Invalid model key")

    model_info = WHISPER_MODELS[model_key]
    repo = model_info["repo"]

    try:
        _download_progress.clear()
        await asyncio.to_thread(Transcriber.download_model_by_repo, repo, _download_progress)
    except Exception as e:
        logger.error("Model download failed: %s — %s", repo, e)
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    # Track downloaded models
    settings = Config.load_settings()
    downloaded = settings.get("downloaded_models", [])
    if model_key not in downloaded:
        downloaded.append(model_key)
        settings["downloaded_models"] = downloaded
        Config.save_settings(settings)

    return {"status": "ok", "model_key": model_key}


@app.get("/api/download-progress")
async def get_download_progress():
    """Poll current model download progress."""
    if not _download_progress:
        return {"status": "idle", "bytes_downloaded": 0, "bytes_total": 0, "percent": 0}
    total = _download_progress.get("bytes_total", 0)
    downloaded = _download_progress.get("bytes_downloaded", 0)
    percent = round((downloaded / total * 100), 1) if total > 0 else 0
    return {
        "status": _download_progress.get("status", "idle"),
        "bytes_downloaded": downloaded,
        "bytes_total": total,
        "percent": percent,
    }


@app.post("/api/models/active")
async def set_active_model(body: dict):
    """Switch the active transcription model."""
    model_key = body.get("model_key")
    if model_key not in WHISPER_MODELS:
        raise HTTPException(status_code=422, detail="Invalid model key")

    model_info = WHISPER_MODELS[model_key]
    repo = model_info["repo"]

    transcriber: Transcriber = app.state.transcriber
    transcriber.set_model(repo)

    # Save preference
    settings = Config.load_settings()
    settings["model_key"] = model_key
    settings["model_repo"] = repo
    Config.save_settings(settings)

    return {"status": "ok", "model_key": model_key, "model_repo": repo}


@app.delete("/api/models/{model_key}")
async def delete_model(model_key: str):
    """Delete a downloaded model to free disk space."""
    if model_key not in WHISPER_MODELS:
        raise HTTPException(status_code=422, detail="Invalid model key")

    model_info = WHISPER_MODELS[model_key]
    repo = model_info["repo"]

    # Cannot delete the active model
    if repo == app.state.transcriber._model_name:
        raise HTTPException(status_code=409, detail="Cannot delete the active model")

    deleted = await asyncio.to_thread(Config.delete_model_cache, repo)

    # Remove from downloaded list in settings
    settings = Config.load_settings()
    downloaded = settings.get("downloaded_models", [])
    if model_key in downloaded:
        downloaded.remove(model_key)
        settings["downloaded_models"] = downloaded
        Config.save_settings(settings)

    return {"status": "ok", "deleted": deleted, "model_key": model_key}


@app.get("/api/mic-test")
async def mic_test():
    """Test microphone access by reading a short sample."""
    import numpy as np
    import sounddevice as sd

    try:
        # Record 0.5s of audio to test mic access
        audio = await asyncio.to_thread(
            sd.rec, int(0.5 * Config.SAMPLE_RATE), samplerate=Config.SAMPLE_RATE,
            channels=1, dtype="float32", blocking=True
        )
        # Synchronous call needed after sd.rec with blocking=True
        rms = float(np.sqrt(np.mean(audio**2)))
        return {"status": "ok", "rms": rms, "has_audio": rms > 0.001}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/sessions/{session_id}/transcribe-file")
async def transcribe_file(session_id: str, file: UploadFile = File(...)):
    """Upload an audio file and transcribe it into the session."""
    session = _sessions.get(session_id)
    if session is None:
        try:
            session = Session.load(session_id)
            _sessions[session_id] = session
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    # Validate file extension
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format. Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
        )

    # Write to temp file for mlx-whisper (it needs a file path)
    audio_data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        transcriber: Transcriber = app.state.transcriber
        segments = await asyncio.to_thread(transcriber.transcribe_file, tmp_path)

        # Add each segment to the session
        for seg in segments:
            if seg["text"]:
                minutes = int(seg["start"]) // 60
                seconds = int(seg["start"]) % 60
                ts = datetime.now().replace(minute=minutes % 60, second=seconds)
                session.add_transcript(seg["text"], ts)

        session.save()
        return {"status": "ok", "segments": segments, "count": len(segments)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/sessions")
async def get_sessions():
    """List saved sessions."""
    return list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Download session as .md."""
    sessions_dir = Config.sessions_path()
    path = sessions_dir / f"{session_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(
        path,
        media_type="text/markdown",
        filename=f"{session_id}.md",
    )


@app.post("/api/sessions/new", status_code=201)
async def create_session(body: dict):
    """Create a new session."""
    name = body.get("name")
    if not name:
        raise HTTPException(status_code=422, detail="Session name is required")

    session = Session(name=name)
    _sessions[session.session_id] = session
    session.save()

    return {
        "session_id": session.session_id,
        "name": session.name,
        "created_at": session.created_at.isoformat(),
    }


@app.post("/api/sessions/{session_id}/note")
async def add_note(session_id: str, body: dict):
    """Add a note to a session."""
    session = _sessions.get(session_id)
    if session is None:
        # Try loading from disk
        try:
            session = Session.load(session_id)
            _sessions[session_id] = session
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    markdown = body.get("markdown", "")
    session.add_note(markdown)
    session.save()
    return {"status": "ok", "message": "Note added"}


@app.post("/api/sessions/{session_id}/image")
async def add_image(session_id: str, file: UploadFile = File(...), caption: str = Form("")):
    """Upload an image to a session."""
    session = _sessions.get(session_id)
    if session is None:
        try:
            session = Session.load(session_id)
            _sessions[session_id] = session
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    image_data = await file.read()
    mime_type = file.content_type or "image/png"
    result = session.add_image(image_data, mime_type, caption)
    session.save()
    return {"status": "ok", "tag": result["tag"], "image_url": result["image_url"]}


@app.get("/api/sessions/{session_id}/images/{filename}")
async def get_session_image(session_id: str, filename: str):
    """Serve an image file from a session."""
    images_dir = Config.images_path(session_id)
    image_path = images_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    # Determine media type from extension
    ext = Path(filename).suffix.lower()
    media_types = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    }
    return FileResponse(image_path, media_type=media_types.get(ext, "image/png"))


@app.post("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export session as downloadable .md."""
    session = _sessions.get(session_id)
    if session is None:
        try:
            session = Session.load(session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    content = session.to_markdown()
    return Response(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{session_id}.md"',
        },
    )


@app.get("/api/sessions/{session_id}/audio")
async def get_session_audio(session_id: str):
    """Serve the saved WAV file for a session."""
    sessions_dir = Config.sessions_path()
    wav_path = sessions_dir / f"{session_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename=f"{session_id}.wav",
    )


@app.post("/api/sessions/{session_id}/retranscribe")
async def retranscribe_session(session_id: str, body: dict):
    """Re-transcribe a session's saved audio with a (potentially different) model."""
    # Load the session
    session = _sessions.get(session_id)
    if session is None:
        try:
            session = Session.load(session_id)
            _sessions[session_id] = session
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")

    # Check that audio exists
    wav_path = session.audio_path()
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="No saved audio for this session")

    # Optionally use a different model
    model_key = body.get("model_key")
    if model_key:
        if model_key not in WHISPER_MODELS:
            raise HTTPException(status_code=422, detail="Invalid model key")
        model_repo = WHISPER_MODELS[model_key]["repo"]
        retranscriber = Transcriber(model_name=model_repo)
    else:
        retranscriber = app.state.transcriber

    try:
        segments = await asyncio.to_thread(retranscriber.transcribe_file, str(wav_path))

        # Clear existing transcript entries and replace with new ones
        session._entries = [e for e in session._entries if e["type"] != "transcript"]
        for seg in segments:
            if seg["text"]:
                minutes = int(seg["start"]) // 60
                seconds = int(seg["start"]) % 60
                ts = session.created_at.replace(
                    minute=minutes % 60, second=seconds
                )
                session.add_transcript(seg["text"], ts)

        session.save()
        return {"status": "ok", "segments": segments, "count": len(segments)}
    except Exception as e:
        logger.error("Re-transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Re-transcription failed: {e}")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated files (audio, images)."""
    import shutil

    sessions_dir = Config.sessions_path()
    path = sessions_dir / f"{session_id}.md"

    if not path.exists() and session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if path.exists():
        path.unlink()

    # Clean up WAV file
    wav_path = sessions_dir / f"{session_id}.wav"
    if wav_path.exists():
        wav_path.unlink()

    # Clean up images directory
    images_dir = sessions_dir / "images" / session_id
    if images_dir.exists():
        shutil.rmtree(images_dir)

    _sessions.pop(session_id, None)
    return {"status": "ok", "message": "Session deleted"}


@app.post("/api/reset")
@app.post("/api/debug/reset")
async def reset_app():
    """Reset the app: delete all sessions and settings, clear in-memory state."""
    # Stop any active recording
    if _audio_capture.is_recording():
        _audio_capture.stop()

    # Clear in-memory sessions
    _sessions.clear()

    # Delete all session files (md + wav)
    import shutil as _shutil
    sessions_dir = Config.sessions_path()
    deleted_sessions = 0
    for f in sessions_dir.glob("*.md"):
        f.unlink()
        deleted_sessions += 1
    for f in sessions_dir.glob("*.wav"):
        f.unlink()
    # Delete all session images
    images_dir = sessions_dir / "images"
    if images_dir.exists():
        _shutil.rmtree(images_dir)

    # Reset settings (keep setup_complete=False to trigger setup wizard)
    Config.save_settings({})
    app.state.setup_complete = False

    # Reset transcriber to default model
    app.state.transcriber = Transcriber(model_name=Config.MODEL_NAME)

    return {
        "status": "ok",
        "deleted_sessions": deleted_sessions,
        "message": "App reset complete. Restart recommended.",
    }


@app.get("/api/debug/info")
async def debug_info():
    """Return debug information about the current app state.

    Guarded in production builds — returns 403 unless STENO_DEBUG is set.
    """
    import os
    import sys
    import platform

    if Config.is_frozen() and not os.environ.get("STENO_DEBUG"):
        raise HTTPException(status_code=403, detail="Debug endpoint disabled in production")

    transcriber: Transcriber = app.state.transcriber
    settings = Config.load_settings()

    # Check model cache directories
    cached_models = []
    models_dir = Config.models_dir()
    # Check custom models dir and fallback HF cache
    for cache_dir in [models_dir, Path.home() / ".cache" / "huggingface" / "hub"]:
        if cache_dir.exists():
            for d in cache_dir.iterdir():
                if d.is_dir() and d.name.startswith("models--"):
                    name = d.name.replace("models--", "").replace("--", "/")
                    # Avoid duplicates
                    if any(m["name"] == name for m in cached_models):
                        continue
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    cached_models.append({
                        "name": name,
                        "size_mb": round(size / (1024 * 1024), 1),
                        "location": str(cache_dir),
                    })

    return {
        "app_version": APP_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "is_frozen": Config.is_frozen(),
        "data_dir": str(Config.data_dir()),
        "sessions_path": str(Config.sessions_path()),
        "settings_path": str(Config.settings_path()),
        "settings": settings,
        "transcriber": {
            "model_name": transcriber._model_name,
            "loaded": transcriber._loaded,
            "language": transcriber._language,
        },
        "audio": {
            "recording": _audio_capture.is_recording(),
            "sample_rate": Config.SAMPLE_RATE,
            "chunk_duration": Config.CHUNK_DURATION,
            "silence_threshold": Config.SILENCE_THRESHOLD,
        },
        "active_sessions": list(_sessions.keys()),
        "session_files": len(list(Config.sessions_path().glob("*.md"))),
        "cached_models": cached_models,
    }


@app.get("/api/i18n/{lang}")
async def get_locale(lang: str):
    """Return locale JSON for given language code."""
    return load_locale(lang)


@app.get("/api/languages")
async def get_languages():
    """Return list of supported languages."""
    return get_supported_languages()


# --- WebSocket ---

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main real-time channel for transcription."""
    session = _sessions.get(session_id)
    if session is None:
        try:
            session = Session.load(session_id)
            _sessions[session_id] = session
        except FileNotFoundError:
            await websocket.close(code=4004, reason="Session not found")
            return

    await websocket.accept()
    transcriber: Transcriber = app.state.transcriber

    # Send initial status
    await websocket.send_json({
        "type": "status",
        "recording": _audio_capture.is_recording(),
        "model_loaded": transcriber.is_loaded(),
    })

    async def status_callback(status: str):
        """Send model status updates to the client."""
        try:
            await websocket.send_json({"type": status})
        except Exception:
            pass

    transcriber._status_callback = status_callback

    wav_writer = None
    chunk_task = None
    recording_start_time = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "start":
                device_index = data.get("device_index")
                language = data.get("language")  # None = auto-detect
                mode = data.get("mode", "full")  # "full" or "light"
                logger.info("Recording start requested, device_index=%s, language=%s, mode=%s", device_index, language, mode)

                # Apply transcription language
                transcriber._language = language

                if _audio_capture.is_recording():
                    logger.warning("Already recording, ignoring start request")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Already recording",
                    })
                    continue

                # Create WAV writer for audio archival
                wav_path = session.audio_path()
                wav_writer = WavWriter(wav_path, sample_rate=Config.SAMPLE_RATE)
                recording_start_time = datetime.now()

                chunk_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def on_chunk(chunk):
                    # sounddevice callback runs in a separate thread;
                    # asyncio.Queue is NOT thread-safe, so schedule via the event loop
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)

                try:
                    _audio_capture.start(device_index, on_chunk, wav_writer=wav_writer)
                    logger.info("Audio capture started successfully (mode=%s)", mode)
                except AudioCaptureError as e:
                    logger.error("Audio capture failed: %s", e)
                    wav_writer.close()
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
                    continue

                await websocket.send_json({
                    "type": "status",
                    "recording": True,
                    "model_loaded": transcriber.is_loaded(),
                    "mode": mode,
                })

                # Process chunks in background
                chunk_task = None

                if mode == "light":
                    # Light mode: record only, no transcription
                    async def process_chunks():
                        chunk_count = 0
                        while _audio_capture.is_recording():
                            try:
                                chunk = await asyncio.wait_for(
                                    chunk_queue.get(), timeout=1.0
                                )
                            except asyncio.TimeoutError:
                                pass
                            chunk_count += 1

                            # Send elapsed time updates every second
                            elapsed = (datetime.now() - recording_start_time).total_seconds()
                            try:
                                await websocket.send_json({
                                    "type": "elapsed",
                                    "seconds": int(elapsed),
                                })
                            except Exception:
                                break

                        logger.info("Light mode recording ended after %d chunks", chunk_count)
                else:
                    # Full mode: transcribe + save WAV
                    async def process_chunks():
                        chunk_count = 0
                        while _audio_capture.is_recording():
                            try:
                                chunk = await asyncio.wait_for(
                                    chunk_queue.get(), timeout=1.0
                                )
                            except asyncio.TimeoutError:
                                continue

                            chunk_count += 1
                            import numpy as np
                            rms = float(np.sqrt(np.mean(chunk**2)))
                            logger.info("Chunk #%d received, RMS=%.4f, len=%d", chunk_count, rms, len(chunk))

                            try:
                                text = await transcriber.transcribe(chunk)
                            except Exception as e:
                                logger.error("Transcription error on chunk #%d: %s", chunk_count, e)
                                try:
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": f"Transcription error: {e}",
                                    })
                                except Exception:
                                    pass
                                continue

                            if text:
                                logger.info("Transcribed: %s", text[:80])
                                now = datetime.now()
                                # Use append-only save during recording (Issue #6/#14)
                                session.append_transcript(text, now)
                                try:
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "text": text,
                                        "timestamp": now.strftime("%H:%M:%S"),
                                        "is_final": True,
                                    })
                                except Exception:
                                    break
                            else:
                                logger.debug("Chunk #%d was silence (RMS=%.4f)", chunk_count, rms)

                        logger.info("Chunk processing loop ended after %d chunks", chunk_count)

                chunk_task = asyncio.create_task(process_chunks())

            elif msg_type == "note":
                # Timestamped note (used in light mode)
                note_text = data.get("text", "")
                elapsed = data.get("elapsed_seconds")
                if note_text.strip():
                    session.add_note(note_text, elapsed_seconds=elapsed)
                    session.save()
                    await websocket.send_json({
                        "type": "note_saved",
                        "text": note_text,
                        "elapsed_seconds": elapsed,
                    })

            elif msg_type == "stop":
                logger.info("Recording stop requested")
                _audio_capture.stop()
                # Close the WAV writer
                if wav_writer is not None:
                    wav_writer.close()
                    wav_writer = None
                # Wait for chunk processing to finish
                if chunk_task is not None:
                    try:
                        await chunk_task
                    except asyncio.CancelledError:
                        pass
                    chunk_task = None
                # Write canonical full file now that recording stopped
                session.save()
                await websocket.send_json({
                    "type": "status",
                    "recording": False,
                    "model_loaded": transcriber.is_loaded(),
                    "has_audio": session.has_audio(),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
        _audio_capture.stop()
        if wav_writer is not None:
            wav_writer.close()
        if chunk_task is not None:
            chunk_task.cancel()
            try:
                await chunk_task
            except asyncio.CancelledError:
                pass
        # Write canonical file and evict session from memory
        if session_id in _sessions:
            _sessions[session_id].save()
            del _sessions[session_id]
    except Exception as e:
        logger.error("WebSocket error for session %s: %s", session_id, e)
        _audio_capture.stop()
        if wav_writer is not None:
            wav_writer.close()
        if chunk_task is not None:
            chunk_task.cancel()
            try:
                await chunk_task
            except asyncio.CancelledError:
                pass
        if session_id in _sessions:
            _sessions[session_id].save()
            del _sessions[session_id]
