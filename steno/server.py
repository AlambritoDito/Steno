"""FastAPI application for Steno."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from steno.audio import AudioCapture, AudioCaptureError
from steno.config import Config
from steno.i18n import load_locale, get_supported_languages
from steno.session import Session, list_sessions
from steno.transcriber import Transcriber

# In-memory state
_sessions: dict[str, Session] = {}
_audio_capture = AudioCapture()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the transcriber (lazy-load, no model loaded yet)."""
    app.state.transcriber = Transcriber()
    yield


app = FastAPI(title="Steno", version="0.1.0", lifespan=lifespan)


# --- Static file serving ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main UI."""
    index_path = Config.static_path() / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


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
    }


@app.get("/api/devices")
async def get_devices():
    """List available microphones."""
    return AudioCapture.list_devices()


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
    tag = session.add_image(image_data, mime_type, caption)
    session.save()
    return {"status": "ok", "tag": tag}


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


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    sessions_dir = Config.sessions_path()
    path = sessions_dir / f"{session_id}.md"

    if not path.exists() and session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if path.exists():
        path.unlink()

    _sessions.pop(session_id, None)
    return {"status": "ok", "message": "Session deleted"}


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

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "start":
                device_index = data.get("device_index")
                if _audio_capture.is_recording():
                    await websocket.send_json({
                        "type": "error",
                        "message": "Already recording",
                    })
                    continue

                chunk_queue: asyncio.Queue = asyncio.Queue()

                def on_chunk(chunk):
                    chunk_queue.put_nowait(chunk)

                try:
                    _audio_capture.start(device_index, on_chunk)
                except AudioCaptureError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
                    continue

                await websocket.send_json({
                    "type": "status",
                    "recording": True,
                    "model_loaded": transcriber.is_loaded(),
                })

                # Process chunks in background
                async def process_chunks():
                    while _audio_capture.is_recording():
                        try:
                            chunk = await asyncio.wait_for(
                                chunk_queue.get(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue

                        text = await transcriber.transcribe(chunk)
                        if text:
                            now = datetime.now()
                            session.add_transcript(text, now)
                            session.save()
                            try:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": text,
                                    "timestamp": now.strftime("%H:%M:%S"),
                                    "is_final": True,
                                })
                            except Exception:
                                break

                asyncio.create_task(process_chunks())

            elif msg_type == "stop":
                _audio_capture.stop()
                await websocket.send_json({
                    "type": "status",
                    "recording": False,
                    "model_loaded": transcriber.is_loaded(),
                })

    except WebSocketDisconnect:
        _audio_capture.stop()
    except Exception:
        _audio_capture.stop()
