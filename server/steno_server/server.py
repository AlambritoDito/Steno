"""FastAPI app for steno-server."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from . import auth as auth_mod
from . import i18n as i18n_mod
from . import jobs as jobs_mod
from . import storage as storage_mod
from .audio_io import validate_upload
from .config import settings
from .jobs import JobStatus, PipelineOptions
from .logging_setup import configure_logging, get_logger
from .worker import QueuedJob, queue

logger = get_logger(__name__)

_BOOT_TIME: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan: configure logging, init DB, run crash recovery, start worker,
# start the cleanup loop. Mirror in reverse on shutdown.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _BOOT_TIME
    configure_logging()
    _BOOT_TIME = time.monotonic()
    logger.info("server_starting", version=__version__)

    await jobs_mod.init_db()
    report = await jobs_mod.recover_orphaned_jobs()
    if report.failed_phase1 or report.failed_phase2 or report.left_phase1_done:
        logger.warning(
            "orphan_recovery_summary",
            failed_phase1=report.failed_phase1,
            failed_phase2=report.failed_phase2,
            left_phase1_done=report.left_phase1_done,
        )

    await queue.start()
    cleanup_task = asyncio.create_task(storage_mod.run_cleanup_loop(), name="cleanup-loop")

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        await queue.stop()
        logger.info(
            "server_stopping", uptime_seconds=round(time.monotonic() - _BOOT_TIME, 2)
        )


app = FastAPI(
    title="Steno Server",
    version=__version__,
    description="LAN/Tailscale transcription service backed by mlx-whisper.",
    lifespan=lifespan,
)
app.middleware("http")(auth_mod.auth_middleware)

# Static assets and SPA root.
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(index_path, media_type="text/html")


# ---------------------------------------------------------------------------
# /api/health, /api/i18n/{lang}, /api/languages
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    model_loaded: bool


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.monotonic() - _BOOT_TIME, 2),
        model_loaded=queue._transcriber.is_loaded if queue else False,  # noqa: SLF001
    )


@app.get("/api/languages", response_model=list[str])
async def list_languages() -> list[str]:
    return i18n_mod.supported_languages()


@app.get("/api/i18n/{lang}", response_model=dict[str, str])
async def get_i18n(lang: str) -> dict[str, str]:
    if lang not in settings.supported_languages:
        raise HTTPException(status_code=404, detail=f"Unsupported language: {lang}")
    return i18n_mod.load_locale(lang)


# ---------------------------------------------------------------------------
# Auth endpoints (only meaningful when AUTH_PASSWORD is set)
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    password: str


class AuthStatusResponse(BaseModel):
    auth_required: bool
    authenticated: bool


@app.get("/api/auth/status", response_model=AuthStatusResponse)
async def auth_status(request: Request) -> AuthStatusResponse:
    if not auth_mod.auth_enabled():
        return AuthStatusResponse(auth_required=False, authenticated=True)
    token = request.cookies.get(settings.session_cookie_name)
    return AuthStatusResponse(
        auth_required=True,
        authenticated=auth_mod.is_valid(token),
    )


@app.post("/api/auth/login")
async def auth_login(payload: LoginRequest, response: Response) -> JSONResponse:
    if not auth_mod.auth_enabled():
        # Auth not configured — login is a no-op.
        return JSONResponse(content={"ok": True, "auth_required": False})
    if not auth_mod.verify_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    session = auth_mod.issue_session()
    body = JSONResponse(content={"ok": True, "auth_required": True})
    body.set_cookie(
        key=settings.session_cookie_name,
        value=session.token,
        max_age=settings.session_duration_hours * 3600,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
    )
    return body


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    token = request.cookies.get(settings.session_cookie_name)
    auth_mod.revoke_session(token)
    body = JSONResponse(content={"ok": True})
    body.delete_cookie(settings.session_cookie_name)
    return body


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int


@app.post("/api/jobs", status_code=201, response_model=JobCreatedResponse)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form(...),
    enable_denoise: bool = Form(True),
    enable_diarization: bool = Form(True),
) -> JobCreatedResponse:
    if language not in settings.supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File missing filename")

    # Create the job row first; we then save the upload under the new job_id.
    job = await jobs_mod.create_job(
        filename=file.filename,
        language=language,
        options=PipelineOptions(
            enable_denoise=enable_denoise,
            enable_diarization=enable_diarization,
        ),
    )
    saved_path = await storage_mod.save_upload(job.id, file)

    # Validate after writing — validation reads duration/format from disk.
    result = validate_upload(saved_path)
    if not result.ok:
        await jobs_mod.update_status(
            job.id,
            JobStatus.FAILED,
            error_message=f"Upload rejected: {result.error}",
        )
        storage_mod.delete_job_files(job.id)
        raise HTTPException(status_code=400, detail=result.error)

    if result.duration_s is not None:
        await jobs_mod.update_status(
            job.id,
            JobStatus.QUEUED,
            audio_duration_seconds=result.duration_s,
        )

    position = await queue.enqueue(
        QueuedJob(
            job_id=job.id,
            source_path=saved_path,
            language=language,
            options=PipelineOptions(
                enable_denoise=enable_denoise,
                enable_diarization=enable_diarization,
            ),
        )
    )
    return JobCreatedResponse(job_id=job.id, status="queued", queue_position=position)


@app.get("/api/jobs", response_model=list[dict])
async def list_jobs(limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    rows = await jobs_mod.list_jobs(limit=limit)
    return [j.to_dict() for j in rows]


@app.get("/api/jobs/{job_id}", response_model=dict)
async def get_job(job_id: str) -> dict:
    job = await jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    deleted = await jobs_mod.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status_code=204, content=None)


def _serve_transcript(job_id: str, kind: str) -> FileResponse:
    if kind not in ("raw", "clean"):
        raise HTTPException(status_code=404, detail="Unknown transcript kind")
    path: Path = storage_mod.get_transcript_path(job_id, kind)  # type: ignore[arg-type]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript not ready")
    return FileResponse(
        path,
        media_type="text/markdown; charset=utf-8",
        filename=f"transcript-{kind}.md",
    )


@app.get("/api/jobs/{job_id}/transcript-raw.md")
async def get_raw_transcript(job_id: str) -> FileResponse:
    return _serve_transcript(job_id, "raw")


@app.get("/api/jobs/{job_id}/transcript-clean.md")
async def get_clean_transcript(job_id: str) -> FileResponse:
    return _serve_transcript(job_id, "clean")


# ---------------------------------------------------------------------------
# /api/logs/recent — last 200 lines of the JSON log file
# ---------------------------------------------------------------------------


@app.get("/api/logs/recent", response_class=PlainTextResponse)
async def recent_logs(lines: int = Query(200, ge=1, le=1000)) -> str:
    """Return the most recent N lines of the server log file as plain text.

    Used by the UI's 'View logs / Contact IT' modal so support flows can
    capture context without ssh access.
    """
    from .logging_setup import get_log_file

    log_file = get_log_file()
    if not log_file.exists():
        return ""
    # Read tail without slurping huge files into memory.
    with log_file.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= lines:
            read_size = min(block, size)
            f.seek(size - read_size)
            data = f.read(read_size) + data
            size -= read_size
    return b"\n".join(data.splitlines()[-lines:]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# WebSocket: /ws/jobs/{job_id}
# ---------------------------------------------------------------------------


@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    """Stream live progress events for a job.

    On connect, the server sends the current state (queue position,
    replayed chunks, completion event with transcript URLs, or error)
    before resuming live updates. Clients aren't expected to send
    anything in MVP.
    """
    await websocket.accept()

    job = await jobs_mod.get_job(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close(code=1008)
        return

    await queue.register_listener(job_id, websocket)
    for event in queue.reconnect_payload(job_id):
        await websocket.send_json(event)

    try:
        while True:
            # We don't expect inbound messages in MVP; just keep the
            # connection alive and detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        queue.unregister_listener(job_id, websocket)
