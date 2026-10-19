"""Tests for steno.server module."""

import pytest
from httpx import ASGITransport, AsyncClient

from steno.config import Config
from steno.server import app, _sessions, _compare_versions
from steno.transcriber import Transcriber


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Use a temporary directory for sessions and reset state."""
    monkeypatch.setattr(Config, "sessions_path", classmethod(lambda cls: tmp_path))
    _sessions.clear()
    from steno.jobs import JobManager
    app.state.jobs = JobManager()
    app.state.model_busy = False
    # Manually set transcriber since lifespan doesn't run in ASGITransport
    app.state.transcriber = Transcriber()
    app.state.setup_complete = True
    yield


@pytest.fixture
def client():
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://127.0.0.1")


@pytest.mark.asyncio
async def test_get_root_returns_html(client):
    """GET / returns 200 with text/html."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_get_status(client):
    """GET /api/status returns 200 with valid JSON."""
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_get_devices(client):
    """GET /api/devices returns 200 with a list."""
    from unittest.mock import patch
    with patch("steno.server.AudioCapture.list_devices", return_value=[]):
        resp = await client.get("/api/devices")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_create_session(client):
    """POST /api/sessions/new returns 201 with session_id."""
    resp = await client.post("/api/sessions/new", json={"name": "Test Session"})
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data


@pytest.mark.asyncio
async def test_create_session_missing_name(client):
    """Returns 422 when name is missing."""
    resp = await client.post("/api/sessions/new", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_add_note_to_session(client):
    """POST note to existing session returns 200."""
    # Create session first
    resp = await client.post("/api/sessions/new", json={"name": "Note Test"})
    session_id = resp.json()["session_id"]

    # Add note
    resp = await client.post(
        f"/api/sessions/{session_id}/note",
        json={"markdown": "# My Note"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_add_note_invalid_session(client):
    """Returns 404 for unknown session_id."""
    resp = await client.post(
        "/api/sessions/nonexistent/note",
        json={"markdown": "test"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_sessions_list(client):
    """GET /api/sessions returns a list."""
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_export_session_returns_md(client):
    """Response has Content-Disposition: attachment."""
    resp = await client.post("/api/sessions/new", json={"name": "Export Test"})
    session_id = resp.json()["session_id"]

    resp = await client.post(f"/api/sessions/{session_id}/export")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_delete_session(client):
    """DELETE /api/sessions/{id} returns 200."""
    resp = await client.post("/api/sessions/new", json={"name": "Delete Test"})
    session_id = resp.json()["session_id"]

    resp = await client.delete(f"/api/sessions/{session_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_websocket_connect_and_ping(client):
    """WS connects, sends ping, receives pong."""
    # Create a session first
    resp = await client.post("/api/sessions/new", json={"name": "WS Test"})
    session_id = resp.json()["session_id"]

    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://127.0.0.1",
    ) as ws_client:
        async with aconnect_ws(f"/ws/{session_id}", ws_client) as ws:
            # Receive initial status
            data = await ws.receive_json()
            assert data["type"] == "status"

            # Send ping
            await ws.send_json({"type": "ping"})
            data = await ws.receive_json()
            assert data["type"] == "pong"


@pytest.mark.asyncio
async def test_websocket_invalid_session():
    """WS to unknown session closes with error."""
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport

    async with AsyncClient(
        transport=ASGIWebSocketTransport(app=app),
        base_url="http://127.0.0.1",
    ) as ws_client:
        try:
            async with aconnect_ws("/ws/nonexistent", ws_client) as ws:
                # Should receive close
                pass
        except Exception:
            # Connection should be rejected
            pass


# --- v0.2.0: _compare_versions ---


def test_compare_versions_newer():
    """Higher numeric version compares greater."""
    assert _compare_versions("0.2.0", "0.1.0") > 0


def test_compare_versions_older():
    """Lower numeric version compares less."""
    assert _compare_versions("0.1.0", "0.2.0") < 0


def test_compare_versions_equal():
    """Identical versions compare equal."""
    assert _compare_versions("0.2.0", "0.2.0") == 0


def test_compare_versions_release_beats_prerelease():
    """Release (no suffix) is greater than pre-release with same numbers."""
    assert _compare_versions("0.2.0", "0.2.0-alpha.1") > 0


def test_compare_versions_prerelease_less_than_release():
    """Pre-release is less than release with same numbers."""
    assert _compare_versions("0.1.0-alpha.1", "0.1.0") < 0


def test_compare_versions_major_bump():
    """Major version difference outweighs minor/patch."""
    assert _compare_versions("1.0.0", "0.99.99") > 0


# --- v0.2.0: /api/status includes portaudio_available ---


@pytest.mark.asyncio
async def test_status_includes_portaudio_available(client):
    """GET /api/status includes portaudio_available boolean."""
    resp = await client.get("/api/status")
    data = resp.json()
    assert "portaudio_available" in data
    assert isinstance(data["portaudio_available"], bool)


# --- v0.2.0: /api/download-progress ---


@pytest.mark.asyncio
async def test_download_progress_idle(client):
    """Returns idle status when no download is active."""
    import steno.server as server_mod
    server_mod._download_progress.clear()
    resp = await client.get("/api/download-progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "idle"
    assert data["percent"] == 0


@pytest.mark.asyncio
async def test_download_progress_with_active_download(client):
    """Returns correct percentage during an active download."""
    import steno.server as server_mod
    server_mod._download_progress.update({
        "bytes_downloaded": 500,
        "bytes_total": 1000,
        "status": "downloading",
    })
    try:
        resp = await client.get("/api/download-progress")
        data = resp.json()
        assert data["status"] == "downloading"
        assert data["percent"] == 50.0
        assert data["bytes_downloaded"] == 500
        assert data["bytes_total"] == 1000
    finally:
        server_mod._download_progress.clear()


# --- v0.2.0: /api/debug/info production guard ---


@pytest.mark.asyncio
async def test_debug_info_returns_data_in_dev(client, monkeypatch):
    """Debug info is accessible in development mode."""
    monkeypatch.setattr(Config, "is_frozen", staticmethod(lambda: False))
    resp = await client.get("/api/debug/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "app_version" in data


@pytest.mark.asyncio
async def test_debug_info_returns_403_in_production(client, monkeypatch):
    """Debug info is blocked in production without STENO_DEBUG."""
    monkeypatch.setattr(Config, "is_frozen", staticmethod(lambda: True))
    monkeypatch.delenv("STENO_DEBUG", raising=False)
    resp = await client.get("/api/debug/info")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_debug_info_allowed_in_production_with_debug_env(client, monkeypatch):
    """Debug info is accessible in production when STENO_DEBUG is set."""
    monkeypatch.setattr(Config, "is_frozen", staticmethod(lambda: True))
    monkeypatch.setenv("STENO_DEBUG", "1")
    resp = await client.get("/api/debug/info")
    assert resp.status_code == 200


# --- v0.2.0: /api/update-check graceful failure ---


@pytest.mark.asyncio
async def test_update_check_handles_network_error(client, monkeypatch):
    """Returns update_available=False when network fails."""
    import urllib.request

    def mock_urlopen(*args, **kwargs):
        raise ConnectionError("No network")
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    resp = await client.get("/api/update-check")
    assert resp.status_code == 200
    data = resp.json()
    assert data["update_available"] is False
    assert "error" in data


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/api/setup/select-model", "/api/models/active"])
async def test_failed_candidate_keeps_selected_model_setup_and_settings(client, monkeypatch, endpoint):
    from steno import server
    original = app.state.transcriber
    app.state.setup_complete = False
    original_settings = {"model_key": "base", "setup_complete": False}
    Config.save_settings(original_settings)
    candidates = []
    class BrokenCandidate:
        def __init__(self, model_name):
            self.unloaded = False
            candidates.append(self)
        def warmup(self):
            raise RuntimeError("weights cannot load")
        def unload(self):
            self.unloaded = True
        @staticmethod
        def download_model_by_repo(*args):
            pass
    monkeypatch.setattr(server, "Transcriber", BrokenCandidate)
    monkeypatch.setattr(Config, "model_is_downloaded", classmethod(lambda cls, repo: True))
    response = await client.post(endpoint, json={"model_key": "tiny"})
    assert response.status_code == 503
    assert app.state.transcriber is original
    assert not app.state.setup_complete
    assert Config.load_settings() == original_settings
    assert candidates[0].unloaded
    assert not app.state.model_busy


@pytest.mark.asyncio
async def test_model_activation_reserves_runtime_and_publishes_only_after_warmup(client, monkeypatch):
    import asyncio
    import threading
    from steno import server
    entered, release = threading.Event(), threading.Event()
    previous = app.state.transcriber
    class Candidate:
        def __init__(self, model_name):
            self._model_name = model_name
        def warmup(self):
            entered.set()
            assert release.wait(5)
        def unload(self):
            pass
    monkeypatch.setattr(server, "Transcriber", Candidate)
    monkeypatch.setattr(Config, "model_is_downloaded", classmethod(lambda cls, repo: True))
    request = asyncio.create_task(client.post("/api/models/active", json={"model_key": "tiny"}))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert app.state.model_busy
        assert app.state.transcriber is previous
        conflict = await client.post("/api/models/active", json={"model_key": "base"})
        assert conflict.status_code == 409
        conflict = await client.delete("/api/models/base")
        assert conflict.status_code == 409
    finally:
        release.set()
    result = await request
    assert result.status_code == 200
    assert isinstance(app.state.transcriber, Candidate)
    assert Config.load_settings()["model_key"] == "tiny"
    assert not app.state.model_busy


@pytest.mark.parametrize("language", ["english", "en-US", "123", {}, ["en"]])
def test_job_payload_rejects_invalid_language_before_queue(language):
    from fastapi import HTTPException
    from steno.server import job_payload
    with pytest.raises(HTTPException) as exc:
        job_payload({"language": language})
    assert exc.value.status_code == 422


@pytest.mark.parametrize("backend,device", [("mlx", "cpu"), ("mlx", "cuda"), ("faster-whisper", "metal")])
def test_device_validation_matches_platform(monkeypatch, backend, device):
    from fastapi import HTTPException
    from steno import server
    monkeypatch.setattr(server, "default_backend", lambda: backend)
    with pytest.raises(HTTPException) as exc:
        server.job_payload({"device": device})
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_reserves_session_until_source_is_queued(client, monkeypatch):
    import asyncio
    from pathlib import Path
    from steno import server
    from steno.jobs import BusyError
    from steno.session import Session
    from steno.storage import Store
    entered, release = asyncio.Event(), asyncio.Event()
    class Upload:
        filename = "sample.wav"
        delivered = False
        async def read(self, size):
            if self.delivered:
                return b""
            entered.set()
            await release.wait()
            self.delivered = True
            return b"audio source bytes"
        async def close(self):
            pass
    session = Session("upload reservation")
    jobs = app.state.jobs
    monkeypatch.setattr(jobs, "kick", lambda: None)
    uploading = asyncio.create_task(server.transcribe_file(session.session_id, Upload(), None, "auto"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert jobs.busy() and jobs.session_busy(session.session_id)
        assert (await client.delete(f"/api/sessions/{session.session_id}")).status_code == 409
        assert (await client.post("/api/reset")).status_code == 409
        assert (await client.get("/api/mic-test")).status_code == 409
        with pytest.raises(BusyError):
            await jobs.start_recording(session, "owner", "tiny")
        with pytest.raises(BusyError):
            jobs.submit(session.session_id, "retranscribe", {})
    finally:
        release.set()
    result = await uploading
    assert Store().get_job(result["job_id"])["status"] == "queued"
    assert Path(session.takes()[0]["path"]).read_bytes() == b"audio source bytes"
    assert not jobs._uploads


@pytest.mark.asyncio
async def test_empty_upload_releases_reservation(client):
    from steno.session import Session
    session = Session("empty upload")
    response = await client.post(f"/api/sessions/{session.session_id}/transcribe-file",
                                 files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 422
    assert not app.state.jobs.busy()
    assert session.takes()[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_microphone_test_reserves_runtime_against_recording_and_upload(client, monkeypatch):
    import asyncio
    import threading
    import numpy as np
    from httpx_ws import aconnect_ws
    from httpx_ws.transport import ASGIWebSocketTransport
    from steno import server
    from steno.session import Session
    entered, release = threading.Event(), threading.Event()
    def record(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return np.zeros((8000, 1), dtype=np.float32)
    monkeypatch.setattr(server, "PORTAUDIO_AVAILABLE", True)
    monkeypatch.setattr("steno.audio.sd.rec", record)
    session = Session("microphone test")
    testing = asyncio.create_task(client.get("/api/mic-test"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert app.state.model_busy
        assert (await client.post("/api/reset")).status_code == 409
        response = await client.post(f"/api/sessions/{session.session_id}/transcribe-file",
                                     files={"file": ("sample.wav", b"source", "audio/wav")})
        assert response.status_code == 409
        async with AsyncClient(transport=ASGIWebSocketTransport(app=app), base_url="http://127.0.0.1") as ws_client:
            async with aconnect_ws(f"/ws/{session.session_id}", ws_client) as ws:
                await ws.receive_json()
                await ws.send_json({"type": "start"})
                assert (await ws.receive_json())["type"] == "error"
                assert not app.state.jobs.recording
    finally:
        release.set()
    assert (await testing).status_code == 200
    assert not app.state.model_busy


@pytest.mark.asyncio
async def test_diagnostic_export_is_authenticated_and_contains_no_session_data(client, monkeypatch):
    import io
    import json
    import zipfile
    from steno.session import Session
    monkeypatch.setenv("STENO_AUTH_TOKEN", "launch-diagnostic-secret")
    session = Session("private-session-title")
    session.add_note("private-note-body")
    assert (await client.get("/api/diagnostics/export")).status_code == 401
    response = await client.get("/api/diagnostics/export", headers={"Authorization": "Bearer launch-diagnostic-secret"})
    assert response.status_code == 200
    assert "application/zip" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["diagnostic.json"]
        report = json.loads(archive.read("diagnostic.json"))
        assert report["schema_version"] == 1 and report["logs_path"]
        assert "private-session-title" not in str(report)
        assert "private-note-body" not in str(report)
        assert "launch-diagnostic-secret" not in str(report)


@pytest.mark.asyncio
async def test_api_action_logs_only_route_template_and_status(client, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="steno")
    created = await client.post("/api/sessions/new?token=private-query", json={"name": "private-title"})
    session_id = created.json()["session_id"]
    await client.put(f"/api/sessions/{session_id}/note", json={"markdown": "private-note"})
    records = "\n".join(record.getMessage() for record in caplog.records if record.name == "steno")
    assert "event=request method=POST route=/api/sessions/new status=201" in records
    assert "route=/api/sessions/{session_id}/note" in records
    assert session_id not in records
    for private in ("private-query", "private-title", "private-note"):
        assert private not in records
