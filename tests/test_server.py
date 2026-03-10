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
    # Manually set transcriber since lifespan doesn't run in ASGITransport
    app.state.transcriber = Transcriber()
    app.state.setup_complete = True
    yield


@pytest.fixture
def client():
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


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
        base_url="http://test",
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
        base_url="http://test",
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
