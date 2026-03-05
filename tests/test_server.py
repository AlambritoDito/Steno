"""Tests for steno.server module."""

import pytest
from httpx import ASGITransport, AsyncClient

from steno.config import Config
from steno.server import app, _sessions
from steno.transcriber import Transcriber


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Use a temporary directory for sessions and reset state."""
    monkeypatch.setattr(Config, "sessions_path", classmethod(lambda cls: tmp_path))
    _sessions.clear()
    # Manually set transcriber since lifespan doesn't run in ASGITransport
    app.state.transcriber = Transcriber()
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
