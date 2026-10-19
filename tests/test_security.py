import pytest
from httpx import ASGITransport, AsyncClient
from httpx_ws import aconnect_ws, WebSocketDisconnect
from httpx_ws.transport import ASGIWebSocketTransport
from steno.server import app


@pytest.mark.asyncio
async def test_token_protects_root_static_health_and_mutations(monkeypatch):
    monkeypatch.setenv("STENO_AUTH_TOKEN", "launch-secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        for path in ("/", "/static/index.html", "/api/health", "/api/sessions"):
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path, headers={"Authorization": "Bearer launch-secret"})).status_code == 200
        assert (await client.post("/api/reset")).status_code == 401
        client.cookies.set("steno_auth", "launch-secret")
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_rejects_cross_origin_even_with_valid_cookie(monkeypatch):
    monkeypatch.setenv("STENO_AUTH_TOKEN", "secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        client.cookies.set("steno_auth", "secret")
        response = await client.post("/api/sessions/new", json={"name": "bad"}, headers={"Origin": "https://evil.example"})
        assert response.status_code == 403
        assert (await client.get("/api/health", headers={"Host": "evil.example"})).status_code == 403
        assert (await client.get("/api/health", headers={"Origin": "null"})).status_code == 403


@pytest.mark.asyncio
async def test_authenticated_desktop_binds_expected_port(monkeypatch):
    monkeypatch.setenv("STENO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("STENO_PORT", "8877")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:8877", headers={"Authorization": "Bearer secret"}) as client:
        assert (await client.get("/api/health")).status_code == 200
        assert (await client.get("http://127.0.0.1:9999/api/health")).status_code == 403


@pytest.mark.asyncio
async def test_websocket_cannot_bypass_auth(monkeypatch):
    monkeypatch.setenv("STENO_AUTH_TOKEN", "secret")
    async with AsyncClient(transport=ASGIWebSocketTransport(app=app), base_url="http://127.0.0.1") as client:
        with pytest.raises(WebSocketDisconnect):
            async with aconnect_ws("/ws/anything", client):
                pass
