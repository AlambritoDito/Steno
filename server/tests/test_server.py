"""Phase 1 server tests — health endpoint and boot."""

from __future__ import annotations


def test_health_returns_ok(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["uptime_seconds"] >= 0
    assert isinstance(body["version"], str) and body["version"]


def test_health_uptime_advances(client) -> None:
    """Successive /api/health calls should report non-decreasing uptime."""
    first = client.get("/api/health").json()["uptime_seconds"]
    second = client.get("/api/health").json()["uptime_seconds"]
    assert second >= first


def test_unknown_route_404s(client) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
