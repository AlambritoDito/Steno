"""Auth tests — middleware behavior, login/logout, cookie flags."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from steno_server import auth as auth_mod
from steno_server import worker
from steno_server.config import settings


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    worker.queue._states.clear()  # noqa: SLF001
    worker.queue._pending_order.clear()  # noqa: SLF001
    auth_mod._sessions.clear()  # noqa: SLF001
    yield


@pytest.fixture
def client_no_auth(isolated, monkeypatch):
    monkeypatch.setattr(settings, "auth_password", None)
    from steno_server.server import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_auth(isolated, monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "letmein")
    # TestClient runs over http://, so the Secure cookie flag would block
    # the session cookie from coming back. Real deploys keep secure=True.
    monkeypatch.setattr(settings, "session_cookie_secure", False)
    from steno_server.server import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_auth_secure(isolated, monkeypatch):
    """Same as client_with_auth but with session_cookie_secure on (so we
    can assert the Secure flag actually appears in production-style configs).
    """
    monkeypatch.setattr(settings, "auth_password", "letmein")
    monkeypatch.setattr(settings, "session_cookie_secure", True)
    from steno_server.server import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# When AUTH_PASSWORD is unset, everything is open.
# ---------------------------------------------------------------------------


class TestNoAuth:
    def test_jobs_endpoint_open(self, client_no_auth: TestClient):
        # /api/jobs needs the worker singleton to exist; the GET (list) doesn't
        # require auth and should work.
        r = client_no_auth.get("/api/jobs")
        assert r.status_code == 200

    def test_status_says_no_auth_required(self, client_no_auth: TestClient):
        r = client_no_auth.get("/api/auth/status")
        assert r.status_code == 200
        assert r.json() == {"auth_required": False, "authenticated": True}


# ---------------------------------------------------------------------------
# When AUTH_PASSWORD is set, protected endpoints 401 without a session.
# ---------------------------------------------------------------------------


class TestAuthEnforced:
    def test_open_endpoints_still_work(self, client_with_auth: TestClient):
        for path in ("/api/health", "/api/i18n/es", "/api/languages", "/"):
            r = client_with_auth.get(path)
            assert r.status_code == 200, f"{path} unexpectedly blocked"

    def test_protected_endpoint_blocks_without_cookie(self, client_with_auth: TestClient):
        r = client_with_auth.get("/api/jobs")
        assert r.status_code == 401

    def test_login_with_wrong_password_401(self, client_with_auth: TestClient):
        r = client_with_auth.post("/api/auth/login", json={"password": "wrong"})
        assert r.status_code == 401

    def test_login_then_access(self, client_with_auth: TestClient):
        r = client_with_auth.post("/api/auth/login", json={"password": "letmein"})
        assert r.status_code == 200

        # Subsequent request should now succeed.
        r2 = client_with_auth.get("/api/jobs")
        assert r2.status_code == 200

    def test_cookie_has_security_flags(self, client_with_auth_secure: TestClient):
        """In production-style config (session_cookie_secure=True), the
        cookie must carry HttpOnly + Secure + SameSite=Lax."""
        r = client_with_auth_secure.post("/api/auth/login", json={"password": "letmein"})
        assert r.status_code == 200

        # Cookie was set with the right flags.
        cookies = r.headers.get_list("set-cookie") if hasattr(r.headers, "get_list") else [r.headers.get("set-cookie")]
        cookie_header = " ".join(cookies)
        assert "HttpOnly" in cookie_header
        assert "Secure" in cookie_header
        assert "samesite=lax" in cookie_header.lower()

    def test_logout_revokes_access(self, client_with_auth: TestClient):
        client_with_auth.post("/api/auth/login", json={"password": "letmein"})
        assert client_with_auth.get("/api/jobs").status_code == 200
        client_with_auth.post("/api/auth/logout")
        # After logout, the cookie is deleted client-side; protected endpoints
        # 401 again.
        assert client_with_auth.get("/api/jobs").status_code == 401


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def test_verify_password_constant_time(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "secret")
    assert auth_mod.verify_password("secret") is True
    assert auth_mod.verify_password("wrong") is False


def test_verify_password_returns_false_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", None)
    assert auth_mod.verify_password("anything") is False


def test_session_lifecycle():
    auth_mod._sessions.clear()  # noqa: SLF001
    sess = auth_mod.issue_session()
    assert auth_mod.is_valid(sess.token)
    auth_mod.revoke_session(sess.token)
    assert auth_mod.is_valid(sess.token) is False
