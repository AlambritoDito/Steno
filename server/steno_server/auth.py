"""Cookie-session auth.

Active only when ``settings.auth_password`` is set. When unset, every
endpoint is open (intended for Tailscale-only deploys where the network
boundary is the auth boundary).

Implementation is intentionally minimal: a single shared password, server
side keeps a set of valid session tokens (random 32 bytes, hex-encoded)
with a TTL. No user accounts.
"""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass

from fastapi import Request

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class _Session:
    token: str
    expires_at: float


_sessions: dict[str, _Session] = {}


def auth_enabled() -> bool:
    return bool(settings.auth_password)


def issue_session() -> _Session:
    """Mint a fresh session token; store and return it."""
    _purge_expired()
    token = secrets.token_hex(32)
    expires_at = time.time() + settings.session_duration_hours * 3600
    sess = _Session(token=token, expires_at=expires_at)
    _sessions[token] = sess
    return sess


def revoke_session(token: str | None) -> None:
    if token and token in _sessions:
        del _sessions[token]


def is_valid(token: str | None) -> bool:
    if not token:
        return False
    _purge_expired()
    return token in _sessions


def verify_password(submitted: str) -> bool:
    """Constant-time compare against the configured password."""
    if not settings.auth_password:
        return False
    return hmac.compare_digest(submitted, settings.auth_password)


def _purge_expired() -> None:
    now = time.time()
    expired = [t for t, s in _sessions.items() if s.expires_at < now]
    for t in expired:
        del _sessions[t]


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Endpoints that are always open (no auth even when AUTH_PASSWORD is set).
_OPEN_PREFIXES = (
    "/api/auth/",
    "/api/health",
    "/api/i18n/",
    "/api/languages",
    "/static/",
    "/ws/",  # WebSocket endpoints handle auth via cookie themselves
)


def _is_open(path: str) -> bool:
    if path == "/" or path == "":
        return True
    return any(path.startswith(prefix) for prefix in _OPEN_PREFIXES)


async def auth_middleware(request: Request, call_next):
    """Block request paths that require auth when no valid session cookie present."""
    from fastapi.responses import JSONResponse

    if not auth_enabled():
        return await call_next(request)

    if _is_open(request.url.path):
        return await call_next(request)

    token = request.cookies.get(settings.session_cookie_name)
    if not is_valid(token):
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})
    return await call_next(request)
