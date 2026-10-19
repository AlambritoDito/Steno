"""Loopback-only API boundary for the desktop child process."""

import hmac
import os
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


class LocalSecurity:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode("latin1")
        origin = headers.get(b"origin", b"").decode("latin1")
        token = os.environ.get("STENO_AUTH_TOKEN", "")
        hostname = urlsplit("http://" + host).hostname
        error = None
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            error = (403, "Loopback host required")
        expected_origin = "http://" + host
        if origin and origin != expected_origin:
            error = (403, "Origin rejected")
        if token:
            from http.cookies import SimpleCookie
            cookie = SimpleCookie()
            try:
                cookie.load(headers.get(b"cookie", b"").decode("latin1"))
                supplied = cookie.get("steno_auth").value if cookie.get("steno_auth") else ""
            except Exception:
                supplied = ""
            auth = headers.get(b"authorization", b"").decode("latin1")
            if auth.startswith("Bearer "):
                supplied = auth[7:]
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                error = (401, "Local authentication required")
            port = os.environ.get("STENO_PORT")
            if port and host != f"127.0.0.1:{port}":
                error = (403, "Host rejected")
        if error:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 4403, "reason": error[1]})
            return await JSONResponse({"detail": error[1]}, status_code=error[0])(scope, receive, send)

        async def secured_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"cache-control", b"no-store"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; object-src 'none'; frame-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
                ])
            await send(message)
        return await self.app(scope, receive, secured_send)
