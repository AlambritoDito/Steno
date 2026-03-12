"""
Steno — Real-time transcription for classes & meetings.

Run with: uv run main.py
"""

import os
import sys
import threading
import webbrowser


def _fix_ssl_for_frozen():
    """Fix SSL certificate path when running inside a PyInstaller bundle.

    PyInstaller bundles don't include certifi's CA bundle by default,
    which causes HTTPS requests (e.g. model downloads) to fail with
    SSL certificate verification errors.
    """
    if getattr(sys, "frozen", False):
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


_fix_ssl_for_frozen()

import uvicorn

from steno.config import Config
from steno.server import app


def _open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://{Config.HOST}:{Config.PORT}")


if __name__ == "__main__":
    print(f"\n  Steno is running at http://{Config.HOST}:{Config.PORT}\n")

    # Don't open browser when running inside Electron
    if not os.environ.get("STENO_ELECTRON"):
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
