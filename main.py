"""
Steno — Real-time transcription for classes & meetings.

Run with: uv run main.py
"""

import logging
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


def _setup_logging():
    """Configure logging. Adds a rotating file handler when running as a packaged app."""
    if getattr(sys, "frozen", False):
        from logging.handlers import RotatingFileHandler
        from pathlib import Path

        log_dir = Path.home() / "Documents" / "Steno" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "steno.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

        # Also keep a console handler for stdout/stderr capture by Electron
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        root.addHandler(console)

        logging.info("Steno logging initialized — log file: %s", log_file)
    else:
        # Dev mode: simple console logging (basicConfig is called in server.py)
        pass


_fix_ssl_for_frozen()
_setup_logging()

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
