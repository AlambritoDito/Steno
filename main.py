"""
Steno — Real-time transcription for classes & meetings.

Run with: uv run main.py
"""

import threading
import webbrowser

import uvicorn

from steno.config import Config
from steno.server import app


def _open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://{Config.HOST}:{Config.PORT}")


if __name__ == "__main__":
    print(f"\n  Steno is running at http://{Config.HOST}:{Config.PORT}\n")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
