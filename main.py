"""Steno backend entry point, including the desktop process handshake."""

import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading

_dll_directories = []


def prepare_runtime():
    """Use bundled certificates and app-local native DLLs, never system PATH."""
    if getattr(sys, "frozen", False):
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        if sys.platform == "win32":
            import ctypes
            # CTranslate2 also loads DLLs dynamically. Include explicitly added
            # app directories, excluding current-working-directory lookup.
            if not ctypes.windll.kernel32.SetDefaultDllDirectories(0x00001000):
                raise OSError("Could not configure app-local DLL lookup")
            root = Path(sys._MEIPASS)
            for directory in (root, root / "cuda"):
                if directory.is_dir():
                    _dll_directories.append(os.add_dll_directory(str(directory)))


def configure_logging(log_dir):
    from logging.handlers import RotatingFileHandler
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "steno.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


def bind_loopback(port=0):
    """Keep the socket bound until Uvicorn takes ownership (no port race)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", port))
        listener.listen(2048)
        listener.setblocking(False)
        return listener
    except BaseException:
        listener.close()
        raise


def main():
    prepare_runtime()
    from steno.config import Config

    desktop = os.environ.get("STENO_ELECTRON") == "1"
    if desktop and not os.environ.get("STENO_AUTH_TOKEN"):
        raise RuntimeError("Desktop launch requires a session token")
    if getattr(sys, "frozen", False):
        configure_logging(Config.logs_path())
    os.environ.setdefault("HF_HUB_CACHE", str(Config.models_dir()))
    listener = bind_loopback(int(os.environ.get("STENO_PORT", "0" if desktop else "8080")))
    port = listener.getsockname()[1]
    os.environ["STENO_PORT"] = str(port)
    Config.PORT = port

    import uvicorn
    from steno.server import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", access_log=False))
    if desktop:
        def watch_parent():
            # Electron owns stdin. Its exit or explicit stop closes the pipe.
            sys.stdin.buffer.read()
            server.should_exit = True
        threading.Thread(target=watch_parent, daemon=True).start()
        print("STENO_READY " + json.dumps({"port": port}), flush=True)
    else:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
