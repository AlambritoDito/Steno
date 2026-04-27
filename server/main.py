"""Steno Server entrypoint.

Run with: ``uv run --directory server main.py``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fix_ssl_for_frozen() -> None:
    """Point SSL_CERT_FILE / REQUESTS_CA_BUNDLE at certifi when frozen.

    PyInstaller bundles don't include certifi's CA bundle by default; without
    this, model downloads from huggingface.co fail in packaged builds. The
    server isn't packaged with PyInstaller in Phase 1, but mirroring the
    desktop app's behavior keeps the option open.
    """
    if getattr(sys, "frozen", False):
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


# IMPORTANT: HF_HUB_CACHE must be set BEFORE the first import of mlx_whisper
# (which transitively imports huggingface_hub at module load). The desktop
# app uses the same resolution rule — both products share
# ~/.cache/huggingface/hub by default so whisper-large-v3-turbo isn't
# duplicated on disk.
def _resolve_hf_cache() -> None:
    if "HF_HUB_CACHE" in os.environ:
        return
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
    # else: leave unset; huggingface_hub falls back to ~/.cache/huggingface/hub


_fix_ssl_for_frozen()
_resolve_hf_cache()


from steno_server.config import settings  # noqa: E402
from steno_server.logging_setup import configure_logging, get_logger  # noqa: E402

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    import uvicorn

    from steno_server.server import app

    logger.info(
        "starting_uvicorn",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()
