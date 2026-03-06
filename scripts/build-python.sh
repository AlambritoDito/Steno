#!/usr/bin/env bash
# -------------------------------------------------------------------
# build-python.sh — Bundle the Steno Python backend with PyInstaller
#
# Creates dist/steno-server/ which electron-builder embeds as an
# extraResource inside the .app bundle.
#
# Prerequisites:
#   brew install portaudio          # required by sounddevice
#   uv sync                        # install Python deps
# -------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "==> Installing PyInstaller..."
uv pip install pyinstaller

echo "==> Cleaning previous build..."
rm -rf dist/steno-server build/steno-server steno-server.spec

echo "==> Building Steno backend..."
uv run pyinstaller \
    --name steno-server \
    --onedir \
    --noconfirm \
    --add-data "static:static" \
    --add-data "locales:locales" \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.http.h11_impl \
    --hidden-import uvicorn.protocols.http.httptools_impl \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.protocols.websockets.websockets_impl \
    --hidden-import uvicorn.protocols.websockets.wsproto_impl \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import uvicorn.lifespan.off \
    --hidden-import steno.server \
    --hidden-import steno.transcriber \
    --hidden-import steno.audio \
    --hidden-import steno.session \
    --hidden-import steno.config \
    --hidden-import steno.i18n \
    --hidden-import multipart \
    --hidden-import aiofiles \
    --hidden-import websockets \
    --hidden-import _sounddevice_data \
    --hidden-import huggingface_hub \
    main.py

echo ""
echo "==> Build complete!"
echo "    Output: dist/steno-server/"
echo "    Binary: dist/steno-server/steno-server"
echo ""
echo "    Run 'npm run build:electron' next to create the .dmg"
