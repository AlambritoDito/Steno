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
    --strip \
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
    --collect-all mlx \
    --collect-all mlx_whisper \
    --collect-all sounddevice \
    --collect-all _sounddevice_data \
    --collect-data certifi \
    --hidden-import huggingface_hub \
    --hidden-import huggingface_hub.utils \
    --hidden-import huggingface_hub.utils._errors \
    --hidden-import huggingface_hub.utils._http \
    --hidden-import requests \
    --hidden-import urllib3 \
    --hidden-import httpx \
    --hidden-import httpcore \
    --hidden-import httpcore._backends \
    --hidden-import httpcore._backends.anyio \
    --hidden-import h11 \
    --hidden-import anyio._backends._asyncio \
    --hidden-import socksio \
    --hidden-import certifi \
    --hidden-import filelock \
    --hidden-import tqdm \
    --hidden-import packaging \
    --hidden-import packaging.version \
    --hidden-import packaging.requirements \
    --exclude-module hf_xet \
    --exclude-module hf_transfer \
    --exclude-module tkinter \
    --exclude-module matplotlib \
    --exclude-module scipy \
    --exclude-module pandas \
    --exclude-module PIL \
    --exclude-module cv2 \
    --exclude-module test \
    --exclude-module unittest \
    --exclude-module pydoc \
    --exclude-module xmlrpc \
    --exclude-module lib2to3 \
    main.py

echo ""
echo "==> Post-build cleanup..."
find dist/steno-server -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find dist/steno-server -name '*.pyc' -delete 2>/dev/null || true
find dist/steno-server -name 'tests' -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "==> Ad-hoc signing bundled native libraries..."
echo "    (required for macOS hardened runtime / Gatekeeper)"
SIGNED=0
for ext in dylib so; do
    while IFS= read -r -d '' lib; do
        codesign --force --sign - "$lib" 2>/dev/null && SIGNED=$((SIGNED + 1))
    done < <(find dist/steno-server -name "*.$ext" -print0)
done
echo "    Signed $SIGNED native libraries"

echo ""
echo "==> Build complete!"
BUNDLE_SIZE=$(du -sh dist/steno-server | cut -f1)
echo "    Output: dist/steno-server/ ($BUNDLE_SIZE)"
echo "    Binary: dist/steno-server/steno-server"
echo ""
echo "    Run 'npm run build:electron' next to create the .dmg"
