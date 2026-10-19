#!/usr/bin/env bash
# Compatibility entry point; the build itself is shared with Windows.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --frozen --extra build python scripts/build_python.py
