# Steno (desktop app)

Real-time, fully-offline transcription app for macOS Apple Silicon — Python FastAPI backend + single-file vanilla HTML/CSS/JS frontend, wrapped by an Electron shell. Transcription runs locally via mlx-whisper on the Metal GPU.

This is one of two products in the [Steno](../) monorepo. The sibling [`server/`](../server/) is the LAN/Tailscale web service.

For end-user features, screenshots, and supported models, see the [root README](../README.md).

## Common commands

All commands assume you are at the workspace root (one level above `app/`). The shared `.venv` is created there by `uv sync`.

| Task | Command |
|---|---|
| Install dev dependencies | `uv sync --extra dev` |
| System dep for sounddevice | `brew install portaudio` |
| Run backend (opens browser) | `uv run --directory app main.py` (serves at `http://127.0.0.1:8080`) |
| Run full app (Electron + backend) | `cd app && npm start` |
| All tests | `uv run --directory app pytest tests/ -v` |
| Single test file | `uv run --directory app pytest tests/test_server.py -v` |
| Single test | `uv run --directory app pytest tests/test_server.py::test_name -v` |
| Build PyInstaller backend | `cd app && npm run build:python` → `app/dist/steno-app-backend/` |
| Build .dmg | `cd app && npm run build:electron` |
| Full release build | `cd app && npm run build` |

No linter, formatter, or typechecker is configured. Tests use `pytest` + `pytest-asyncio` + `httpx`/`httpx-ws`.

## Architecture

```
Electron (electron/main.js)
   └─► spawns Python subprocess (detached, env STENO_ELECTRON=1)
         └─► FastAPI server (steno/server.py) on 127.0.0.1:8080
                ├─ REST: /api/status, /api/devices, /api/hardware,
                │        /api/models, /api/sessions/*, /api/i18n/{lang}
                ├─ WS:   /ws/{session_id}  (live transcription)
                └─ serves static/index.html (entire UI)
```

Live transcription data flow:
`mic → sounddevice callback (thread) → asyncio.Queue via call_soon_threadsafe → 5s chunks w/ 0.5s overlap → mlx_whisper on Metal GPU → WebSocket → typing animation`

## Where data lives at runtime

| What | Dev | Packaged |
|---|---|---|
| Sessions (`.md` + `.wav` + images) | `./sessions/` | `~/Documents/Steno/sessions/` |
| Settings (`.steno_settings.json`) | project root | `~/Documents/Steno/` |
| Models (HF Hub layout) | `~/.cache/huggingface/hub/` (or `$HF_HUB_CACHE` / `$HF_HOME/hub`); read-fallback to `~/Documents/Steno/models/` for v0.2.0 users | same |
| Logs | stdout | `~/Documents/Steno/logs/` |

## Things that bite you (non-obvious)

- **Dev vs packaged paths diverge.** `Config.is_frozen()` (= PyInstaller's `sys.frozen`) flips: sessions/settings/logs go to `~/Documents/Steno/...` when packaged, project root when developing. Always go through `Config.sessions_path()` / `Config.data_dir()` — never hardcode. **Models are NOT in `data_dir()`** — they live in the HF cache (see below).
- **HF model cache is shared with the steno-server sibling product.** `Config.models_dir()` returns the HuggingFace Hub cache: `$HF_HUB_CACHE` if set, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`. `app/main.py` honours `HF_HUB_CACHE`/`HF_HOME` before the first `mlx_whisper` import. Re-ordering imports in `main.py` will silently bypass any custom cache the user set.
- **Model cache lookup falls back to legacy locations.** `Config.model_cache_path()` checks the resolved cache first, then `~/Documents/Steno/models/` (legacy v0.2.0), then `~/.cache/huggingface/hub/`.
- **Electron kills the whole process group.** `process.kill(-pid, "SIGTERM")` then SIGKILL after 3s; Python is spawned with `detached: true` specifically so this works.
- **PyInstaller bundle requires ad-hoc signing.** `scripts/build-python.sh` runs `codesign --force --sign -` over every `*.dylib` and `*.so` in `dist/steno-app-backend/`. Without it, the hardened-runtime `.app` will refuse to launch.
- **The frontend is one file by design.** New UI features go into `static/index.html` (and the locale JSONs).
- **i18n is mandatory for user strings.** Every new user-facing string must have a key in both `locales/en.json` and `locales/es.json`.
