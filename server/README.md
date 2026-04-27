# Steno Server

LAN/Tailscale transcription service backed by mlx-whisper. Sibling to the [Steno desktop app](../app/). 24/7 service for office staff and headless agents (OpenClaw); produces a fast `transcript-raw.md` followed by a slower `transcript-clean.md` with denoise + diarization.

> **Status:** v0.1.0 in active development. Phase 1 (boot + health) complete; subsequent phases land in this branch.

## Quick start (development)

System prerequisite: ffmpeg (mlx-whisper shells out to it for audio decoding):

```bash
brew install ffmpeg
```

From the **workspace root** (one directory above `server/`):

```bash
uv sync --extra dev                      # one shared .venv for app/ + server/
uv run --directory server main.py        # boots on http://0.0.0.0:8090
```

Verify:

```bash
curl http://127.0.0.1:8090/api/health
# {"status":"ok","version":"0.1.0","uptime_seconds":1.23,"model_loaded":false}
```

## Tests

```bash
uv run --directory server pytest tests/ -v
```

## Configuration

All settings load from environment variables (prefix `STENO_SERVER_`) with sensible defaults. Two are special-cased and read without the prefix:

| Variable | Required? | Notes |
|---|---|---|
| `HF_TOKEN` | Yes for Phase 2 (pyannote diarization) | Free token from huggingface.co |
| `AUTH_PASSWORD` | Optional | If set, the web UI gates access behind a cookie session |
| `HF_HUB_CACHE` / `HF_HOME` | Optional | Shared with the desktop app at `~/.cache/huggingface/hub` by default |

Other useful overrides: `STENO_SERVER_PORT`, `STENO_SERVER_DEFAULT_LANGUAGE`, `STENO_SERVER_STORAGE_DIR`, `STENO_SERVER_LOG_DIR`.

## Architecture (filling in by phase)

Phase 1 only ships `/api/health` and structured logging. Subsequent phases populate the rest of the surface — see the project plan and PR description for milestone breakdown.
