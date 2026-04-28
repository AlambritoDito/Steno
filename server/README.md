# Steno Server

LAN/Tailscale transcription service backed by mlx-whisper. Sibling to the [Steno desktop app](../app/) — the two share the Hugging Face model cache and live in the same uv workspace.

Steno Server is designed to run 24/7 on an Apple Silicon Mac (M-series, ≥16 GB RAM recommended for `whisper-large-v3-turbo`). Users upload audio through a web UI; agents (e.g. OpenClaw) call the REST API directly.

## What you get per upload

| Artifact | When | Approx. wall time (1 h audio) |
|---|---|---|
| `transcript-raw.md` | end of Phase 1 | ~6–8 min |
| `transcript-clean.md` (denoise + diarization) | end of Phase 2 | additional ~20–30 min |

Both files include a YAML frontmatter block with the job id, source filename, language, durations, model, and the post-processing applied.

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

For 24/7 deploy, see [`deploy/README-DEPLOY.md`](deploy/README-DEPLOY.md).

## Tests

```bash
uv run --directory server pytest tests/ -v
```

Some tests skip gracefully:

- Real Whisper inference tests skip without `ffmpeg`.
- Real pyannote diarization tests skip without `HF_TOKEN`.

To regenerate the audio fixtures (gitignored .wav files):

```bash
uv run --directory server python tests/generate_fixtures.py
```

The script uses macOS `say -v Paulina` for the Spanish samples and round-trips each through Whisper to confirm intelligibility. See [`tests/fixtures/README.md`](tests/fixtures/README.md) for the manual-recording fallback.

## Configuration

Settings load from environment variables (prefix `STENO_SERVER_`) with sensible defaults. Two are special-cased and read **without** the prefix:

| Variable | Required? | Notes |
|---|---|---|
| `HF_TOKEN` | Yes for Phase 2 (pyannote diarization) | Free token from <https://huggingface.co/settings/tokens> |
| `AUTH_PASSWORD` | Optional | If set, the API gates protected endpoints behind a session cookie |
| `HF_HUB_CACHE` / `HF_HOME` | Optional | Shared with the desktop app at `~/.cache/huggingface/hub` by default |

Other useful overrides:

- `STENO_SERVER_HOST` (default `0.0.0.0`)
- `STENO_SERVER_PORT` (default `8090`)
- `STENO_SERVER_DEFAULT_LANGUAGE` (default `es`)
- `STENO_SERVER_STORAGE_DIR` (default `/tmp/steno-server`)
- `STENO_SERVER_LOG_DIR` (default `~/Library/Logs/steno-server`)
- `STENO_SERVER_JOB_RETENTION_HOURS` (default `24`)
- `STENO_SERVER_MAX_UPLOAD_SIZE_MB` (default `500`)

## Architecture

```
   ┌──────────────────┐
   │  Browser (UI)    │  ← https via Caddy
   │  curl / httpx    │  ← optional cookie auth
   └────────┬─────────┘
            │
   ┌────────▼─────────┐    ┌────────────────────────┐
   │  FastAPI app     │    │  asyncio.Queue worker  │
   │  REST + WS       │───▶│  (single concurrent    │
   │                  │    │   mlx-whisper job)     │
   └────────┬─────────┘    └─────────┬──────────────┘
            │                        │
            │              ┌─────────▼──────────────┐
            │              │  pipeline.run_phase_1  │
            │              │   normalize → VAD →    │
            │              │   Whisper → delooping  │
            │              │   → BoH filter → md    │
            │              └─────────┬──────────────┘
            │                        │
            │              ┌─────────▼──────────────┐
            │              │  pipeline.run_phase_2  │
            │              │   demucs → pyannote →  │
            │              │   re-Whisper → merge   │
            │              │   speakers → md        │
            │              └────────────────────────┘
            │
   ┌────────▼─────────┐
   │  SQLite (jobs)   │  ← jobs.recover_orphaned_jobs() runs at boot
   │  filesystem      │  ← /tmp/steno-server/{job_id}/
   └──────────────────┘
```

### State machine

`queued → phase1_running → phase1_done → phase2_running → done`

Failure branches:

- `phase1_running → failed` (no raw md exists)
- `phase2_running → phase2_failed` (raw md still on disk and downloadable)

Crash recovery on boot reapplies the same rules to in-flight jobs whose worker died (server restart). `phase1_done` jobs that were waiting for Phase 2 are left as `phase1_done` — the user can resubmit if they want Phase 2.

### WebSocket protocol

`GET /ws/jobs/{job_id}` is one-way (server → client) in MVP. Events:

```json
{"type": "queue_position", "job_id": "...", "position": 2}
{"type": "phase1_started", "job_id": "..."}
{"type": "phase1_chunk", "job_id": "...", "text": "...", "start_s": 0.0, "end_s": 30.0, "is_partial": false}
{"type": "phase1_completed", "job_id": "...", "transcript_url": "/api/jobs/.../transcript-raw.md"}
{"type": "phase2_started", "job_id": "...", "step": "denoise"}
{"type": "phase2_progress", "job_id": "...", "step": "diarization", "percent": 45}
{"type": "phase2_completed", "job_id": "...", "transcript_url": "/api/jobs/.../transcript-clean.md"}
{"type": "error", "job_id": "...", "phase": "phase1|phase2", "message": "..."}
```

On reconnect, the server sends the current state (queue position, replay of phase1 chunks if running, terminal events with transcript URLs if done) before resuming live updates.

## Consumo programático (OpenClaw, scripts headless)

The REST API is fully usable headless. Two examples — adjust the host as needed (`http://127.0.0.1:8090` for local; `https://mac-studio.local` over LAN; `https://<your>.ts.net` over Tailscale).

### curl

```bash
HOST=http://127.0.0.1:8090

# (Optional) Login if AUTH_PASSWORD is set. Saves the cookie to /tmp/cookies.txt
# so subsequent calls reuse it.
curl -s -c /tmp/cookies.txt -H 'Content-Type: application/json' \
     -d '{"password":"YOUR_PASSWORD"}' \
     "$HOST/api/auth/login"

# Upload a job. Reuse the cookie if you logged in.
JOB=$(curl -s -b /tmp/cookies.txt -X POST \
     -F "file=@meeting.m4a" \
     -F "language=es" \
     -F "enable_denoise=true" \
     -F "enable_diarization=true" \
     "$HOST/api/jobs")
JOB_ID=$(echo "$JOB" | python -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB_ID"

# Poll until phase1_done (Phase 1 is the fast path).
while :; do
    STATUS=$(curl -s -b /tmp/cookies.txt "$HOST/api/jobs/$JOB_ID" | \
             python -c "import json,sys; print(json.load(sys.stdin)['status'])")
    echo "  status: $STATUS"
    case "$STATUS" in
      phase1_done|done)   break;;
      failed)             echo "Job failed."; exit 1;;
      phase2_failed)      echo "Phase 2 failed but raw md is still available."; break;;
    esac
    sleep 5
done

# Download the raw transcript.
curl -s -b /tmp/cookies.txt -o "transcript-$JOB_ID-raw.md" \
     "$HOST/api/jobs/$JOB_ID/transcript-raw.md"

# Optionally wait for phase2 and download the clean version.
while [[ "$STATUS" != "done" && "$STATUS" != "phase2_failed" ]]; do
    sleep 30
    STATUS=$(curl -s -b /tmp/cookies.txt "$HOST/api/jobs/$JOB_ID" | \
             python -c "import json,sys; print(json.load(sys.stdin)['status'])")
done
if [[ "$STATUS" == "done" ]]; then
    curl -s -b /tmp/cookies.txt -o "transcript-$JOB_ID-clean.md" \
         "$HOST/api/jobs/$JOB_ID/transcript-clean.md"
fi
```

### Python (httpx)

```python
import time
import httpx

HOST = "http://127.0.0.1:8090"
PASSWORD = None  # set to your AUTH_PASSWORD if configured

with httpx.Client(base_url=HOST, timeout=30.0) as client:
    if PASSWORD:
        r = client.post("/api/auth/login", json={"password": PASSWORD})
        r.raise_for_status()  # cookies are now in client.cookies

    with open("meeting.m4a", "rb") as fh:
        r = client.post(
            "/api/jobs",
            files={"file": ("meeting.m4a", fh, "audio/mp4")},
            data={"language": "es", "enable_denoise": "true", "enable_diarization": "true"},
        )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"Job: {job_id}")

    # Poll until at least Phase 1 is done.
    while True:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        print(f"  status: {status}")
        if status in ("phase1_done", "done"):
            break
        if status == "failed":
            raise SystemExit("Job failed")
        if status == "phase2_failed":
            print("Phase 2 failed but raw transcript is downloadable.")
            break
        time.sleep(5)

    raw = client.get(f"/api/jobs/{job_id}/transcript-raw.md")
    raw.raise_for_status()
    open(f"transcript-{job_id}-raw.md", "w").write(raw.text)
```

> **Note on crash recovery:** if the server restarts mid-flight, jobs in `phase2_running` become `phase2_failed` (the raw md is preserved). Headless callers should treat `phase2_failed` as a terminal-but-partial outcome and download the raw md anyway. Jobs in `phase1_running` become `failed`.

## Status

This is v0.1.0. See [the v1.1 backlog issues](https://github.com/AlambritoDito/Steno/issues?q=is%3Aissue+label%3Av1.1) for everything explicitly deferred (browser mic capture, LLM summarization, multi-tenant, more languages, etc.).
