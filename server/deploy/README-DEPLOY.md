# Deploying Steno Server

This guide installs Steno Server as a long-running service on a Mac Studio (or similar Apple Silicon Mac) and exposes it on the LAN and over Tailscale.

## What you'll end up with

- The Python backend running 24/7 as a user LaunchAgent (auto-restart on crash, auto-start on login).
- Caddy reverse-proxying HTTPS on the LAN (`mac-studio.local`) and Tailscale (`alans-mac-studio.tailcbc049.ts.net`) to the backend on `127.0.0.1:8090`.

## Prerequisites

```bash
# uv (Python package manager).
brew install uv

# ffmpeg (mlx-whisper requires it for audio decoding).
brew install ffmpeg

# Caddy (reverse proxy with automatic HTTPS).
brew install caddy

# (Optional) Tailscale, if you don't already have it.
brew install --cask tailscale
```

You also need a Hugging Face account with a token (free) so Phase 2 can download `pyannote/speaker-diarization-3.1`. Visit <https://huggingface.co/settings/tokens> to create one.

## 1. Clone and set up the repo

```bash
cd ~/Development
git clone https://github.com/AlambritoDito/Steno.git
cd Steno
uv sync --extra dev
```

A `.venv` is created at the workspace root. Confirm the server boots:

```bash
uv run --directory server main.py
# Ctrl+C to stop. Server should reach http://127.0.0.1:8090/api/health.
```

## 2. Install the LaunchAgent

```bash
bash server/deploy/install-launchagent.sh
```

The installer prompts you for:

- Path to `uv` (auto-detected).
- Path to the repo (auto-detected).
- `HF_TOKEN` — paste the token from Hugging Face. Press Enter to skip if you don't need Phase 2.
- `AUTH_PASSWORD` — set this if you want the LAN UI to require a password. Leave empty for an open Tailscale-only deploy.

After install, the agent is registered in `~/Library/LaunchAgents/com.steno.server.plist`. Useful commands:

```bash
# Check it's running.
launchctl list | grep com.steno.server

# Tail the JSON server log.
tail -f ~/Library/Logs/steno-server/server.log

# Stop the agent.
launchctl unload -w ~/Library/LaunchAgents/com.steno.server.plist

# Re-load it (after env changes).
launchctl load -w ~/Library/LaunchAgents/com.steno.server.plist
```

Verify auto-restart:

```bash
pkill -f "steno_server"        # find the python process and kill it
launchctl list | grep com.steno.server   # PID should change within ~10 s
```

## 3. Set up Caddy

Caddy needs to listen on ports 80 and 443, which requires running it as a system daemon (LaunchDaemon). Homebrew handles this:

```bash
sudo brew services start caddy
```

Edit `/opt/homebrew/etc/Caddyfile` (Apple Silicon) using `server/deploy/Caddyfile.example` as a template:

```bash
sudo cp server/deploy/Caddyfile.example /opt/homebrew/etc/Caddyfile
sudo $EDITOR /opt/homebrew/etc/Caddyfile     # adjust hostnames
sudo brew services restart caddy
```

### LAN hostname (Bonjour)

Caddy issues a self-signed certificate via the `tls internal` directive. Each LAN client needs to trust it the first time:

- macOS: visit `https://mac-studio.local/` in Safari, click "Show Details" → "visit this website", and accept.
- iOS: visit the URL, then go to **Settings → General → About → Certificate Trust Settings** and enable trust for the cert that was added.

Or install Caddy's local CA on each client (`caddy trust` on the server makes the CA available; copy it manually to clients).

### Tailscale hostname

Caddy auto-provisions a Let's Encrypt cert for the public Tailscale hostname (the `*.ts.net` domain). No client trust step needed — modern OSes already trust Let's Encrypt.

Tailscale must be running on the Mac Studio (`tailscale up`) and on the client device.

## 4. Verify end-to-end

From a different LAN device:

```
https://mac-studio.local/
```

From outside the LAN with Tailscale on:

```
https://alans-mac-studio.tailcbc049.ts.net/
```

Both should show the upload UI. Try uploading a short audio file and confirm the streaming transcription appears.

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Agent isn't running after install | `~/Library/Logs/steno-server/stderr.log` (LaunchAgent stderr) |
| Server starts but never responds | `~/Library/Logs/steno-server/server.log` (structured JSON logs) |
| Phase 2 fails with "HF_TOKEN is not set" | Re-run `install-launchagent.sh` and provide the token |
| `mac-studio.local` doesn't resolve | Bonjour/mDNS issue — confirm with `dns-sd -B _http._tcp` |
| Caddy can't bind to 443 | Make sure no other process holds the port: `sudo lsof -i :443` |
| Tailscale URL gives a cert error | Confirm `caddy` is running and the hostname matches your Tailscale machine name exactly |

## Updating the server

```bash
cd ~/Development/Steno
git pull
uv sync --extra dev
launchctl unload -w ~/Library/LaunchAgents/com.steno.server.plist
launchctl load -w ~/Library/LaunchAgents/com.steno.server.plist
```

If `pyproject.toml` changed, the `uv sync` step picks up the new deps. If `com.steno.server.plist` template changed, re-run `install-launchagent.sh`.
