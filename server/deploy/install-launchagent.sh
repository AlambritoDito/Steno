#!/usr/bin/env bash
# install-launchagent.sh — Install Steno Server as a user LaunchAgent.
#
# Renders com.steno.server.plist with the values you provide, copies it
# into ~/Library/LaunchAgents/, and loads it via launchctl. The agent
# starts at login and is restarted automatically by launchd if it crashes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/com.steno.server.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$LAUNCH_AGENTS_DIR/com.steno.server.plist"
LOG_DIR="$HOME/Library/Logs/steno-server"

if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "Cannot find plist template at $PLIST_TEMPLATE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Prompt for parameters.
# ---------------------------------------------------------------------------

prompt() {
    local var_name=$1
    local prompt_text=$2
    local default_value=${3:-}
    local current_value
    if [[ -n "$default_value" ]]; then
        read -r -p "$prompt_text [$default_value]: " current_value
        current_value="${current_value:-$default_value}"
    else
        read -r -p "$prompt_text: " current_value
    fi
    printf -v "$var_name" '%s' "$current_value"
}

prompt_secret() {
    local var_name=$1
    local prompt_text=$2
    local current_value
    read -r -s -p "$prompt_text: " current_value
    echo
    printf -v "$var_name" '%s' "$current_value"
}

echo "==> Installing Steno Server as a user LaunchAgent."
echo

# Detect uv path.
UV_PATH_DEFAULT="$(command -v uv || true)"
prompt UV_PATH "Path to uv binary" "$UV_PATH_DEFAULT"
if [[ -z "$UV_PATH" || ! -x "$UV_PATH" ]]; then
    echo "Error: uv binary not found at '$UV_PATH'." >&2
    echo "Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Detect repo root (assumes this script lives at server/deploy/install-launchagent.sh).
REPO_PATH_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
prompt REPO_PATH "Path to the Steno repo (workspace root)" "$REPO_PATH_DEFAULT"
if [[ ! -d "$REPO_PATH/server" ]]; then
    echo "Error: $REPO_PATH/server not found. Did you clone the repo?" >&2
    exit 1
fi

# HF_TOKEN: required for Phase 2 diarization.
prompt_secret HF_TOKEN "HF_TOKEN (free token from huggingface.co; press Enter to skip — Phase 2 diarization will fail without it)"

# AUTH_PASSWORD: optional.
prompt_secret AUTH_PASSWORD "AUTH_PASSWORD (press Enter to leave the server open on the LAN)"

# ---------------------------------------------------------------------------
# Render and install.
# ---------------------------------------------------------------------------

echo
echo "==> Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"

echo "==> Creating LaunchAgents dir: $LAUNCH_AGENTS_DIR"
mkdir -p "$LAUNCH_AGENTS_DIR"

# Render placeholders.
TMP_PLIST="$(mktemp)"
trap 'rm -f "$TMP_PLIST"' EXIT
sed \
    -e "s|{{UV_PATH}}|$UV_PATH|g" \
    -e "s|{{REPO_PATH}}|$REPO_PATH|g" \
    -e "s|{{HF_TOKEN}}|$HF_TOKEN|g" \
    -e "s|{{AUTH_PASSWORD}}|$AUTH_PASSWORD|g" \
    -e "s|{{HOME}}|$HOME|g" \
    "$PLIST_TEMPLATE" > "$TMP_PLIST"

# Validate the plist before loading.
plutil -lint "$TMP_PLIST"

# If a previous version is loaded, unload it first (silently).
if launchctl list | grep -q '\bcom\.steno\.server\b'; then
    echo "==> Unloading existing com.steno.server agent"
    launchctl unload -w "$TARGET_PLIST" 2>/dev/null || true
fi

cp "$TMP_PLIST" "$TARGET_PLIST"
chmod 600 "$TARGET_PLIST"

echo "==> Loading agent"
launchctl load -w "$TARGET_PLIST"

# Verify.
sleep 1
if launchctl list | grep -q '\bcom\.steno\.server\b'; then
    echo
    echo "==> Steno Server is running."
    echo
    echo "  Health check:    curl http://127.0.0.1:8090/api/health"
    echo "  Tail logs:       tail -f $LOG_DIR/server.log"
    echo "  Stop agent:      launchctl unload -w '$TARGET_PLIST'"
    echo "  Restart agent:   launchctl unload -w '$TARGET_PLIST' && launchctl load -w '$TARGET_PLIST'"
    echo
    echo "  Plist installed at: $TARGET_PLIST"
else
    echo "Error: agent did not appear in launchctl list. Check $LOG_DIR/stderr.log." >&2
    exit 1
fi
