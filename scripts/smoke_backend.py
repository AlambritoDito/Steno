"""Exercise a native bundled backend through authenticated HTTP, without developer imports.

Default: startup, origin/token protection, static UI and shutdown. Pass --audio and
--expect-device to download the tiny model, run actual inference, then verify
persisted data and cached inference after restart with HF_HUB_OFFLINE=1.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def request(url, token, method="GET", body=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = content_type
    message = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(message, timeout=600)
    except urllib.error.HTTPError as error:
        error.msg = f"{error.reason}: {error.read().decode('utf-8', errors='replace')[:2000]}"
        raise
    with response:
        payload = response.read()
        return json.loads(payload) if "application/json" in response.headers.get("Content-Type", "") else payload


@contextmanager
def launch(binary, data_directory, *, offline=False):
    token = secrets.token_hex(32)
    env = {**os.environ, "STENO_ELECTRON": "1", "STENO_PORT": "0", "STENO_AUTH_TOKEN": token,
           "STENO_DATA_DIR": str(data_directory), "HF_HUB_CACHE": str(Path(data_directory) / "models"),
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_OFFLINE": "1" if offline else "0"}
    process = subprocess.Popen([str(binary.resolve())], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = queue.Queue()
    errors = []

    def read_stdout():
        for line in process.stdout:
            lines.put(line)

    def read_stderr():
        for line in process.stderr:
            errors.append(line)
            del errors[:-50]

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    try:
        deadline = time.monotonic() + 120
        url = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Backend exited: " + "".join(errors))
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                continue
            if line.startswith("STENO_READY "):
                port = json.loads(line[12:])["port"]
                url = f"http://127.0.0.1:{port}"
                break
        if not url:
            raise RuntimeError("No backend readiness handshake")
        while time.monotonic() < deadline:
            try:
                if request(url + "/api/health", token).get("status") == "ok":
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise RuntimeError("Backend health check timed out")
        try:
            request(url + "/api/health", "incorrect-token")
        except urllib.error.HTTPError as error:
            assert error.code in (401, 403), error.code
        else:
            raise AssertionError("Backend accepted an incorrect token")
        assert b"Steno" in request(url + "/", token)
        print("PASS: native startup, authenticated readiness, unauthorized rejection, UI assets", flush=True)
        yield url, token
    finally:
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise RuntimeError("Backend did not shut down when its parent pipe closed")
        process.stdout.close()
        process.stderr.close()
    if process.returncode != 0:
        raise RuntimeError(f"Backend shutdown exit {process.returncode}: {''.join(errors)}")
    print("PASS: clean parent-owned shutdown", flush=True)


def wait_inference(url, token, job, session_id, expected_device):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        state = request(url + f"/api/jobs/{job['job_id']}", token)
        if state["status"] == "completed":
            assert state["result"]["count"] > 0, "Audio fixture must contain recognizable speech"
            session = request(url + f"/api/sessions/{session_id}/state", token)
            assert any(entry["kind"] == "transcript" and entry["text"].strip()
                       for entry in session["entries"]), "Recognized speech must be persisted in the session"
            capabilities = state["result"].get("capabilities", {})
            if expected_device:
                assert capabilities.get("device") == expected_device, capabilities
                assert capabilities.get("validated"), capabilities
            print("PASS: packaged model load and speech inference", capabilities, flush=True)
            return session
        if state["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Inference did not complete: {state}")
        time.sleep(0.25)
    raise RuntimeError("Inference timed out")


def run(binary, audio=None, expected_device=None):
    if not binary.is_file():
        raise RuntimeError(f"Build the native backend first: {binary}")
    device = expected_device if expected_device in {"cuda", "cpu"} else "auto"
    with tempfile.TemporaryDirectory(prefix="steno-smoke-") as temporary:
        with launch(binary, temporary) as (url, token):
            if not audio:
                return
            request(url + "/api/setup/select-model", token, "POST", json.dumps({"model_key": "tiny"}).encode())
            created = request(url + "/api/sessions/new", token, "POST", json.dumps({"name": "Installer smoke test"}).encode())
            session_id = created["session_id"]
            boundary = secrets.token_hex(16)
            body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"device\"\r\n\r\n{device}\r\n"
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"smoke{audio.suffix}\"\r\n"
                    "Content-Type: application/octet-stream\r\n\r\n").encode() + audio.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            job = request(url + f"/api/sessions/{session_id}/transcribe-file", token,
                          "POST", body, f"multipart/form-data; boundary={boundary}")
            original = wait_inference(url, token, job, session_id, expected_device)
            note = "Persistent smoke note — survives application restart."
            request(url + f"/api/sessions/{session_id}/note", token, "PUT", json.dumps({"markdown": note}).encode())
        with launch(binary, temporary, offline=True) as (url, token):
            restored = request(url + f"/api/sessions/{session_id}/state", token)
            assert restored["notes"] == note, "Saved note must survive restart"
            assert restored["entries"] == original["entries"], "Saved transcript must survive restart"
            assert restored["takes"] and restored["has_audio"], "Source audio must survive restart"
            assert request(url + f"/api/sessions/{session_id}/audio", token), "Source audio must remain readable"
            job = request(url + f"/api/sessions/{session_id}/retranscribe", token, "POST",
                          json.dumps({"device": device}).encode())
            completed = wait_inference(url, token, job, session_id, expected_device)
            assert completed["notes"] == note, "Retranscription must preserve notes"
            print("PASS: persisted notes, transcript and source; cached restart inference with HF_HUB_OFFLINE=1", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "dist" / "steno-server" / ("steno-server.exe" if sys.platform == "win32" else "steno-server"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--expect-device", choices=["cuda", "cpu", "metal"])
    args = parser.parse_args()
    if args.expect_device and not args.audio:
        parser.error("--expect-device requires a real speech fixture via --audio")
    run(args.binary, args.audio, args.expect_device)


if __name__ == "__main__":
    main()
