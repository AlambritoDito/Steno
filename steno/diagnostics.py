"""Bounded support exports: runtime metadata and allowlisted logs, never user data."""

import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

from steno.config import Config, _detect_hardware

LOG_NAMES = ("steno.log", "electron.log", "steno.log.1", "electron.log.old", "steno.log.2", "steno.log.3")
MAX_LOG_FILE_BYTES = 512 * 1024
MAX_LOG_TOTAL_BYTES = 2 * 1024 * 1024
RUNTIME_PACKAGES = ("fastapi", "numpy", "av", "sounddevice", "huggingface-hub", "psutil", "setuptools",
                    "mlx", "mlx-whisper", "mlx-metal", "faster-whisper", "ctranslate2", "onnxruntime")


def redact_secrets(text):
    """Remove current credentials and recognizable credentials from older logs."""
    for key in ("STENO_AUTH_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
                "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, "[redacted]")
    text = re.sub(r"(?i)(\b(?:Bearer|Basic)\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)([\"']?\b(?:steno_auth|STENO_AUTH_TOKEN|HF_TOKEN|access_token|refresh_token|"
                  r"token|api[_-]?key|password|secret|X-Amz-Signature|X-Amz-Credential|"
                  r"X-Amz-Security-Token|Signature|Policy|Key-Pair-Id)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;}]+",
                  r"\1[redacted]", text)
    return re.sub(r"\bhf_[A-Za-z0-9]{16,}\b", "[redacted]", text)


def nvidia_diagnostics():
    """Read GPU/driver inventory without starting inference or listing processes."""
    if platform.system() != "Windows":
        return {"available": False, "reason": "not_windows"}
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "tool_not_found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            return {"available": False, "reason": "query_failed", "return_code": result.returncode}
        gpus = []
        for row in csv.reader(result.stdout.splitlines()[:16]):
            if len(row) != 3:
                continue
            name, driver, memory = (part.strip() for part in row)
            gpus.append({"name": name[:200], "driver_version": driver[:50],
                         "vram_mb": int(memory) if memory.isdigit() else None})
        return {"available": bool(gpus), "gpus": gpus, "reason": None if gpus else "no_gpus_reported"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": "query_failed", "error_type": type(exc).__name__}


def collect_diagnostics(app_version, transcriber, portaudio_available):
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "schema_version": 1, "app_version": app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python_version": sys.version,
        "is_frozen": Config.is_frozen(), "hardware": _detect_hardware(),
        "portaudio_available": portaudio_available,
        "transcriber": transcriber.get_model_info(), "runtime_versions": versions,
        "nvidia": nvidia_diagnostics(), "data_dir": str(Config.data_dir()),
        "sessions_path": str(Config.sessions_path()), "logs_path": str(Config.logs_path()),
    }
    return json.loads(redact_secrets(json.dumps(result, ensure_ascii=False)))


def build_support_zip(diagnostics):
    """Include only bounded tails of fixed log names; symlinks are never followed."""
    report = dict(diagnostics)
    report["log_export"] = {"mode": "limited_tail", "max_bytes_per_file": MAX_LOG_FILE_BYTES,
                            "max_total_log_bytes": MAX_LOG_TOTAL_BYTES, "files": []}
    buffer = io.BytesIO()
    remaining = MAX_LOG_TOTAL_BYTES
    directory = Path(Config.logs_path())
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in LOG_NAMES:
            path = directory / name
            if remaining <= 0 or path.is_symlink() or not path.is_file():
                continue
            limit = min(MAX_LOG_FILE_BYTES, remaining)
            try:
                with path.open("rb") as stream:
                    size = stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, size - limit))
                    raw = stream.read(limit)
                text = redact_secrets(raw.decode("utf-8", errors="replace"))
                encoded = text.encode("utf-8")
                # Redaction and replacement characters may increase byte length.
                if len(encoded) > limit:
                    encoded = encoded[-limit:].decode("utf-8", errors="ignore").encode("utf-8")
                archive.writestr("logs/" + name, encoded)
                report["log_export"]["files"].append({"name": name, "source_bytes": size,
                                                     "exported_bytes": len(encoded), "truncated": size > limit})
                remaining -= len(encoded)
            except OSError as exc:
                report["log_export"]["files"].append({"name": name, "error_type": type(exc).__name__})
        archive.writestr("diagnostic.json", redact_secrets(json.dumps(report, ensure_ascii=False, indent=2)))
    return buffer.getvalue()
