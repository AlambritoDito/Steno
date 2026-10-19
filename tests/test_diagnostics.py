import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from steno.config import Config
from steno import diagnostics


def open_export(report=None):
    return zipfile.ZipFile(io.BytesIO(diagnostics.build_support_zip(report or {"schema_version": 1})))


def test_support_zip_allowlists_logs_and_redacts_current_and_old_credentials(monkeypatch):
    logs = Config.logs_path()
    logs.mkdir(parents=True, exist_ok=True)
    token = "current-launch-credential"
    monkeypatch.setenv("STENO_AUTH_TOKEN", token)
    (logs / "steno.log").write_text(
        f"startup {token}\nAuthorization: Bearer previous-launch-credential\n"
        "Cookie: steno_auth=old-cookie-credential; other=okay\n"
        '"access_token": "old-access-credential"\nHF_TOKEN=old-hf-credential\n', encoding="utf-8")
    (logs / "electron.log.old").write_text("old desktop event", encoding="utf-8")
    for name in ("steno.sqlite3", "recording.wav", ".steno_settings.json", "notes.md", "model.bin", "unknown.log"):
        (logs / name).write_text("private user content", encoding="utf-8")
    (logs / "nested").mkdir()
    (logs / "nested" / "steno.log").write_text("private nested content", encoding="utf-8")
    with open_export() as archive:
        assert set(archive.namelist()) == {"diagnostic.json", "logs/steno.log", "logs/electron.log.old"}
        content = b"\n".join(archive.read(name) for name in archive.namelist()).decode("utf-8")
        for forbidden in (token, "previous-launch-credential", "old-cookie-credential", "old-access-credential",
                          "old-hf-credential", "private user content", "private nested content"):
            assert forbidden not in content
        assert "[redacted]" in content
        assert json.loads(archive.read("diagnostic.json"))["log_export"]["mode"] == "limited_tail"


def test_support_log_tails_obey_per_file_and_total_limits(monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_LOG_FILE_BYTES", 64)
    monkeypatch.setattr(diagnostics, "MAX_LOG_TOTAL_BYTES", 100)
    logs = Config.logs_path()
    logs.mkdir(parents=True, exist_ok=True)
    for name in diagnostics.LOG_NAMES:
        (logs / name).write_bytes(("á" * 150).encode("utf-8"))
    with open_export() as archive:
        payloads = [archive.read(name) for name in archive.namelist() if name.startswith("logs/")]
        assert sum(map(len, payloads)) <= 100
        assert all(len(content) <= 64 for content in payloads)
        for content in payloads:
            content.decode("utf-8")
        files = json.loads(archive.read("diagnostic.json"))["log_export"]["files"]
        assert files and all(item["truncated"] for item in files)


def test_support_zip_never_follows_allowlisted_log_symlink(tmp_path):
    logs = Config.logs_path()
    logs.mkdir(parents=True, exist_ok=True)
    private = tmp_path / "private.txt"
    private.write_text("private file", encoding="utf-8")
    try:
        (logs / "steno.log").symlink_to(private)
    except OSError:
        pytest.skip("This runner does not permit symlinks")
    with open_export() as archive:
        assert archive.namelist() == ["diagnostic.json"]


def test_nvidia_inventory_is_bounded_and_contains_only_gpu_driver_memory(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(diagnostics.shutil, "which", lambda command: "nvidia-smi.exe")
    def run(command, **kwargs):
        assert command == ["nvidia-smi.exe", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(returncode=0, stdout="NVIDIA Test GPU, 555.99, 8192\n")
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    assert diagnostics.nvidia_diagnostics() == {
        "available": True, "gpus": [{"name": "NVIDIA Test GPU", "driver_version": "555.99", "vram_mb": 8192}],
        "reason": None,
    }


def test_nvidia_missing_tool_and_timeout_are_nonfatal(monkeypatch):
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")
    monkeypatch.setattr(diagnostics.shutil, "which", lambda command: None)
    assert diagnostics.nvidia_diagnostics()["reason"] == "tool_not_found"
    monkeypatch.setattr(diagnostics.shutil, "which", lambda command: "nvidia-smi.exe")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)
    monkeypatch.setattr(diagnostics.subprocess, "run", timeout)
    assert diagnostics.nvidia_diagnostics()["error_type"] == "TimeoutExpired"


def test_runtime_report_uses_metadata_and_does_not_start_model(monkeypatch):
    monkeypatch.setattr(diagnostics, "_detect_hardware", lambda: {"ram_gb": 16, "chip": "test"})
    monkeypatch.setattr(diagnostics, "nvidia_diagnostics", lambda: {"available": False})
    monkeypatch.setattr(diagnostics.importlib.metadata, "version", lambda package: "1.2.3")
    transcriber = SimpleNamespace(get_model_info=lambda: {"backend": "fake", "loaded": False, "device": "cpu"})
    report = diagnostics.collect_diagnostics("beta", transcriber, True)
    assert report["schema_version"] == 1
    assert report["hardware"]["ram_gb"] == 16
    assert report["runtime_versions"]["ctranslate2"] == "1.2.3"
    assert report["logs_path"] == str(Config.logs_path())
    assert report["transcriber"]["loaded"] is False
    assert "settings" not in report and "sessions" not in report
