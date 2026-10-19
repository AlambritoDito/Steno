"""Exercise native smoke rejection and failure evidence without a model download."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("native_smoke", Path(__file__).resolve().parents[1] / "scripts" / "smoke_backend.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def windows_report(**capabilities):
    return {"is_frozen": True,
            "runtime_versions": {name: "1.0" for name in ("fastapi", "numpy", "av", "sounddevice",
                                 "huggingface-hub", "psutil", "faster-whisper", "ctranslate2", "onnxruntime", "setuptools")},
            "transcriber": {"backend": "faster-whisper", "device": "cpu", "available": True,
                            "cpu_available": True, **capabilities}}


def support_zip(report, log="Native startup log"):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("diagnostic.json", json.dumps(report))
        archive.writestr("logs/steno.log", log)
    return data.getvalue()


def test_frozen_smoke_rejects_missing_engine_even_when_metadata_is_present():
    with pytest.raises(AssertionError, match="runtime is unavailable"):
        smoke.validate_runtime(windows_report(available=False), "win32")
    with pytest.raises(AssertionError, match="CPU int8"):
        smoke.validate_runtime(windows_report(cpu_available=False), "win32")
    smoke.validate_runtime(windows_report(), "win32")


def test_failure_retains_only_support_export_and_redacted_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STENO_SMOKE_DIAGNOSTICS_DIR", str(tmp_path))
    token = "launch-secret-value"
    payload = support_zip(windows_report(available=False, diagnostic="No module named pkg_resources"))
    smoke.retain_failure(None, token, payload, RuntimeError(f"Authorization: Bearer {token}"),
                         [f"token={token} hf_exampleCredential\n"])
    files = sorted(tmp_path.iterdir())
    assert {file.suffix for file in files} == {".zip", ".log"}
    archive = next(file for file in files if file.suffix == ".zip")
    assert archive.read_bytes() == payload
    log = next(file for file in files if file.suffix == ".log").read_text()
    assert "pkg_resources" in log
    assert token not in log and "hf_exampleCredential" not in log
    assert token not in capsys.readouterr().err


def test_failure_never_retains_export_containing_launch_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("STENO_SMOKE_DIAGNOSTICS_DIR", str(tmp_path))
    token = "launch-secret-value"
    payload = support_zip(windows_report(), log=token)
    with pytest.raises(AssertionError, match="redact"):
        smoke.support_report(payload, token)
    smoke.retain_failure(None, token, payload, RuntimeError("Unsafe support export"), [])
    assert not list(tmp_path.glob("*.zip"))
    assert len(list(tmp_path.glob("*.log"))) == 1
