"""Platform paths, model cache integrity, and settings recovery."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from steno.config import Config, _detect_hardware, default_backend, resolve_model_repo


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("STENO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "legacy_data_dirs", classmethod(lambda cls: []))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", "mlx"), ("Windows", "AMD64", "faster-whisper"),
    ("Darwin", "x86_64", "unsupported"), ("Linux", "aarch64", "unsupported"),
])
def test_platform_backend(system, machine, expected, monkeypatch):
    monkeypatch.setattr("steno.config.platform.system", lambda: system)
    monkeypatch.setattr("steno.config.platform.machine", lambda: machine)
    assert default_backend() == expected


def test_cross_platform_saved_model_translation():
    assert resolve_model_repo("mlx-community/whisper-small", "faster-whisper") == "Systran/faster-whisper-small"
    assert resolve_model_repo("Systran/faster-whisper-small", "mlx") == "mlx-community/whisper-small"


def test_native_windows_data_location(tmp_path, monkeypatch):
    monkeypatch.delenv("STENO_DATA_DIR")
    monkeypatch.setattr("steno.config.platform.system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert Config.data_dir() == tmp_path / "Steno"


def test_atomic_settings_failure_preserves_previous(tmp_path, monkeypatch):
    Config.save_settings({"model_key": "tiny"})
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("steno.config.os.replace", fail)
    with pytest.raises(OSError):
        Config.save_settings({"model_key": "large-v3"})
    assert Config.load_settings() == {"model_key": "tiny"}
    assert list(tmp_path.iterdir()) == [Config.settings_path()]


@pytest.mark.parametrize("content", ["{bad-json", "[]", "null"])
def test_corrupt_settings_preserved_and_recoverable(content):
    path = Config.settings_path()
    path.write_text(content)
    assert Config.load_settings() == {}
    backups = list(path.parent.glob("*.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == content
    Config.save_settings({"setup_complete": False})
    assert Config.load_settings() == {"setup_complete": False}


def test_hardware_unknown_ram_is_none(monkeypatch):
    def fail():
        raise OSError("unavailable")
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=fail))
    monkeypatch.setattr("steno.config.platform.system", lambda: "Windows")
    assert _detect_hardware()["ram_gb"] is None


def test_incomplete_cache_is_not_downloaded():
    cache = Config.models_dir() / "models--mlx-community--whisper-tiny"
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("a" * 40)
    (snapshot / "config.json").write_text("{}")
    assert not Config.model_is_downloaded("mlx-community/whisper-tiny")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    assert Config.model_snapshot_path("mlx-community/whisper-tiny") == snapshot
    (snapshot / "weights.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1")
    assert not Config.model_is_downloaded("mlx-community/whisper-tiny")


def test_ref_cannot_escape_cache(tmp_path):
    cache = Config.models_dir() / "models--mlx-community--whisper-tiny"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("../../outside")
    assert Config.model_snapshot_path("mlx-community/whisper-tiny") is None


def test_repository_and_session_paths_reject_traversal():
    for repo in ("../../x", "a/b/c", "a\\b", "/tmp/model"):
        with pytest.raises(ValueError):
            Config.model_cache_path(repo)
    with pytest.raises(ValueError):
        Config.images_path("..\\outside")


def test_cache_size_does_not_double_count_hf_symlinks(tmp_path):
    cache = Config.models_dir() / "models--test--repo"
    cache.mkdir()
    blob = cache / "blob"
    blob.write_bytes(b"x" * (1024 * 1024))
    (cache / "snapshot-weight").symlink_to(blob)
    assert Config.model_cache_size_mb("test/repo") == 1.0
