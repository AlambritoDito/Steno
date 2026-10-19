"""Behavior checks for desktop startup and native build inputs."""
import importlib.util
import json
from pathlib import Path
import socket
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loopback_binding_is_held_and_dynamic():
    entry = module_from_file("desktop_entry", ROOT / "main.py")
    first = entry.bind_loopback()
    second = entry.bind_loopback()
    try:
        assert first.getsockname()[0] == "127.0.0.1"
        assert first.getsockname()[1] != second.getsockname()[1]
        with socket.socket() as conflict, pytest.raises(OSError):
            conflict.bind(first.getsockname())
    finally:
        first.close()
        second.close()


def test_desktop_refuses_missing_token(monkeypatch):
    entry = module_from_file("desktop_entry", ROOT / "main.py")
    monkeypatch.setenv("STENO_ELECTRON", "1")
    monkeypatch.delenv("STENO_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="session token"):
        entry.main()


def test_runtime_extraction_contains_only_dlls_and_licenses(tmp_path):
    fetcher = module_from_file("runtime_fetcher", ROOT / "scripts" / "fetch_cuda_runtime.py")
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("cuda/bin/cublas64_12.dll", b"runtime")
        stream.writestr("../../escaped.dll", b"flattened")
        stream.writestr("..\\..\\windows.dll", b"flattened on Windows")
        stream.writestr("cuda/LICENSE.txt", b"license terms")
        stream.writestr("cuda/include/cublas.h", b"not shipped")
        stream.writestr("cuda/setup.exe", b"not executed")
    target = tmp_path / "bundle"
    fetcher.extract_runtime(archive, target, "cuda")
    assert (target / "cublas64_12.dll").read_bytes() == b"runtime"
    assert (target / "escaped.dll").read_bytes() == b"flattened"
    assert (target / "windows.dll").read_bytes() == b"flattened on Windows"
    assert not (tmp_path.parent / "escaped.dll").exists()
    assert sorted(path.name for path in target.iterdir()) == ["cublas64_12.dll", "escaped.dll", "licenses", "windows.dll"]
    assert (target / "licenses" / "cuda-LICENSE.txt").read_bytes() == b"license terms"


def test_build_requires_cuda_on_windows():
    builder = module_from_file("backend_builder", ROOT / "scripts" / "build_python.py")
    with pytest.raises(ValueError, match="CUDA"):
        builder.build_arguments("win32")
    with pytest.raises(ValueError, match="native"):
        builder.build_arguments("linux")
    args = builder.build_arguments("win32", Path("runtime"), [Path("msvcp140.dll")])
    binaries = [args[index + 1] for index, value in enumerate(args) if value == "--add-binary"]
    assert "msvcp140.dll:." in binaries
    assert "runtime:cuda" in args


def test_version_matches_all_manifests():
    import tomllib
    js = json.loads((ROOT / "package.json").read_text())
    python = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert js["version"] == python["project"]["version"]
    assert js["version"] == "1.0.0-beta.1"


def test_tracked_native_icons_are_readable():
    # Build extra is required only for artifact generation, not unit tests.
    ico = (ROOT / "packaging" / "resources" / "icon.ico").read_bytes()
    icns = (ROOT / "packaging" / "resources" / "icon.icns").read_bytes()
    assert ico[:4] == b"\x00\x00\x01\x00"
    assert icns[:4] == b"icns"
    assert int.from_bytes(icns[4:8], "big") == len(icns)


def test_portable_zip_has_marker_and_guide_without_mutating_installer_input(tmp_path):
    builder = module_from_file("portable_builder", ROOT / "scripts" / "build_portable.py")
    source = tmp_path / "win-unpacked"
    source.mkdir()
    (source / "Steno.exe").write_bytes(b"native executable fixture")
    (source / "resources").mkdir()
    (source / "resources" / "app.asar").write_bytes(b"application fixture")
    target = builder.build_portable(source, tmp_path / "portable.zip", "1.0.0-beta.1")
    assert not (source / "portable.json").exists()
    assert not (source / "Steno Data").exists()
    with zipfile.ZipFile(target) as archive:
        assert archive.read("Steno/Steno.exe") == b"native executable fixture"
        assert archive.read("Steno/resources/app.asar") == b"application fixture"
        assert json.loads(archive.read("Steno/portable.json"))["format"] == "steno-portable"
        assert archive.read("Steno/LEEME.txt") == (ROOT / "docs" / "PRUEBA-WINDOWS.md").read_bytes()
    (source / "Steno Data").mkdir()
    with pytest.raises(RuntimeError, match="used or modified"):
        builder.build_portable(source, tmp_path / "unsafe.zip", "1.0.0-beta.1")
    assert not (tmp_path / "unsafe.zip").exists()
