"""Validate NSIS install, installed CPU inference, and data-preserving uninstall on a clean CI user."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"


def require_ci_environment(environ, system_platform):
    if system_platform != "win32" or environ.get("GITHUB_ACTIONS") != "true" or environ.get("RUNNER_OS") != "Windows":
        raise RuntimeError("Installer smoke is restricted to an isolated Windows GitHub Actions runner")


def uninstaller_path(command):
    """Accept only electron-builder's quoted, per-user Steno uninstaller command."""
    match = re.fullmatch(r'"([^"\r\n]+)"\s+/currentuser(?:\s+/S)?', command, re.IGNORECASE)
    if not match or PureWindowsPath(match[1]).name.casefold() != "uninstall steno.exe":
        raise RuntimeError("Unexpected per-user Steno uninstall registration")
    return Path(match[1])


def registry_entries(hive):
    import winreg
    entries = {}
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        try:
            root = winreg.OpenKey(hive, UNINSTALL_KEY, 0, winreg.KEY_READ | view)
        except FileNotFoundError:
            continue
        with root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                key_name = winreg.EnumKey(root, index)
                with winreg.OpenKey(root, key_name) as key:
                    try:
                        name = winreg.QueryValueEx(key, "DisplayName")[0]
                    except FileNotFoundError:
                        continue
                    if not re.match(r"^Steno(?:$|\s+\d)", str(name)):
                        continue
                    values = {"key": key_name, "DisplayName": name}
                    for field in ("DisplayVersion", "QuietUninstallString", "UninstallString"):
                        try:
                            values[field] = winreg.QueryValueEx(key, field)[0]
                        except FileNotFoundError:
                            values[field] = ""
                    entries[key_name.casefold()] = values
    return list(entries.values())


def wait_until(condition, description, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {description}")


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(installer, audio, version):
    require_ci_environment(os.environ, sys.platform)
    import winreg
    from smoke_backend import run as smoke_backend

    if not installer.is_file() or not audio.is_file():
        raise RuntimeError("The NSIS installer and generated speech fixture must exist")
    if registry_entries(winreg.HKEY_CURRENT_USER) or registry_entries(winreg.HKEY_LOCAL_MACHINE):
        raise RuntimeError("Refusing to replace an existing Steno installation")
    local_data = Path(os.environ["LOCALAPPDATA"]) / "Steno"
    roaming_data = Path(os.environ["APPDATA"]) / "Steno"
    default_install = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Steno"
    if any(directory.exists() for directory in (local_data, roaming_data, default_install)):
        raise RuntimeError("Installer smoke requires a fresh CI user with no existing Steno data or files")
    sentinels = {}
    for directory in (local_data, roaming_data):
        directory.mkdir(parents=True)
        sentinel = directory / "installer-smoke-preserve.txt"
        sentinels[sentinel] = secrets.token_bytes(32)
        sentinel.write_bytes(sentinels[sentinel])

    # /S is assisted NSIS silent mode; /currentuser prevents an all-user install.
    subprocess.run([str(installer.resolve()), "/S", "/currentuser"], check=True,
                   timeout=600, creationflags=subprocess.CREATE_NO_WINDOW)
    wait_until(lambda: registry_entries(winreg.HKEY_CURRENT_USER), "HKCU installation registration")
    entries = registry_entries(winreg.HKEY_CURRENT_USER)
    if len(entries) != 1 or entries[0]["DisplayVersion"] != version:
        raise RuntimeError(f"Unexpected installed version or registration count: {entries}")
    if registry_entries(winreg.HKEY_LOCAL_MACHINE):
        raise RuntimeError("Installer unexpectedly created a machine-wide registration")
    entry = entries[0]
    uninstall = uninstaller_path(entry["QuietUninstallString"] or entry["UninstallString"])
    installed = uninstall.parent.resolve()
    if not installed.is_relative_to(Path(os.environ["USERPROFILE"]).resolve()):
        raise RuntimeError("Per-user installer wrote outside the CI user's profile")
    executable = installed / "Steno.exe"
    backend = installed / "resources" / "steno-server" / "steno-server.exe"
    wait_until(lambda: executable.is_file() and backend.is_file() and uninstall.is_file(), "installed application files")
    if (installed / "portable.json").exists():
        raise RuntimeError("NSIS installation must not contain the portable marker")
    print("PASS: native NSIS silent per-user installation and HKCU registration", flush=True)
    persisted_files = {}
    try:
        # Keep the real session/model data here so uninstall preservation is tested.
        smoke_backend(backend, audio, "cpu", data_directory=local_data)
        database = local_data / "sessions" / "steno.sqlite3"
        if not database.is_file():
            raise RuntimeError("Installed backend did not persist its session database")
        persisted_files = {file: digest(file) for file in local_data.rglob("*")
                           if file.is_file() and file.suffix in {".sqlite3", ".wav"}}
        if not any(file.suffix == ".wav" for file in persisted_files):
            raise RuntimeError("Installed backend did not preserve the speech source")
    finally:
        # Pass fixed arguments, never execute a registry command through a shell.
        subprocess.run([str(uninstall), "/currentuser", "/S"], check=True,
                       timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
        wait_until(lambda: not executable.exists() and not backend.exists()
                   and not registry_entries(winreg.HKEY_CURRENT_USER), "silent uninstall completion")
        for sentinel, expected in sentinels.items():
            if not sentinel.is_file() or sentinel.read_bytes() != expected:
                raise RuntimeError("Uninstall removed or modified persisted application data")
        for file, expected in persisted_files.items():
            if not file.is_file() or digest(file) != expected:
                raise RuntimeError("Uninstall removed or changed the saved session database or source audio")
    print("PASS: silent uninstall removed the app and preserved app data, notes and source audio", flush=True)
    report = {"app_version": version, "platform": platform.platform(), "installer": installer.name,
              "install_mode": "current_user", "registry": "HKCU", "cpu_inference": "passed",
              "cached_restart": "passed", "data_preserved_after_uninstall": True,
              "interactive_installer_ui_validated": False, "nvidia_cuda_validated": False}
    (ROOT / "release" / "installer-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    version = json.loads((ROOT / "package.json").read_text())["version"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", type=Path, default=ROOT / "release" / f"Steno-{version}-win-x64.exe")
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    run(args.installer, args.audio, version)


if __name__ == "__main__":
    main()
