"""Guard the installer verifier against changing a normal workstation or running arbitrary registry commands."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("installer_smoke", Path(__file__).resolve().parents[1] / "scripts" / "smoke_windows_installer.py")
installer_smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer_smoke)


def test_installer_verifier_requires_isolated_windows_ci():
    installer_smoke.require_ci_environment({"GITHUB_ACTIONS": "true", "RUNNER_OS": "Windows"}, "win32")
    with pytest.raises(RuntimeError, match="isolated Windows"):
        installer_smoke.require_ci_environment({}, "win32")
    with pytest.raises(RuntimeError, match="isolated Windows"):
        installer_smoke.require_ci_environment({"GITHUB_ACTIONS": "true", "RUNNER_OS": "Windows"}, "darwin")


def test_uninstaller_registration_is_parsed_without_a_shell():
    command = r'"C:\Users\Runner\AppData\Local\Programs\Steno\Uninstall Steno.exe" /currentuser /S'
    assert str(installer_smoke.uninstaller_path(command)) == r"C:\Users\Runner\AppData\Local\Programs\Steno\Uninstall Steno.exe"


@pytest.mark.parametrize("command", [
    r'"C:\Steno\Uninstall Steno.exe" /allusers /S',
    r'cmd.exe /c "C:\Steno\Uninstall Steno.exe" /currentuser /S',
    r'"C:\Other\payload.exe" /currentuser /S',
    r'"C:\Steno\Uninstall Steno.exe" /currentuser /S & echo unsafe',
])
def test_unexpected_uninstall_commands_are_rejected(command):
    with pytest.raises(RuntimeError, match="Unexpected per-user"):
        installer_smoke.uninstaller_path(command)
