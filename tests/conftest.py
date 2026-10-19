"""Never touch user recordings/settings in unit tests."""
import pytest


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("STENO_DATA_DIR", str(tmp_path / "steno"))
    monkeypatch.delenv("STENO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("STENO_PORT", raising=False)
    from steno.config import Config
    monkeypatch.setattr(Config, "legacy_data_dirs", classmethod(lambda cls: []))
