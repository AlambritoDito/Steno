"""Shared fixtures for steno-server tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect storage_dir + log_dir to a temp directory for the test.

    Avoids polluting /tmp/steno-server and the user's Library/Logs folder.
    """
    from steno_server.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "log_dir", tmp_path / "logs")
    return tmp_path


@pytest.fixture
def client(isolated_storage: Path) -> TestClient:
    """FastAPI TestClient with isolated storage already wired up."""
    from steno_server.server import app

    with TestClient(app) as test_client:
        yield test_client
