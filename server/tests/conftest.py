"""Shared fixtures for steno-server tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Whether the current environment has ffmpeg installed. mlx-whisper shells
# out to ffmpeg for ALL audio decoding (even WAV), so any test that exercises
# real Whisper inference must skip when this is False.
HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG,
    reason="ffmpeg not installed; mlx-whisper cannot decode audio without it. "
    "Install with `brew install ffmpeg`.",
)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the audio fixtures directory.

    The fixtures themselves are gitignored .wav files; regenerate them with
    ``uv run --directory server python tests/generate_fixtures.py``.
    """
    if not FIXTURES_DIR.exists():
        pytest.skip(
            f"Fixtures dir missing: {FIXTURES_DIR}. "
            "Run `uv run --directory server python tests/generate_fixtures.py`."
        )
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def silence_wav(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_silence.wav"
    if not path.exists():
        pytest.skip(f"Missing fixture: {path}. Regenerate with generate_fixtures.py.")
    return path


@pytest.fixture(scope="session")
def clean_es_wav(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_clean_es.wav"
    if not path.exists():
        pytest.skip(f"Missing fixture: {path}. Regenerate with generate_fixtures.py.")
    return path


@pytest.fixture(scope="session")
def noisy_es_wav(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "sample_noisy_es.wav"
    if not path.exists():
        pytest.skip(f"Missing fixture: {path}. Regenerate with generate_fixtures.py.")
    return path


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
