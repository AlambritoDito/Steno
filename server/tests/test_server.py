"""REST endpoint tests for steno_server.server."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from steno_server import jobs as jobs_mod
from steno_server import worker
from steno_server.config import settings
from steno_server.jobs import JobStatus


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    worker.queue._states.clear()  # noqa: SLF001
    worker.queue._pending_order.clear()  # noqa: SLF001
    yield


@pytest.fixture
def client(isolated):
    from steno_server.server import app

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health & i18n
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_uptime_advances(client: TestClient):
    a = client.get("/api/health").json()["uptime_seconds"]
    b = client.get("/api/health").json()["uptime_seconds"]
    assert b >= a


def test_unknown_route_404s(client: TestClient):
    assert client.get("/api/missing").status_code == 404


def test_languages_list(client: TestClient):
    r = client.get("/api/languages")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == set(settings.supported_languages)


def test_i18n_es_returns_strings(client: TestClient):
    r = client.get("/api/i18n/es")
    assert r.status_code == 200
    body = r.json()
    assert "app_title" in body
    assert "upload_heading" in body


def test_i18n_unsupported_language_404s(client: TestClient):
    assert client.get("/api/i18n/xx").status_code == 404


def test_locale_files_have_same_keys():
    """Every key present in ES must also be present in EN, and vice versa."""
    from steno_server import i18n as i18n_mod

    es_keys = set(i18n_mod.load_locale("es").keys())
    en_keys = set(i18n_mod.load_locale("en").keys())
    only_in_es = es_keys - en_keys
    only_in_en = en_keys - es_keys
    assert not only_in_es, f"keys in ES but not EN: {only_in_es}"
    assert not only_in_en, f"keys in EN but not ES: {only_in_en}"


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------


def _silence_bytes() -> bytes:
    """Build a minimal valid 1-second 16 kHz mono WAV in-memory."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_create_job_returns_201_and_queue_position(client: TestClient):
    files = {"file": ("silence.wav", _silence_bytes(), "audio/wav")}
    data = {"language": "es", "enable_denoise": "true", "enable_diarization": "true"}
    r = client.post("/api/jobs", files=files, data=data)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    assert body["queue_position"] >= 0


def test_create_job_rejects_unsupported_language(client: TestClient):
    files = {"file": ("silence.wav", _silence_bytes(), "audio/wav")}
    data = {"language": "xx"}
    r = client.post("/api/jobs", files=files, data=data)
    assert r.status_code == 400


def test_create_job_rejects_short_audio(client: TestClient):
    """Validation runs on the saved file; <0.5 s audio gets rejected with 400."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    # 0.1 s — under MIN_DURATION_S
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16", format="WAV")
    files = {"file": ("tiny.wav", buf.getvalue(), "audio/wav")}
    r = client.post("/api/jobs", files=files, data={"language": "es"})
    assert r.status_code == 400
    assert "too short" in r.text.lower()


# ---------------------------------------------------------------------------
# Job listing & retrieval
# ---------------------------------------------------------------------------


async def _seed_job(filename: str = "x.wav", status: JobStatus = JobStatus.QUEUED):
    job = await jobs_mod.create_job(filename=filename, language="es")
    if status is not JobStatus.QUEUED:
        await jobs_mod.update_status(job.id, status)
    return job


def test_list_jobs(client: TestClient):
    job = asyncio.run(_seed_job())
    r = client.get("/api/jobs")
    assert r.status_code == 200
    body = r.json()
    assert any(j["id"] == job.id for j in body)


def test_get_job(client: TestClient):
    job = asyncio.run(_seed_job())
    r = client.get(f"/api/jobs/{job.id}")
    assert r.status_code == 200
    assert r.json()["id"] == job.id


def test_get_unknown_job_404(client: TestClient):
    r = client.get("/api/jobs/no-such-id")
    assert r.status_code == 404


def test_delete_job(client: TestClient):
    job = asyncio.run(_seed_job())
    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 204
    assert client.get(f"/api/jobs/{job.id}").status_code == 404


# ---------------------------------------------------------------------------
# Transcript download
# ---------------------------------------------------------------------------


def test_raw_transcript_404_when_not_ready(client: TestClient):
    job = asyncio.run(_seed_job())
    r = client.get(f"/api/jobs/{job.id}/transcript-raw.md")
    assert r.status_code == 404


def test_raw_transcript_serves_md_when_present(client: TestClient):
    from steno_server import storage

    job = asyncio.run(_seed_job(status=JobStatus.PHASE1_DONE))
    md_path = storage.get_job_dir(job.id, create=True) / "transcript-raw.md"
    md_path.write_text("---\ntype: raw\n---\n# hi\n")
    r = client.get(f"/api/jobs/{job.id}/transcript-raw.md")
    assert r.status_code == 200
    assert "type: raw" in r.text
    assert r.headers["content-type"].startswith("text/markdown")


def test_clean_transcript_404_when_not_ready(client: TestClient):
    job = asyncio.run(_seed_job(status=JobStatus.PHASE1_DONE))
    r = client.get(f"/api/jobs/{job.id}/transcript-clean.md")
    assert r.status_code == 404
