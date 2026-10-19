"""API-level integrity checks for saved source files and portable documents."""
import io
import zipfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from steno.jobs import JobManager
from steno.server import app
from steno.session import Session
from steno.storage import Store
from steno.transcriber import Transcriber


@pytest_asyncio.fixture
async def client(monkeypatch):
    app.state.jobs = JobManager()
    # Keep jobs durable and queued while inspecting the upload transaction.
    monkeypatch.setattr(app.state.jobs, "kick", lambda: None)
    app.state.transcriber = Transcriber()
    app.state.model_busy = False
    app.state.setup_complete = True
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        yield client


@pytest.mark.asyncio
async def test_upload_copies_source_and_survives_retry(client):
    session = Session("Source recovery")
    original = b"RIFF archived source bytes"
    response = await client.post(f"/api/sessions/{session.session_id}/transcribe-file",
                                 files={"file": ("lecture.wav", original, "audio/wav")})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = Store().get_job(job_id)
    path = session.audio_path(job["payload"]["take_id"])
    assert path.read_bytes() == original
    assert (await client.delete(f"/api/sessions/{session.session_id}")).status_code == 409
    assert (await client.post(f"/api/jobs/{job_id}/cancel")).json()["status"] == "cancelled"
    assert path.read_bytes() == original
    assert (await client.post(f"/api/jobs/{job_id}/retry", json={})).status_code == 202
    assert Store().get_job(job_id)["payload"]["take_id"] == job["payload"]["take_id"]
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_notes_images_and_relative_zip_survive_reload(client):
    session = Session("Portable document")
    note = "# Detailed note\nFirst line\nSecond line\n\n- acción\n- 漢字"
    for _ in range(3):
        response = await client.put(f"/api/sessions/{session.session_id}/note", json={"markdown": note})
        assert response.status_code == 200
    image = b"\x89PNG\r\n\x1a\nlocal image fixture"
    uploaded = await client.post(f"/api/sessions/{session.session_id}/image",
                                 files={"file": ("image.png", image, "image/png")})
    assert uploaded.status_code == 200
    reloaded = (await client.get(f"/api/sessions/{session.session_id}/state")).json()
    assert reloaded["notes"] == note
    assert len(reloaded["entries"]) == 1
    markdown = (await client.post(f"/api/sessions/{session.session_id}/export")).text
    assert markdown.count(note) == 1
    assert "data:image/png;base64," in markdown
    exported = await client.post(f"/api/sessions/{session.session_id}/export?format=zip")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as bundle:
        document = bundle.read("session.md").decode()
        assert note in document and "/api/" not in document
        assets = [p for p in bundle.namelist() if p.startswith("images/")]
        assert len(assets) == 1 and assets[0] in document
        assert bundle.read(assets[0]) == image


@pytest.mark.asyncio
async def test_assets_cannot_cross_session_boundaries(client):
    first, second = Session("First"), Session("Second")
    take = first.begin_take()
    first.audio_path(take["id"]).write_bytes(b"private recording")
    first.finish_take(take["id"])
    foreign_audio = await client.get(f"/api/sessions/{second.session_id}/audio?take_id={take['id']}")
    assert foreign_audio.status_code == 404
    svg = await client.post(f"/api/sessions/{first.session_id}/image",
                            files={"file": ("attack.svg", b'<svg onload="alert(1)"/>', "image/svg+xml")})
    assert svg.status_code == 422
    invalid = await client.post(f"/api/sessions/{first.session_id}/transcribe-file",
                                files={"file": ("script.py", b"print('wrong format')", "audio/wav")})
    assert invalid.status_code == 422
    assert len(first.takes()) == 1
