"""Tests for steno_server.storage."""

from __future__ import annotations

import asyncio
import os
import time
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from steno_server import storage
from steno_server.config import settings


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every storage test gets its own storage_dir so /tmp/steno-server stays clean."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")


def test_get_job_dir_creates_when_asked(tmp_path: Path):
    job_id = "abc"
    path = storage.get_job_dir(job_id, create=True)
    assert path.exists()
    assert path.name == job_id
    assert path.parent == settings.storage_dir


def test_get_job_dir_does_not_create_by_default():
    path = storage.get_job_dir("ghost")
    assert not path.exists()


def test_get_paths_compose_correctly():
    job_id = "j1"
    assert storage.get_source_path(job_id, ".mp3").name == "source.mp3"
    assert storage.get_source_path(job_id, "wav").name == "source.wav"  # accepts no-dot
    assert storage.get_normalized_path(job_id).name == "normalized.wav"
    assert storage.get_denoised_path(job_id).name == "denoised.wav"
    assert storage.get_transcript_path(job_id, "raw").name == "transcript-raw.md"
    assert storage.get_transcript_path(job_id, "clean").name == "transcript-clean.md"


def test_save_upload_writes_bytes_to_job_dir():
    job_id = "save-test"
    upload = UploadFile(filename="meeting.m4a", file=BytesIO(b"x" * 4096))
    saved_path = asyncio.run(storage.save_upload(job_id, upload))
    assert saved_path.name == "source.m4a"
    assert saved_path.read_bytes() == b"x" * 4096


def test_save_upload_rejects_filename_less_uploads():
    upload = UploadFile(filename=None, file=BytesIO(b"data"))
    with pytest.raises(ValueError):
        asyncio.run(storage.save_upload("nope", upload))


def test_delete_job_files_removes_directory():
    job_id = "to-delete"
    job_dir = storage.get_job_dir(job_id, create=True)
    (job_dir / "marker.txt").write_text("hi")
    assert storage.delete_job_files(job_id) is True
    assert not job_dir.exists()


def test_delete_job_files_returns_false_when_absent():
    assert storage.delete_job_files("never-existed") is False


def test_list_job_dirs_returns_oldest_first():
    a = storage.get_job_dir("a", create=True)
    time.sleep(0.01)
    b = storage.get_job_dir("b", create=True)
    time.sleep(0.01)
    c = storage.get_job_dir("c", create=True)
    # Touch a's mtime forward so order changes; ensures we're sorting on mtime.
    os.utime(a, (time.time(), time.time()))
    listed = storage.list_job_dirs()
    assert [p.name for p in listed] == ["b", "c", "a"]


def test_cleanup_old_jobs_deletes_past_cutoff():
    # Create three job dirs and back-date two of them past the cutoff.
    job_dirs = [storage.get_job_dir(f"j{i}", create=True) for i in range(3)]
    very_old = time.time() - 48 * 3600
    os.utime(job_dirs[0], (very_old, very_old))
    os.utime(job_dirs[1], (very_old, very_old))

    deleted = asyncio.run(storage.cleanup_old_jobs(retention_hours=24))
    assert deleted == 2
    remaining = {p.name for p in storage.list_job_dirs()}
    assert remaining == {"j2"}


def test_cleanup_does_not_delete_fresh_jobs():
    storage.get_job_dir("fresh", create=True)
    deleted = asyncio.run(storage.cleanup_old_jobs(retention_hours=24))
    assert deleted == 0
    assert storage.get_job_dir("fresh").exists()
