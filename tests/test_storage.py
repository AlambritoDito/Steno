"""Data durability, migration and export regression tests."""

import io
import json
import sqlite3
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from steno.config import Config
from steno.session import Session, import_legacy_sessions, list_sessions, recover_interrupted_takes, safe_path
from steno.storage import Store


@pytest.fixture(autouse=True)
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "sessions_path", classmethod(lambda cls: tmp_path))
    return tmp_path


def write_wave(path, frames=1600):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * frames)


def test_identical_names_do_not_collide():
    sessions = [Session("same") for _ in range(30)]
    assert len({s.session_id for s in sessions}) == 30


def test_roundtrip_preserves_arbitrary_markdown():
    session = Session("Rich notes")
    text = "# Heading\n\n![diagram](example.png)\n\nAfter image\n\n### Notes\n\n**[00:01:02]** not a transcript\n```\n<div>x</div>\n```"
    session.add_note(text)
    session.add_note("First\nSecond\n\nThird", elapsed_seconds=4)
    session.add_transcript("First line\nSecond line")
    loaded = Session.load(session.session_id)
    assert loaded.notes == text
    assert loaded.to_markdown() == session.to_markdown()
    assert loaded._entries[0]["text"] == "First\nSecond\n\nThird"
    assert loaded._entries[1]["text"] == "First line\nSecond line"


def test_notes_upsert_and_clear():
    session = Session("Notes")
    session.add_note("first")
    session.add_note("first")
    assert session.to_markdown().count("first") == 1
    session.add_note("second")
    assert "first" not in session.to_markdown()
    session.add_note("")
    assert Session.load(session.session_id).notes == ""


def test_legacy_import_preserves_exact_document_and_original(storage_root):
    text = "# Old session\r\n\r\n## Unexpected heading\r\n![img](data:image/png;base64,AA==)\r\nAfter image\r\n"
    path = storage_root / "20260101_120000_old.md"
    original = text.encode()
    path.write_bytes(original)
    assert import_legacy_sessions() == 1
    assert import_legacy_sessions() == 0
    session = Session.load(path.stem)
    assert session.to_markdown() == text
    session.add_note("New notes")
    snapshot = session.save()
    assert snapshot != path
    assert path.read_bytes() == original
    assert Session.load(path.stem).to_dict()["legacy_markdown"] == text
    session.delete()
    assert path.read_bytes() == original
    assert list_sessions() == []


def test_invalid_legacy_utf8_left_untouched(storage_root):
    path = storage_root / "broken.md"
    path.write_bytes(b"\xff\xfe\x80")
    assert list_sessions() == []
    assert path.read_bytes() == b"\xff\xfe\x80"


def test_legacy_unusual_filename_imported(storage_root):
    path = storage_root / "clase física (1).md"
    path.write_text("# Física\ncontenido", encoding="utf-8")
    sessions = list_sessions()
    assert sessions[0]["id"].startswith("legacy-")
    assert Session.load(sessions[0]["id"]).to_markdown() == "# Física\ncontenido"
    assert len(list_sessions()) == 1


def test_new_snapshots_are_not_reimported():
    session = Session("new")
    session.save()
    assert import_legacy_sessions() == 0
    assert len(list_sessions()) == 1
    session.delete()
    assert list_sessions() == []


@pytest.mark.parametrize("identifier", ["../outside", "..", ".", "/etc/passwd", "a/b", "a\\b", "x%2fy", ""])
def test_invalid_session_ids_rejected(identifier):
    with pytest.raises(ValueError):
        Session.load(identifier)


def test_symlink_asset_escape_rejected(storage_root, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (storage_root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(storage_root, "escape/private.txt")


def test_portable_markdown_embeds_images():
    session = Session("Images")
    session.add_image(b"PNG-BYTES", "image/png", "diagram")
    text = session.to_markdown(portable=True)
    assert "data:image/png;base64,UE5HLUJZVEVT" in text
    assert "/api/sessions/" not in text


def test_zip_contains_assets_and_audio_with_relative_links():
    session = Session("Bundle")
    session.add_image(b"image", "image/png", "diagram")
    take = session.begin_take()
    write_wave(take["path"])
    session.finish_take(take["id"])
    with zipfile.ZipFile(io.BytesIO(session.export_zip())) as archive:
        text = archive.read("session.md").decode()
        assert "/api/sessions/" not in text
        image_name = next(name for name in archive.namelist() if name.startswith("images/"))
        assert image_name in text
        assert archive.read(image_name) == b"image"
        assert f"audio/{take['id']}.wav" in archive.namelist()
        assert all(not name.startswith("/") and ".." not in name for name in archive.namelist())
        metadata = json.loads(archive.read("session.json"))
        assert metadata["takes"][0]["path"] == f"audio/{take['id']}.wav"


def test_legacy_local_images_are_portable_without_modifying_original(storage_root):
    session_id = "20260101_120000_images"
    image = storage_root / "images" / session_id / "example.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"legacy image")
    original = f"# Legacy\n![pic](/api/sessions/{session_id}/images/example.png)\nAFTER"
    (storage_root / f"{session_id}.md").write_text(original)
    session = Session.load(session_id)
    assert "data:image/png;base64," in session.to_markdown(portable=True)
    assert session.to_dict()["legacy_markdown"] == original


def test_take_paths_are_unique_and_not_precreated():
    session = Session("Record")
    first = session.begin_take()
    second = session.begin_take(suffix=".mp3", source="upload")
    assert first["path"] != second["path"]
    assert Path(second["path"]).suffix == ".mp3"
    assert not Path(first["path"]).exists()
    assert not Path(second["path"]).exists()


def test_recovery_marks_recording_interrupted_and_preserves_audio():
    session = Session("Interrupted")
    take = session.begin_take()
    write_wave(take["path"], 3200)
    session.update_take(take["id"], processed_frames=1600)
    original = Path(take["path"]).read_bytes()
    assert recover_interrupted_takes() == 1
    assert recover_interrupted_takes() == 0
    loaded = Session.load(session.session_id).takes()[0]
    assert loaded["status"] == "interrupted"
    assert loaded["processed_frames"] == 1600
    assert loaded["frames"] == 3200
    assert loaded["duration_seconds"] == pytest.approx(0.2)
    assert Path(take["path"]).read_bytes() == original


def test_legacy_audio_remains_on_delete(storage_root):
    session_id = "20260101_120000_audio"
    (storage_root / f"{session_id}.md").write_text("# old")
    audio = storage_root / f"{session_id}.wav"
    write_wave(audio)
    session = Session.load(session_id)
    assert session.has_audio()
    assert session.audio_path() == audio
    session.delete()
    assert audio.exists()


def test_retry_segment_is_idempotent_and_takes_are_isolated():
    session = Session("Idempotent")
    take = session.begin_take()
    other = session.begin_take()
    session.add_transcript("one", take_id=take["id"], start=0, end=1)
    session.add_transcript("one", take_id=take["id"], start=0, end=1)
    session.add_transcript("other", take_id=other["id"], start=0, end=1)
    assert len(session.to_dict()["entries"]) == 2
    session.replace_transcripts([{"text": "replacement", "start": 0, "end": 2}], take["id"])
    assert [e["text"] for e in session.to_dict()["entries"]] == ["other", "replacement"]


def test_replace_transcripts_rolls_back_on_invalid_segment():
    session = Session("Atomic")
    session.add_transcript("keep me")
    with pytest.raises((KeyError, AttributeError)):
        session.replace_transcripts([{"text": "first"}, None])
    assert [e["text"] for e in session.to_dict()["entries"]] == ["keep me"]


def test_cannot_attach_other_sessions_take():
    first, second = Session("one"), Session("two")
    take = first.begin_take()
    with pytest.raises(ValueError):
        second.add_transcript("wrong", take_id=take["id"])


def test_parallel_append_retains_all_entries():
    session = Session("Parallel")
    def append(index):
        Session.load(session.session_id).add_transcript(str(index), entry_key=f"chunk-{index}")
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(append, range(30)))
    assert {e["text"] for e in session.to_dict()["entries"]} == {str(i) for i in range(30)}


def test_jobs_persist_and_filter():
    session = Session("Jobs")
    store = Store()
    job = store.create_job(session.session_id, "transcribe", {"take_id": "x"})
    assert job["status"] == "queued"
    assert Store().get_job(job["id"]) == job
    store.update_job(job["id"], status="complete", result={"segments": []})
    assert Store().list_jobs(status="queued") == []
    assert Store().list_jobs(session_id=session.session_id)[0]["result"] == {"segments": []}
    with pytest.raises(ValueError):
        store.update_job(job["id"], invalid="field")


def test_database_remains_authoritative_if_export_is_changed():
    session = Session("Authoritative")
    session.add_transcript("original")
    path = session.save()
    path.write_text("corrupted")
    assert "original" in Session.load(session.session_id).to_markdown()
    assert "corrupted" not in Session.load(session.session_id).to_markdown()


def test_search_includes_notes_and_transcripts():
    first, second = Session("First"), Session("Second")
    first.add_note("Física")
    second.add_transcript("Quantum")
    assert [row["id"] for row in list_sessions("fÍsica")] == [first.session_id]
    assert [row["id"] for row in list_sessions("QUANTUM")] == [second.session_id]


def test_external_legacy_migration_preserves_original_and_copies_media(tmp_path, monkeypatch):
    from steno.session import import_legacy_sessions, Session, list_sessions
    from steno.config import Config
    import wave
    old = tmp_path / "previous"
    source = old / "sessions"
    source.mkdir(parents=True)
    doc = source / "20200101_120000_lesson.md"
    original = b'# Lesson\n\nFirst line\nSecond line\n![x](/api/sessions/20200101_120000_lesson/images/a.png)\n'
    doc.write_bytes(original)
    images = source / "images" / doc.stem
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"picture")
    with wave.open(str(doc.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 160)
    monkeypatch.setattr(Config, "legacy_data_dirs", classmethod(lambda cls: [old]))
    assert import_legacy_sessions() == 1
    assert import_legacy_sessions() == 0
    session = Session.load(doc.stem)
    assert session.audio_path() != doc.with_suffix(".wav")
    assert session.audio_path().is_file()
    assert "data:image/png;base64," in session.to_markdown(portable=True)
    session.add_note("new notes")
    session.delete()
    assert doc.read_bytes() == original
    assert doc.with_suffix(".wav").is_file()
    assert import_legacy_sessions() == 0


def test_legacy_same_name_in_two_locations_keeps_both(tmp_path, monkeypatch):
    from steno.session import import_legacy_sessions, list_sessions, Session
    from steno.config import Config
    roots = [tmp_path / "first", tmp_path / "second"]
    for n, root in enumerate(roots):
        (root / "sessions").mkdir(parents=True)
        (root / "sessions" / "old.md").write_text(f"# Old\nContent {n}")
    monkeypatch.setattr(Config, "legacy_data_dirs", classmethod(lambda cls: roots))
    assert import_legacy_sessions() == 2
    content = [Session.load(item["id"]).to_markdown() for item in list_sessions()]
    assert any("Content 0" in item for item in content)
    assert any("Content 1" in item for item in content)


def test_interrupted_delete_is_retried(tmp_path, monkeypatch):
    from steno.session import Session, recover_interrupted_takes
    from steno.storage import Store
    from pathlib import Path
    session = Session("delete")
    take = session.begin_take()
    path = Path(take["path"])
    path.write_bytes(b"audio")
    original = Path.unlink
    def fail_unlink(self, *args, **kwargs):
        if self == path:
            raise PermissionError("file temporarily busy")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    session.delete()
    assert path.exists()
    assert Store().get_session(session.session_id) is None
    monkeypatch.setattr(Path, "unlink", original)
    recover_interrupted_takes()
    assert not path.exists()


def test_interrupted_recording_recovers_trailing_pcm():
    from steno.session import Session, recover_interrupted_takes
    from steno.audio import WavWriter
    from pathlib import Path
    import wave
    session = Session("repair")
    take = session.begin_take()
    path = Path(take["path"])
    writer = WavWriter(path)
    writer.close()
    with path.open("ab") as stream:
        stream.write(b"\0\0" * 32000)
    recover_interrupted_takes()
    with wave.open(str(path)) as audio:
        assert audio.getnframes() == 32000
    assert session.takes()[0]["duration_seconds"] == 2


def test_locked_snapshot_cannot_resurrect_deleted_session(monkeypatch):
    from pathlib import Path
    from steno.session import Session, import_legacy_sessions, list_sessions
    session = Session("deleted snapshot")
    snapshot = session.save()
    original = Path.unlink

    def locked(path, *args, **kwargs):
        if path == snapshot:
            raise PermissionError("Windows file still open")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    session.delete()
    assert snapshot.is_file()
    assert import_legacy_sessions() == 0
    assert list_sessions() == []
