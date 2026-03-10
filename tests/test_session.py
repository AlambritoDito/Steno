"""Tests for steno.session module."""

import re
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from steno.config import Config
from steno.session import Session, list_sessions


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    """Use a temporary directory for sessions during tests."""
    monkeypatch.setattr(Config, "SESSIONS_DIR", str(tmp_path))
    # Override sessions_path to return tmp_path directly
    monkeypatch.setattr(Config, "sessions_path", classmethod(lambda cls: tmp_path))
    yield


def test_session_creation():
    """Session has session_id, name, and created_at."""
    s = Session("Test Session")
    assert s.session_id is not None
    assert s.name == "Test Session"
    assert isinstance(s.created_at, datetime)


def test_session_id_format():
    """session_id matches YYYYMMDD_HHMMSS_slug format."""
    s = Session("My Physics Class")
    pattern = r"^\d{8}_\d{6}_[a-z0-9-]+$"
    assert re.match(pattern, s.session_id), f"session_id '{s.session_id}' does not match expected format"


def test_add_transcript():
    """Added text appears in to_markdown() output."""
    s = Session("Test")
    s.add_transcript("Hello world", datetime.now())
    md = s.to_markdown()
    assert "Hello world" in md


def test_add_note():
    """Added note appears in to_markdown() output."""
    s = Session("Test")
    s.add_note("This is a **note**")
    md = s.to_markdown()
    assert "This is a **note**" in md


def test_add_image_returns_md_tag():
    """Returns a string starting with ![."""
    s = Session("Test")
    tag = s.add_image(b"\x89PNG\r\n", "image/png", "whiteboard")
    assert tag.startswith("![")


def test_add_image_embedded_in_markdown():
    """to_markdown() contains data:image."""
    s = Session("Test")
    s.add_image(b"\x89PNG\r\n", "image/png", "diagram")
    md = s.to_markdown()
    assert "data:image" in md


def test_to_markdown_has_header():
    """Output contains session name and date."""
    s = Session("Physics 101")
    md = s.to_markdown()
    assert "Physics 101" in md
    assert "Date:" in md


def test_save_creates_file():
    """save() creates a .md file on disk."""
    s = Session("Save Test")
    path = s.save()
    assert path.exists()
    assert path.suffix == ".md"


def test_save_and_load_roundtrip():
    """save then load preserves name and session_id."""
    s = Session("Roundtrip Test")
    s.add_transcript("Some text", datetime.now())
    s.save()

    loaded = Session.load(s.session_id)
    assert loaded.name == s.name
    assert loaded.session_id == s.session_id


def test_list_sessions_after_save():
    """Saved session appears in list_sessions()."""
    s = Session("Listed Session")
    s.save()

    sessions = list_sessions()
    ids = [sess["id"] for sess in sessions]
    assert s.session_id in ids


def test_get_duration_format():
    """Returns string matching HH:MM:SS."""
    s = Session("Duration Test")
    duration = s.get_duration()
    assert re.match(r"^\d{2}:\d{2}:\d{2}$", duration)


# --- v0.2.0: append_transcript ---


def test_append_transcript_creates_file_with_header():
    """append_transcript() creates the file with a header on first call."""
    s = Session("Append Test")
    path = s.append_transcript("First line", datetime.now())
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Append Test" in content
    assert "First line" in content


def test_append_transcript_appends_without_rewriting_header():
    """Multiple appends keep a single header."""
    s = Session("Multi Append")
    s.append_transcript("Line one", datetime.now())
    s.append_transcript("Line two", datetime.now())
    path = Config.sessions_path() / f"{s.session_id}.md"
    content = path.read_text(encoding="utf-8")
    assert content.count("# Multi Append") == 1
    assert "Line one" in content
    assert "Line two" in content


def test_append_transcript_returns_path():
    """append_transcript() returns a Path object."""
    s = Session("Path Test")
    result = s.append_transcript("text", datetime.now())
    assert isinstance(result, Path)


def test_save_overwrites_append_file():
    """save() after append_transcript() writes the canonical format."""
    s = Session("Overwrite Test")
    s.append_transcript("appended text", datetime.now())
    s.save()
    path = Config.sessions_path() / f"{s.session_id}.md"
    content = path.read_text(encoding="utf-8")
    assert "**Duration:**" in content
    assert "appended text" in content
