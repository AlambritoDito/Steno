"""Session management for Steno."""

import base64
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from steno.config import Config


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text or "session"


class Session:
    """Represents a transcription session."""

    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        slug = _slugify(name)
        self.session_id = self.created_at.strftime("%Y%m%d_%H%M%S") + f"_{slug}"
        self._entries: list[dict] = []

    def add_transcript(self, text: str, timestamp: datetime | None = None) -> None:
        """Append a transcript entry."""
        if timestamp is None:
            timestamp = datetime.now()
        self._entries.append({
            "type": "transcript",
            "text": text,
            "timestamp": timestamp,
        })

    def add_note(self, markdown: str) -> None:
        """Append a user note block."""
        self._entries.append({
            "type": "note",
            "text": markdown,
            "timestamp": datetime.now(),
        })

    def add_image(self, image_data: bytes, mime_type: str, caption: str = "") -> str:
        """Embed the image as base64 in the MD.

        Returns the ![caption](data:...) tag.
        """
        b64 = base64.b64encode(image_data).decode("ascii")
        tag = f"![{caption}](data:{mime_type};base64,{b64})"
        self._entries.append({
            "type": "image",
            "tag": tag,
            "timestamp": datetime.now(),
        })
        return tag

    def to_markdown(self) -> str:
        """Generate the full Markdown document."""
        lines = []
        lines.append(f"# {self.name}")
        lines.append("")
        lines.append(f"**Date:** {self.created_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Duration:** {self.get_duration()}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for entry in self._entries:
            if entry["type"] == "transcript":
                ts = entry["timestamp"].strftime("%H:%M:%S")
                lines.append(f"**[{ts}]** {entry['text']}")
                lines.append("")
            elif entry["type"] == "note":
                lines.append("### Notes")
                lines.append("")
                lines.append(entry["text"])
                lines.append("")
            elif entry["type"] == "image":
                lines.append(entry["tag"])
                lines.append("")

        return "\n".join(lines)

    def append_transcript(self, text: str, timestamp: datetime | None = None) -> Path:
        """Append a single transcript line to the session file (fast path).

        Used during recording to avoid rewriting the entire file on every chunk.
        Call save() when recording stops to write the canonical file.
        """
        if timestamp is None:
            timestamp = datetime.now()
        self._entries.append({
            "type": "transcript",
            "text": text,
            "timestamp": timestamp,
        })
        sessions_dir = Config.sessions_path()
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.session_id}.md"

        # If file doesn't exist yet, write the header first
        if not path.exists():
            header = (
                f"# {self.name}\n\n"
                f"**Date:** {self.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"**Duration:** {self.get_duration()}\n\n"
                "---\n\n"
            )
            path.write_text(header, encoding="utf-8")

        ts = timestamp.strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"**[{ts}]** {text}\n\n")
        return path

    def save(self) -> Path:
        """Save full .md to sessions/ directory (canonical write)."""
        sessions_dir = Config.sessions_path()
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.session_id}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str) -> "Session":
        """Reload a session from disk.

        Restores name and session_id from the file.
        """
        sessions_dir = Config.sessions_path()
        path = sessions_dir / f"{session_id}.md"
        content = path.read_text(encoding="utf-8")

        # Parse header to extract name
        name = "Untitled"
        for line in content.split("\n"):
            if line.startswith("# "):
                name = line[2:].strip()
                break

        session = cls.__new__(cls)
        session.name = name
        session.session_id = session_id
        session._entries = []

        # Parse timestamp from session_id (YYYYMMDD_HHMMSS_slug)
        try:
            ts_str = "_".join(session_id.split("_")[:2])
            session.created_at = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            session.created_at = datetime.now()

        return session

    def get_duration(self) -> str:
        """Return elapsed time as HH:MM:SS."""
        if not self._entries:
            return "00:00:00"

        timestamps = [e["timestamp"] for e in self._entries]
        delta = max(timestamps) - self.created_at
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            total_seconds = 0
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def list_sessions() -> list[dict]:
    """Return all saved sessions with id, name, created_at, size_kb."""
    sessions_dir = Config.sessions_path()
    if not sessions_dir.exists():
        return []

    result = []
    for path in sorted(sessions_dir.glob("*.md"), reverse=True):
        session_id = path.stem
        content = path.read_text(encoding="utf-8")

        name = "Untitled"
        for line in content.split("\n"):
            if line.startswith("# "):
                name = line[2:].strip()
                break

        try:
            ts_str = "_".join(session_id.split("_")[:2])
            created_at = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").isoformat()
        except (ValueError, IndexError):
            created_at = ""

        size_kb = round(path.stat().st_size / 1024, 1)

        result.append({
            "id": session_id,
            "name": name,
            "created_at": created_at,
            "size_kb": size_kb,
        })

    return result
