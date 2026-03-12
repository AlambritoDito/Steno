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

    def add_note(self, markdown: str, elapsed_seconds: int | None = None) -> None:
        """Append a user note block.

        If elapsed_seconds is provided (light mode), the note is stamped
        with the elapsed time since recording started.
        """
        entry = {
            "type": "note",
            "text": markdown,
            "timestamp": datetime.now(),
        }
        if elapsed_seconds is not None:
            mins = elapsed_seconds // 60
            secs = elapsed_seconds % 60
            entry["elapsed"] = f"{mins}:{secs:02d}"
        self._entries.append(entry)

    def add_image(self, image_data: bytes, mime_type: str, caption: str = "") -> dict:
        """Save the image as a file and reference it in the session.

        Returns a dict with 'tag' (markdown) and 'image_url' (serving path).
        """
        # Determine file extension from mime type
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
        }
        ext = ext_map.get(mime_type, ".png")

        # Generate unique filename
        ts = datetime.now().strftime("%H%M%S")
        filename = f"img_{ts}_{len(self._entries)}{ext}"

        # Save to disk
        images_dir = Config.images_path(self.session_id)
        image_path = images_dir / filename
        image_path.write_bytes(image_data)

        # Use relative URL for markdown and serving
        image_url = f"/api/sessions/{self.session_id}/images/{filename}"
        tag = f"![{caption}]({image_url})"

        self._entries.append({
            "type": "image",
            "tag": tag,
            "timestamp": datetime.now(),
        })
        return {"tag": tag, "image_url": image_url}

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
                if "elapsed" in entry:
                    # Light mode: timestamped note with elapsed time
                    lines.append(f"**[{entry['elapsed']}]** {entry['text']}")
                    lines.append("")
                else:
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

        Restores name, session_id, and all entries from the file.
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

        # Parse entries from markdown content
        session._entries = cls._parse_entries(content, session.created_at)

        return session

    @staticmethod
    def _parse_entries(content: str, created_at: datetime) -> list[dict]:
        """Parse markdown content into structured entries."""
        entries: list[dict] = []
        lines = content.split("\n")
        i = 0

        # Patterns
        transcript_re = re.compile(r"^\*\*\[(\d{2}:\d{2}:\d{2})\]\*\*\s+(.+)$")
        image_re = re.compile(r"^!\[([^\]]*)\]\((.+)\)$")
        elapsed_note_re = re.compile(r"^\*\*\[(\d+:\d{2})\]\*\*\s+(.+)$")

        while i < len(lines):
            line = lines[i]

            # Skip header lines (title, date, duration, hr)
            if (line.startswith("# ") or line.startswith("**Date:**")
                    or line.startswith("**Duration:**") or line == "---"
                    or line.strip() == ""):
                i += 1
                continue

            # Transcript entry: **[HH:MM:SS]** text
            m = transcript_re.match(line)
            if m:
                ts_str, text = m.group(1), m.group(2)
                try:
                    t = datetime.strptime(ts_str, "%H:%M:%S")
                    timestamp = created_at.replace(
                        hour=t.hour, minute=t.minute, second=t.second
                    )
                except ValueError:
                    timestamp = created_at
                entries.append({
                    "type": "transcript",
                    "text": text,
                    "timestamp": timestamp,
                })
                i += 1
                continue

            # Image entry: ![caption](url or data:...)
            m = image_re.match(line)
            if m:
                entries.append({
                    "type": "image",
                    "tag": line,
                    "timestamp": created_at,
                })
                i += 1
                continue

            # Note section: ### Notes followed by content
            if line == "### Notes":
                i += 1
                # Skip blank line after header
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                note_lines = []
                while i < len(lines):
                    # Stop at next section or transcript entry
                    if (transcript_re.match(lines[i]) or lines[i] == "### Notes"
                            or image_re.match(lines[i])):
                        break
                    note_lines.append(lines[i])
                    i += 1
                # Remove trailing blank lines
                while note_lines and note_lines[-1].strip() == "":
                    note_lines.pop()
                if note_lines:
                    entries.append({
                        "type": "note",
                        "text": "\n".join(note_lines),
                        "timestamp": created_at,
                    })
                continue

            # Elapsed-time note: **[MM:SS]** text (light mode)
            m = elapsed_note_re.match(line)
            if m:
                elapsed_str, text = m.group(1), m.group(2)
                entries.append({
                    "type": "note",
                    "text": text,
                    "timestamp": created_at,
                    "elapsed": elapsed_str,
                })
                i += 1
                continue

            i += 1

        return entries

    def audio_path(self) -> Path:
        """Return the .wav path for this session."""
        return Config.sessions_path() / f"{self.session_id}.wav"

    def has_audio(self) -> bool:
        """Return whether this session has a saved audio file."""
        return self.audio_path().exists()

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
        wav_path = path.parent / f"{session_id}.wav"

        result.append({
            "id": session_id,
            "name": name,
            "created_at": created_at,
            "size_kb": size_kb,
            "has_audio": wav_path.exists(),
        })

    return result
