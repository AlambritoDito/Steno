"""Durable sessions backed by SQLite, with portable Markdown/ZIP exports."""

import base64
import io
import json
import os
import mimetypes
import re
import shutil
import unicodedata
import wave
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from steno.config import Config
from steno.storage import Store, now


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}\Z")


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def safe_path(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks escaping the application data directory."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Invalid asset path")
    return path


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")[:64] or "session"


def import_legacy_sessions() -> int:
    """Import local and previous-version directories without touching originals."""
    import hashlib
    store = Store()
    imported = 0
    roots = [store.root] + [root / "sessions" for root in Config.legacy_data_dirs()]
    visited = set()
    for root in roots:
        if not root.is_dir() or root.resolve() in visited:
            continue
        visited.add(root.resolve())
        external = root.resolve() != store.root.resolve()
        for path in sorted(root.glob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            source = str(path.resolve()) if external else path.name
            with store.connection() as db:
                if db.execute("SELECT 1 FROM legacy_imports WHERE source=?", (source,)).fetchone():
                    continue
                if not external and db.execute("SELECT 1 FROM sessions WHERE id=?", (path.stem,)).fetchone():
                    continue
                try:
                    content = path.read_bytes().decode("utf-8")
                except (UnicodeError, OSError):
                    continue
                session_id = path.stem if _ID.fullmatch(path.stem) else "legacy-" + hashlib.sha256(source.encode()).hexdigest()[:24]
                if db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
                    session_id = "legacy-" + hashlib.sha256(source.encode()).hexdigest()[:24]
                name = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")), path.stem)
                try:
                    created = datetime.strptime("_".join(path.stem.split("_")[:2]), "%Y%m%d_%H%M%S")
                except ValueError:
                    created = datetime.fromtimestamp(path.stat().st_mtime)
                timestamp = now()
                db.execute("INSERT OR IGNORE INTO legacy_imports VALUES (?,?,?)", (source, session_id, timestamp))
                owner = db.execute("SELECT session_id FROM legacy_imports WHERE source=?", (source,)).fetchone()
                if owner[0] != session_id:
                    continue
                cursor = db.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?,?,?,?,?)", (
                    session_id, name, created.isoformat(), timestamp, "", content, source,
                ))
                if not cursor.rowcount:
                    continue
                imported += 1
                if external:
                    image_root = root / "images" / path.stem
                    destination = store.root / "images" / session_id
                    if image_root.is_dir() and not image_root.is_symlink():
                        destination.mkdir(parents=True, exist_ok=True)
                        for image in image_root.iterdir():
                            if image.is_file() and not image.is_symlink():
                                shutil.copy2(image, destination / image.name)
                audio = path.with_suffix(".wav")
                if audio.is_file() and not audio.is_symlink():
                    take_id = uuid4().hex
                    stored_audio = audio
                    if external:
                        stored_audio = store.root / "audio" / session_id / f"{take_id}.wav"
                        stored_audio.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(audio, stored_audio)
                    frames, rate = _wave_info(stored_audio)
                    db.execute("INSERT INTO takes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                        take_id, session_id, "full", "legacy-copy" if external else "legacy",
                        str(stored_audio.relative_to(store.root)), "complete", created.isoformat(),
                        timestamp, frames / rate if rate else None, rate or 16000, frames, 0,
                    ))
    return imported


def _wave_info(path):
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes(), audio.getframerate()
    except (OSError, EOFError, wave.Error):
        return 0, 0


class Session:
    def __init__(self, name: str):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Session name is required")
        self.name = name.strip()
        self.created_at = datetime.now()
        self.session_id = self.created_at.strftime("%Y%m%d_%H%M%S") + f"_{_slugify(name)}-{uuid4().hex[:12]}"
        self._store = Store()
        with self._store.connection() as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)", (
                self.session_id, self.name, self.created_at.isoformat(), now(), "", None, None,
            ))

    @classmethod
    def load(cls, session_id: str) -> "Session":
        validate_id(session_id)
        store = Store()
        row = store.get_session(session_id)
        if row is None:
            import_legacy_sessions()
            row = store.get_session(session_id)
        if row is None:
            raise FileNotFoundError(session_id)
        session = cls.__new__(cls)
        session.session_id = session_id
        session.name = row["name"]
        session.created_at = datetime.fromisoformat(row["created_at"])
        session._store = store
        return session

    def _row(self):
        row = self._store.get_session(self.session_id)
        if row is None:
            raise FileNotFoundError(self.session_id)
        return row

    @property
    def notes(self):
        return self._row()["notes"]

    @property
    def _entries(self):
        """Compatibility read view. Mutations must go through transactional methods."""
        result = []
        for row in self._store.entries(self.session_id):
            entry = dict(row)
            entry["type"] = entry.pop("kind")
            entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
            if entry["type"] == "image":
                entry["tag"] = self._image_tag(entry)
            if entry["elapsed_seconds"] is not None:
                seconds = entry["elapsed_seconds"]
                entry["elapsed"] = f"{seconds // 60}:{seconds % 60:02d}"
            result.append(entry)
        notes = self.notes
        if notes:
            result.append({"type": "note", "text": notes, "timestamp": self.created_at})
        return result

    def _insert(self, kind, text, timestamp=None, **fields):
        allowed = {"take_id", "start", "end", "elapsed_seconds", "asset_path", "mime_type", "entry_key"}
        if not set(fields) <= allowed:
            raise ValueError("Invalid entry fields")
        timestamp = timestamp or datetime.now()
        columns = ["session_id", "kind", "text", "timestamp", *fields]
        values = [self.session_id, kind, text, timestamp.isoformat(), *fields.values()]
        with self._store.connection() as db:
            if fields.get("take_id") and not db.execute(
                "SELECT 1 FROM takes WHERE id=? AND session_id=?",
                (fields["take_id"], self.session_id),
            ).fetchone():
                raise ValueError("Take does not belong to session")
            db.execute(f"INSERT OR IGNORE INTO entries ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), self.session_id))

    def add_transcript(self, text: str, timestamp: datetime | None = None, *,
                       take_id=None, start=None, end=None, entry_key=None):
        if take_id is not None and entry_key is None and start is not None and end is not None:
            entry_key = f"segment:{take_id}:{float(start)}:{float(end)}"
        self._insert("transcript", text, timestamp, take_id=take_id, start=start,
                     end=end, entry_key=entry_key)

    def append_transcript(self, text, timestamp=None, **fields) -> Path:
        self.add_transcript(text, timestamp, **fields)
        # This small append is a convenience export only. SQLite is authoritative.
        path = self._snapshot_path()
        if not path.exists():
            return self.save()
        with path.open("a", encoding="utf-8") as output:
            output.write(f"**[{(timestamp or datetime.now()).strftime('%H:%M:%S')}]** {text}\n\n")
        return path

    def add_note(self, markdown: str, elapsed_seconds: int | None = None):
        if elapsed_seconds is not None:
            self._insert("note", markdown, elapsed_seconds=max(0, int(elapsed_seconds)))
        else:
            with self._store.connection() as db:
                db.execute("UPDATE sessions SET notes=?,updated_at=? WHERE id=?",
                           (markdown, now(), self.session_id))

    def add_image(self, image_data: bytes, mime_type: str, caption: str = "") -> dict:
        extensions = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
        if mime_type not in extensions:
            raise ValueError("Unsupported image type")
        filename = f"img_{uuid4().hex}{extensions[mime_type]}"
        relative = f"images/{self.session_id}/{filename}"
        path = safe_path(self._store.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_data)
        try:
            self._insert("image", caption, asset_path=relative, mime_type=mime_type)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        url = f"/api/sessions/{self.session_id}/images/{filename}"
        return {"tag": self._image_tag({"text": caption, "asset_path": relative}), "image_url": url}

    def _image_tag(self, entry, portable=False):
        caption = entry["text"].replace("\\", "\\\\").replace("]", "\\]").replace("\n", " ")
        path = safe_path(self._store.root, entry["asset_path"])
        url = f"/api/sessions/{self.session_id}/images/{path.name}"
        if portable:
            url = self._data_uri(path, entry.get("mime_type"))
        return f"![{caption}]({url})"

    @staticmethod
    def _data_uri(path, mime_type=None):
        mime = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    def to_markdown(self, portable=False) -> str:
        row = self._row()
        parts = [row["legacy_markdown"]] if row["legacy_markdown"] is not None else [
            f"# {self.name}\n\n**Date:** {self.created_at:%Y-%m-%d %H:%M}\n**Duration:** {self.get_duration()}\n\n---\n"
        ]
        for entry in self._store.entries(self.session_id):
            if entry["kind"] == "transcript":
                stamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
                if entry["start"] is not None:
                    seconds = int(entry["start"])
                    stamp = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
                parts.append(f"**[{stamp}]** {entry['text']}\n")
            elif entry["kind"] == "note":
                seconds = entry["elapsed_seconds"]
                prefix = f"**[{seconds // 60}:{seconds % 60:02d}]** " if seconds is not None else "### Notes\n\n"
                parts.append(prefix + entry["text"] + "\n")
            elif entry["kind"] == "image":
                parts.append(self._image_tag(entry, portable=portable) + "\n")
        if row["notes"]:
            parts.append("### Notes\n\n" + row["notes"] + "\n")
        # Preserve the original document byte-for-byte when no content was added.
        content = "\n".join(parts)
        if portable:
            content = self._rewrite_assets(content, lambda path: self._data_uri(path))
        elif row["legacy_source"]:
            content = self._rewrite_assets(content, lambda path: f"/api/sessions/{self.session_id}/images/{path.name}")
        return content

    def _rewrite_assets(self, content, replacement):
        source = self._row()["legacy_source"]
        permitted = {self.session_id}
        if source:
            permitted.add(Path(source).stem)
        pattern = re.compile(r"/api/sessions/([A-Za-z0-9_-]+)/images/([A-Za-z0-9_.-]+)")
        def replace(match):
            if match[1] not in permitted:
                return match[0]
            path = safe_path(self._store.root, f"images/{self.session_id}/{match[2]}")
            if not path.is_file():
                path = safe_path(self._store.root, f"images/{match[1]}/{match[2]}")
            if not path.is_file():
                return match[0]
            return replacement(path)
        return pattern.sub(replace, content)

    def _snapshot_path(self):
        if self._row()["legacy_source"]:
            return safe_path(self._store.root, f"exports/{self.session_id}.md")
        return safe_path(self._store.root, f"{self.session_id}.md")

    def save(self) -> Path:
        """Refresh a derived Markdown export without ever overwriting legacy files."""
        path = self._snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(self.to_markdown(), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def to_dict(self):
        row = self._row()
        entries = self._store.entries(self.session_id)
        for entry in entries:
            entry["type"] = entry["kind"]
            if entry["kind"] == "image":
                entry["tag"] = self._image_tag(entry)
                entry["image_url"] = f"/api/sessions/{self.session_id}/images/{Path(entry['asset_path']).name}"
        return {"id": self.session_id, "session_id": self.session_id, "name": self.name,
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "notes": row["notes"], "legacy_markdown": self._rewrite_assets(row["legacy_markdown"], lambda path: f"/api/sessions/{self.session_id}/images/{path.name}") if row["legacy_markdown"] is not None else None,
                "entries": entries, "takes": self.takes(), "has_audio": self.has_audio()}

    def begin_take(self, mode="full", source="microphone", suffix=".wav"):
        if suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}:
            raise ValueError("Unsupported audio format")
        take_id = uuid4().hex
        relative = f"audio/{self.session_id}/{take_id}{suffix.lower()}"
        path = safe_path(self._store.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = now()
        with self._store.connection() as db:
            db.execute("INSERT INTO takes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                take_id, self.session_id, mode, source, relative, "recording",
                timestamp, timestamp, None, 16000, 0, 0,
            ))
        return next(take for take in self.takes() if take["id"] == take_id)

    def takes(self):
        return self._store.takes(self.session_id)

    def update_take(self, take_id, **fields):
        if not any(take["id"] == take_id for take in self.takes()):
            raise FileNotFoundError(take_id)
        return self._store.update_take(take_id, **fields)

    def finish_take(self, take_id, status="complete", duration_seconds=None):
        path = self.audio_path(take_id)
        frames, rate = _wave_info(path)
        fields = {"status": status}
        if rate:
            fields.update(frames=frames, sample_rate=rate)
        if duration_seconds is not None:
            fields["duration_seconds"] = duration_seconds
        elif rate:
            fields["duration_seconds"] = frames / rate
        return self.update_take(take_id, **fields)

    def audio_path(self, take_id=None) -> Path:
        takes = self.takes()
        if take_id is not None:
            validate_id(take_id)
            take = next((t for t in takes if t["id"] == take_id), None)
            if take is None:
                raise FileNotFoundError(take_id)
            return Path(take["path"])
        if takes:
            return Path(takes[-1]["path"])
        return safe_path(self._store.root, f"{self.session_id}.wav")

    def has_audio(self):
        return any(Path(t["path"]).is_file() for t in self.takes()) or self.audio_path().is_file()

    def replace_transcripts(self, segments, take_id=None):
        with self._store.connection() as db:
            if take_id is None:
                db.execute("DELETE FROM entries WHERE session_id=? AND kind='transcript'", (self.session_id,))
            else:
                if not db.execute("SELECT 1 FROM takes WHERE id=? AND session_id=?", (take_id, self.session_id)).fetchone():
                    raise ValueError("Take does not belong to session")
                db.execute("DELETE FROM entries WHERE session_id=? AND kind='transcript' AND take_id=?", (self.session_id, take_id))
            for segment in segments:
                if not segment.get("text", "").strip():
                    continue
                start, end = segment.get("start"), segment.get("end")
                key = f"segment:{take_id}:{float(start)}:{float(end)}" if take_id and start is not None and end is not None else None
                db.execute("INSERT OR IGNORE INTO entries (session_id,kind,text,timestamp,take_id,start,end,entry_key) VALUES (?,?,?,?,?,?,?,?)", (
                    self.session_id, "transcript", segment["text"], now(), take_id, start, end, key,
                ))
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), self.session_id))

    def get_duration(self):
        duration = sum(t["duration_seconds"] or 0 for t in self.takes())
        if not duration:
            stamps = [datetime.fromisoformat(e["timestamp"]) for e in self._store.entries(self.session_id)]
            duration = max(0, (max(stamps) - self.created_at).total_seconds()) if stamps else 0
        total = int(duration)
        return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"

    def export_zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            included = set()
            def include(path):
                name = f"images/{path.parent.name}/{path.name}"
                if name not in included:
                    archive.write(path, name)
                    included.add(name)
                return name
            content = self._rewrite_assets(self.to_markdown(), include)
            archive.writestr("session.md", content)
            for take in self.takes():
                path = Path(take["path"])
                if path.is_file():
                    archive.write(path, f"audio/{take['id']}{path.suffix}")
            metadata = self.to_dict()
            for take in metadata["takes"]:
                take["path"] = f"audio/{take['id']}{Path(take['path']).suffix}"
                take.pop("relative_path", None)
            archive.writestr("session.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        return buffer.getvalue()

    def delete(self):
        row = self._row()
        paths = [Path(take["path"]) for take in self.takes() if take["source"] != "legacy"]
        paths += [safe_path(self._store.root, entry["asset_path"])
                  for entry in self._store.entries(self.session_id) if entry["asset_path"]]
        paths.append(self._snapshot_path())
        # Copied legacy images belong to this installation; source originals stay intact.
        if row["legacy_source"] and Path(row["legacy_source"]).is_absolute():
            image_dir = safe_path(self._store.root, f"images/{self.session_id}")
            if image_dir.is_dir():
                paths += [p for p in image_dir.iterdir() if p.is_file() and not p.is_symlink()]
        with self._store.connection() as db:
            # A derived Markdown snapshot may remain locked (notably on Windows).
            # Remember its deletion before removing the row, so legacy migration
            # cannot resurrect the session while filesystem cleanup is retried.
            if not row["legacy_source"]:
                db.execute("INSERT OR IGNORE INTO legacy_imports VALUES (?,?,?)",
                           (self._snapshot_path().name, self.session_id, now()))
            for path in paths:
                relative = str(path.relative_to(self._store.root))
                db.execute("INSERT OR IGNORE INTO pending_deletions VALUES (?)", (relative,))
            db.execute("DELETE FROM sessions WHERE id=?", (self.session_id,))
        self._store.cleanup_deletions()


def _repair_recorded_wave(path):
    """Recover a Steno PCM WAV whose final header update was interrupted."""
    import struct
    if not path.is_file() or path.is_symlink():
        return
    with path.open("r+b") as audio:
        header = audio.read(44)
        if (len(header) != 44 or header[:4] != b"RIFF" or header[8:16] != b"WAVEfmt "
                or header[36:40] != b"data" or header[20:24] != b"\x01\x00\x01\x00"):
            return
        size = path.stat().st_size
        data_size = (size - 44) // 2 * 2
        if data_size > 0xffffffff - 36:
            return
        audio.truncate(44 + data_size)
        audio.seek(4)
        audio.write(struct.pack("<I", 36 + data_size))
        audio.seek(40)
        audio.write(struct.pack("<I", data_size))
        audio.flush()
        os.fsync(audio.fileno())


def recover_interrupted_takes() -> int:
    store = Store()
    store.cleanup_deletions()
    with store.connection() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM takes WHERE status='recording'")]
    for row in rows:
        session = Session.load(row["session_id"])
        path = session.audio_path(row["id"])
        _repair_recorded_wave(path)
        session.finish_take(row["id"], status="interrupted")
    return len(rows)


def list_sessions(query=None) -> list[dict]:
    import_legacy_sessions()
    store = Store()
    with store.connection() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM sessions ORDER BY created_at DESC,id DESC")]
    result = []
    for row in rows:
        session = Session.load(row["id"])
        content = session.to_markdown()
        if query and query.casefold() not in (row["name"] + "\n" + content).casefold():
            continue
        result.append({"id": row["id"], "name": row["name"], "created_at": row["created_at"],
                       "updated_at": row["updated_at"], "size_kb": round(len(content.encode("utf-8")) / 1024, 1),
                       "has_audio": session.has_audio()})
    return result
