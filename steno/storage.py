"""Transactional local storage. Markdown files are exports, never the database."""

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from steno.config import Config


def now() -> str:
    return datetime.now().isoformat()


class Store:
    """Open short-lived connections so worker threads never share a connection."""

    def __init__(self):
        self.root = Path(Config.sessions_path())
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "steno.sqlite3"
        with self.connection() as db:
            if db.execute("PRAGMA user_version").fetchone()[0] > 2:
                raise RuntimeError("This database needs a newer version of Steno")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '', legacy_markdown TEXT,
                    legacy_source TEXT
                );
                CREATE TABLE IF NOT EXISTS legacy_imports (
                    source TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS takes (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL
                        REFERENCES sessions(id) ON DELETE CASCADE,
                    mode TEXT NOT NULL, source TEXT NOT NULL, path TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, duration_seconds REAL,
                    sample_rate INTEGER NOT NULL DEFAULT 16000,
                    frames INTEGER NOT NULL DEFAULT 0,
                    processed_frames INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL, take_id TEXT REFERENCES takes(id),
                    start REAL, end REAL, elapsed_seconds INTEGER,
                    asset_path TEXT, mime_type TEXT, entry_key TEXT,
                    UNIQUE(session_id, entry_key)
                );
                CREATE INDEX IF NOT EXISTS entries_session ON entries(session_id, id);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id)
                        ON DELETE CASCADE,
                    kind TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_checkpoints (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    take_id TEXT NOT NULL REFERENCES takes(id) ON DELETE CASCADE,
                    processed_frames INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(job_id,take_id)
                );
                CREATE TABLE IF NOT EXISTS job_checkpoint_batches (
                    job_id TEXT NOT NULL, take_id TEXT NOT NULL,
                    end_frame INTEGER NOT NULL, segments TEXT NOT NULL,
                    PRIMARY KEY(job_id,take_id,end_frame),
                    FOREIGN KEY(job_id,take_id) REFERENCES job_checkpoints(job_id,take_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS pending_deletions (path TEXT PRIMARY KEY);
                PRAGMA user_version=2;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def get_session(self, session_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return dict(row) if row else None

    def entries(self, session_id):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM entries WHERE session_id=? ORDER BY id", (session_id,)
            )]

    def takes(self, session_id):
        with self.connection() as db:
            return [self._take(row) for row in db.execute(
                "SELECT * FROM takes WHERE session_id=? ORDER BY created_at,id", (session_id,)
            )]

    def _take(self, row):
        result = dict(row)
        path = (self.root / result["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid take path")
        result["relative_path"] = result["path"]
        result["path"] = str(path)
        return result

    def update_take(self, take_id, **fields):
        allowed = {"status", "duration_seconds", "sample_rate", "frames", "processed_frames"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("Invalid take fields")
        for key in ("frames", "processed_frames", "sample_rate"):
            if key in fields and (not isinstance(fields[key], int) or fields[key] < 0):
                raise ValueError(f"Invalid {key}")
        fields["updated_at"] = now()
        with self.connection() as db:
            db.execute(f"UPDATE takes SET {','.join(k + '=?' for k in fields)} WHERE id=?",
                       (*fields.values(), take_id))
            row = db.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
            if row is None:
                raise FileNotFoundError(take_id)
            return self._take(row)

    @staticmethod
    def _job(row):
        if row is None:
            return None
        result = dict(row)
        for key in ("payload", "result"):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        return result

    def create_job(self, session_id, kind, payload):
        job_id = uuid4().hex
        timestamp = now()
        with self.connection() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)", (
                job_id, session_id, kind, "queued", json.dumps(payload), None,
                None, timestamp, timestamp,
            ))
        return self.get_job(job_id)

    def update_job(self, job_id, **fields):
        if not fields or not set(fields) <= {"status", "payload", "result", "error"}:
            raise ValueError("Invalid job fields")
        for key in ("payload", "result"):
            if key in fields:
                fields[key] = json.dumps(fields[key]) if fields[key] is not None else None
        fields["updated_at"] = now()
        with self.connection() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {','.join(k + '=?' for k in fields)} WHERE id=?",
                (*fields.values(), job_id),
            )
            if cursor.rowcount == 0:
                raise FileNotFoundError(job_id)
        return self.get_job(job_id)

    def get_job(self, job_id):
        with self.connection() as db:
            return self._job(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def file_checkpoint(self, job_id, take_id):
        """A checkpoint belongs to one immutable job payload and one source take."""
        with self.connection() as db:
            owner = db.execute("SELECT 1 FROM jobs j JOIN takes t ON t.session_id=j.session_id "
                               "WHERE j.id=? AND t.id=?", (job_id, take_id)).fetchone()
            if owner is None:
                raise ValueError("Take does not belong to this job's session")
            db.execute("INSERT OR IGNORE INTO job_checkpoints(job_id,take_id) VALUES (?,?)", (job_id, take_id))
            return dict(db.execute("SELECT * FROM job_checkpoints WHERE job_id=? AND take_id=?",
                                   (job_id, take_id)).fetchone())

    def save_file_batch(self, job_id, take_id, end_frame, segments):
        """Commit a bounded batch and its cursor together, including silent windows."""
        if not isinstance(end_frame, int) or end_frame < 0:
            raise ValueError("Invalid checkpoint frame")
        normalized = []
        for segment in segments:
            start, end = float(segment["start"]), float(segment["end"])
            text = segment["text"]
            if not isinstance(text, str) or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError("Invalid checkpoint segment")
            if text.strip():
                normalized.append({"start": start, "end": end, "text": text})
        encoded = json.dumps(normalized, allow_nan=False)
        with self.connection() as db:
            checkpoint = db.execute("SELECT * FROM job_checkpoints WHERE job_id=? AND take_id=?",
                                    (job_id, take_id)).fetchone()
            if checkpoint is None or checkpoint["completed"]:
                raise ValueError("Checkpoint is missing or already committed")
            if end_frame < checkpoint["processed_frames"]:
                raise ValueError("Checkpoint cannot move backwards")
            # Identical cursors are retries of the same batch, not new text.
            db.execute("INSERT OR IGNORE INTO job_checkpoint_batches VALUES (?,?,?,?)",
                       (job_id, take_id, end_frame, encoded))
            db.execute("UPDATE job_checkpoints SET processed_frames=? WHERE job_id=? AND take_id=?",
                       (end_frame, job_id, take_id))

    def commit_file_checkpoint(self, job_id, take_id):
        """Publish a completed take atomically; original text survives any failure.

        Retain the small completion marker so a multi-take retry skips this take.
        Staged text is removed only in the transaction that publishes its entries.
        """
        with self.connection() as db:
            checkpoint = db.execute("SELECT * FROM job_checkpoints WHERE job_id=? AND take_id=?",
                                    (job_id, take_id)).fetchone()
            if checkpoint is None:
                raise ValueError("Checkpoint does not exist")
            if checkpoint["completed"]:
                return checkpoint["segment_count"]
            session_id = db.execute("SELECT session_id FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
            timestamp = now()
            db.execute("DELETE FROM entries WHERE session_id=? AND kind='transcript' AND take_id=?",
                       (session_id, take_id))
            count = 0
            batches = db.execute("SELECT segments FROM job_checkpoint_batches WHERE job_id=? AND take_id=? ORDER BY end_frame",
                                 (job_id, take_id))
            for row in batches:
                for segment in json.loads(row["segments"]):
                    start, end = segment["start"], segment["end"]
                    cursor = db.execute("INSERT OR IGNORE INTO entries "
                                        "(session_id,kind,text,timestamp,take_id,start,end,entry_key) VALUES (?,?,?,?,?,?,?,?)",
                                        (session_id, "transcript", segment["text"], timestamp, take_id,
                                         start, end, f"segment:{take_id}:{float(start)}:{float(end)}"))
                    count += cursor.rowcount
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (timestamp, session_id))
            db.execute("UPDATE takes SET status='complete',duration_seconds=?,updated_at=? WHERE id=?",
                       (checkpoint["processed_frames"] / Config.SAMPLE_RATE, timestamp, take_id))
            db.execute("UPDATE job_checkpoints SET completed=1,segment_count=? WHERE job_id=? AND take_id=?",
                       (count, job_id, take_id))
            db.execute("DELETE FROM job_checkpoint_batches WHERE job_id=? AND take_id=?", (job_id, take_id))
            return count

    def list_jobs(self, session_id=None, status=None):
        conditions, args = [], []
        for column, value in (("session_id", session_id), ("status", status)):
            if value is not None:
                conditions.append(column + "=?")
                args.append(value)
        query = "SELECT * FROM jobs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        with self.connection() as db:
            return [self._job(row) for row in db.execute(query + " ORDER BY created_at,id", args)]

    def cleanup_deletions(self):
        """Retry interrupted deletions; originals are never added to this ledger."""
        with self.connection() as db:
            paths = [row[0] for row in db.execute("SELECT path FROM pending_deletions")]
        for relative in paths:
            path = (self.root / relative).resolve()
            if path == self.root.resolve() or not path.is_relative_to(self.root.resolve()):
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            with self.connection() as db:
                db.execute("DELETE FROM pending_deletions WHERE path=?", (relative,))
