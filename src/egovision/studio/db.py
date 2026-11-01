"""SQLite storage: the single source of truth for videos, workers, events, analyser batches and reports.

One connection per thread (FastAPI runs sync endpoints in a thread pool; analysis runs in its own
thread). WAL mode lets the browser read while the analysis thread writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from egovision.studio.models import EventRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    label TEXT,
    stored_path TEXT NOT NULL,
    preview_path TEXT,
    preview_status TEXT NOT NULL DEFAULT 'not_needed',
    size_bytes INTEGER,
    duration_s REAL, fps REAL, width INTEGER, height INTEGER, n_frames INTEGER, codec TEXT,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    frames_total INTEGER NOT NULL DEFAULT 0, frames_done INTEGER NOT NULL DEFAULT 0,
    batches_total INTEGER NOT NULL DEFAULT 0, batches_done INTEGER NOT NULL DEFAULT 0,
    processed_until_s REAL NOT NULL DEFAULT 0,
    analysis_limit_s REAL,
    sample_every_s REAL, batch_frames INTEGER,
    frames_source TEXT,
    origin TEXT NOT NULL DEFAULT 'upload',  -- upload | forge_import
    analyzer TEXT,                          -- gemini | offline
    model TEXT,
    analysis_note TEXT,                     -- which analyser covered which time range, fallbacks
    hands3d_status TEXT NOT NULL DEFAULT 'none',  -- none | queued | running | ready | failed
    hands3d_progress REAL NOT NULL DEFAULT 0,
    hands3d_error TEXT,
    sop_json TEXT,
    error TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, started_at REAL, finished_at REAL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS workers (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    description TEXT NOT NULL,
    is_camera_wearer INTEGER NOT NULL DEFAULT 0,
    first_seen_ts REAL, last_seen_ts REAL,
    PRIMARY KEY (video_id, worker_id)
);
CREATE TABLE IF NOT EXISTS events (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    action TEXT NOT NULL, object TEXT NOT NULL, tool TEXT,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, duration_s REAL NOT NULL,
    confidence REAL NOT NULL,
    verification_status TEXT NOT NULL, completion_status TEXT NOT NULL,
    frames_seen INTEGER NOT NULL, sop_step INTEGER, evidence TEXT, corrections_json TEXT,
    source TEXT NOT NULL DEFAULT '',
    first_batch INTEGER NOT NULL, last_batch INTEGER NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL,
    PRIMARY KEY (video_id, event_id)
);
CREATE INDEX IF NOT EXISTS events_by_time ON events(video_id, start_ts, seq);
CREATE TABLE IF NOT EXISTS batches (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    batch_index INTEGER NOT NULL,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, n_frames INTEGER NOT NULL,
    status TEXT NOT NULL,
    n_new INTEGER NOT NULL DEFAULT 0, n_updated INTEGER NOT NULL DEFAULT 0, n_rejected INTEGER NOT NULL DEFAULT 0,
    rejected_json TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_s REAL, raw_path TEXT, notes TEXT, error TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (video_id, batch_index)
);
CREATE TABLE IF NOT EXISTS reports (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,          -- csv | json
    table_name TEXT NOT NULL,    -- events | workers | report
    status TEXT NOT NULL,        -- partial | final
    path TEXT NOT NULL,
    n_events INTEGER NOT NULL,
    processed_until_s REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (video_id, kind, table_name, status)
);
"""

VIDEO_JSON = {"sop_json": "sop"}
MIGRATIONS = [("videos", "origin", "TEXT NOT NULL DEFAULT 'upload'"), ("videos", "analyzer", "TEXT"),
              ("videos", "analysis_note", "TEXT"), ("events", "source", "TEXT NOT NULL DEFAULT ''"),
              ("videos", "hands3d_status", "TEXT NOT NULL DEFAULT 'none'"),
              ("videos", "hands3d_progress", "REAL NOT NULL DEFAULT 0"), ("videos", "hands3d_error", "TEXT")]
EVENT_COLS = ["video_id", "event_id", "seq", "worker_id", "action", "object", "tool", "start_ts", "end_ts", "duration_s",
              "confidence", "verification_status", "completion_status", "frames_seen", "sop_step", "evidence",
              "corrections_json", "source", "first_batch", "last_batch", "created_at", "updated_at"]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(SCHEMA)
            for table, col, decl in MIGRATIONS:  # databases created by an earlier version
                if col not in {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.conn = c
        return c

    # ---- videos ---------------------------------------------------------------------------------
    @staticmethod
    def _video(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["sop"] = json.loads(d.pop("sop_json") or "null")
        return d

    def create_video(self, **fields: Any) -> dict[str, Any]:
        now = time.time()
        fields = {"created_at": now, "updated_at": now, "status": "uploaded", **fields}
        if "sop" in fields:
            fields["sop_json"] = json.dumps(fields.pop("sop"))
        cols = ", ".join(fields)
        with self.conn() as c:
            c.execute(f"INSERT INTO videos ({cols}) VALUES ({', '.join('?' * len(fields))})", list(fields.values()))
        return self.get_video(fields["video_id"])  # type: ignore[return-value]

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        return self._video(self.conn().execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone())

    def list_videos(self) -> list[dict[str, Any]]:
        rows = self.conn().execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [v for v in (self._video(r) for r in rows) if v is not None]

    def update_video(self, video_id: str, **fields: Any) -> dict[str, Any] | None:
        if "sop" in fields:
            fields["sop_json"] = json.dumps(fields.pop("sop"))
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE videos SET {sets} WHERE video_id = ?", [*fields.values(), video_id])
        return self.get_video(video_id)

    def add_tokens(self, video_id: str, input_tokens: int, output_tokens: int) -> None:
        with self.conn() as c:
            c.execute("UPDATE videos SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ? "
                      "WHERE video_id = ?", (input_tokens, output_tokens, video_id))

    def delete_video(self, video_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    def clear_analysis(self, video_id: str) -> None:
        """Forget all analysis results of a video (restart); the upload itself stays."""
        with self.conn() as c:
            for t in ("events", "workers", "batches", "reports"):
                c.execute(f"DELETE FROM {t} WHERE video_id = ?", (video_id,))  # noqa: S608 - fixed table names

    def mark_interrupted(self) -> list[str]:
        """On server start: analyses that were running or queued when the server stopped."""
        with self.conn() as c:
            c.execute("UPDATE videos SET hands3d_status = 'failed', hands3d_error = 'server restarted; compute again' "
                      "WHERE hands3d_status IN ('queued', 'running')")
        rows = self.conn().execute("SELECT video_id FROM videos WHERE status IN ('processing', 'queued')").fetchall()
        ids = [r["video_id"] for r in rows]
        for vid in ids:
            self.update_video(vid, status="interrupted", stage="server restarted; press Start to resume")
        return ids

    # ---- workers --------------------------------------------------------------------------------
    def upsert_worker(self, video_id: str, worker_id: str, description: str | None, is_camera_wearer: bool | None,
                      seen_from: float | None = None, seen_to: float | None = None) -> None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM workers WHERE video_id = ? AND worker_id = ?", (video_id, worker_id)).fetchone()
            if row is None:
                c.execute("INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?)",
                          (video_id, worker_id, description or "not described", int(bool(is_camera_wearer)),
                           seen_from, seen_to))
                return
            first = row["first_seen_ts"] if seen_from is None or (row["first_seen_ts"] is not None and
                                                                   row["first_seen_ts"] <= seen_from) else seen_from
            last = row["last_seen_ts"] if seen_to is None or (row["last_seen_ts"] is not None and
                                                               row["last_seen_ts"] >= seen_to) else seen_to
            desc = description if description and (row["description"] == "not described" or len(description) > 3) \
                else row["description"]
            wearer = row["is_camera_wearer"] if is_camera_wearer is None else int(bool(is_camera_wearer) or
                                                                                    row["is_camera_wearer"])
            c.execute("UPDATE workers SET description = ?, is_camera_wearer = ?, first_seen_ts = ?, last_seen_ts = ? "
                      "WHERE video_id = ? AND worker_id = ?", (desc, wearer, first, last, video_id, worker_id))

    def list_workers(self, video_id: str) -> list[dict[str, Any]]:
        rows = self.conn().execute("SELECT * FROM workers WHERE video_id = ? ORDER BY CAST(SUBSTR(worker_id, 2) AS INTEGER)",
                                   (video_id,)).fetchall()
        return [dict(r, is_camera_wearer=bool(r["is_camera_wearer"])) for r in rows]

    # ---- events ---------------------------------------------------------------------------------
    @staticmethod
    def _event(row: sqlite3.Row) -> EventRecord:
        d = dict(row)
        d["corrections"] = json.loads(d.pop("corrections_json") or "[]")
        return EventRecord.model_validate(d)

    def save_event(self, ev: EventRecord) -> None:
        d = ev.model_dump()
        d["corrections_json"] = json.dumps(d.pop("corrections"))
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO events ({', '.join(EVENT_COLS)}) VALUES ({', '.join('?' * len(EVENT_COLS))})",
                      [d[k] for k in EVENT_COLS])

    def list_events(self, video_id: str) -> list[EventRecord]:
        rows = self.conn().execute("SELECT * FROM events WHERE video_id = ? ORDER BY start_ts, seq", (video_id,)).fetchall()
        return [self._event(r) for r in rows]

    def events_where(self, sql: str, args: tuple[Any, ...] = ()) -> list[EventRecord]:
        return [self._event(r) for r in self.conn().execute(f"SELECT * FROM events WHERE {sql}", args).fetchall()]

    def next_seq(self, video_id: str) -> int:
        row = self.conn().execute("SELECT MAX(seq) AS m FROM events WHERE video_id = ?", (video_id,)).fetchone()
        return int(row["m"] or 0) + 1

    # ---- batches --------------------------------------------------------------------------------
    def save_batch(self, video_id: str, batch_index: int, **fields: Any) -> None:
        fields = {"video_id": video_id, "batch_index": batch_index, "created_at": time.time(), **fields}
        if "rejected" in fields:
            fields["rejected_json"] = json.dumps(fields.pop("rejected"))
        cols = ", ".join(fields)
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO batches ({cols}) VALUES ({', '.join('?' * len(fields))})", list(fields.values()))

    def list_batches(self, video_id: str) -> list[dict[str, Any]]:
        rows = self.conn().execute("SELECT * FROM batches WHERE video_id = ? ORDER BY batch_index", (video_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["rejected"] = json.loads(d.pop("rejected_json") or "[]")
            out.append(d)
        return out

    def last_done_batch(self, video_id: str) -> int:
        row = self.conn().execute("SELECT MAX(batch_index) AS m FROM batches WHERE video_id = ? AND status != 'pending'",
                                  (video_id,)).fetchone()
        return -1 if row["m"] is None else int(row["m"])

    # ---- reports --------------------------------------------------------------------------------
    def save_report(self, video_id: str, kind: str, table_name: str, status: str, path: Path, n_events: int,
                    processed_until_s: float, size_bytes: int, sha256: str) -> dict[str, Any]:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (video_id, kind, table_name, status, Path(path).as_posix(), n_events, processed_until_s, size_bytes,
                       sha256, time.time()))
        return self.get_report(video_id, kind, table_name, status)  # type: ignore[return-value]

    def get_report(self, video_id: str, kind: str, table_name: str, status: str) -> dict[str, Any] | None:
        row = self.conn().execute("SELECT * FROM reports WHERE video_id = ? AND kind = ? AND table_name = ? AND status = ?",
                                  (video_id, kind, table_name, status)).fetchone()
        return dict(row) if row else None

    def list_reports(self, video_id: str) -> list[dict[str, Any]]:
        rows = self.conn().execute("SELECT * FROM reports WHERE video_id = ? ORDER BY created_at DESC", (video_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_partial_reports(self, video_id: str) -> list[str]:
        rows = self.conn().execute("SELECT path FROM reports WHERE video_id = ? AND status = 'partial'", (video_id,)).fetchall()
        with self.conn() as c:
            c.execute("DELETE FROM reports WHERE video_id = ? AND status = 'partial'", (video_id,))
        return [r["path"] for r in rows]
