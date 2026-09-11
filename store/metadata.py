"""SQLite metadata store: chunks, versions, trace log."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,          -- content hash
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',  -- active | superseded | duplicate
    superseded_by TEXT REFERENCES chunks(id),
    token_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    hops INTEGER NOT NULL,
    detail TEXT NOT NULL            -- JSON
);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
"""


def chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MetadataStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(SCHEMA)
        self.db.commit()

    def add_chunk(self, text: str, source: str = "",
                  token_count: int = 0) -> tuple[str, bool]:
        """Insert chunk; returns (id, is_new). Idempotent on content hash."""
        cid = chunk_id(text)
        cur = self.db.execute("SELECT id FROM chunks WHERE id = ?", (cid,))
        if cur.fetchone():
            return cid, False
        self.db.execute(
            "INSERT INTO chunks (id, text, source, created_at, token_count)"
            " VALUES (?, ?, ?, ?, ?)",
            (cid, text, source, time.time(), token_count),
        )
        self.db.commit()
        return cid, True

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        self.db.execute(
            "UPDATE chunks SET status='superseded', superseded_by=? WHERE id=?",
            (new_id, old_id),
        )
        self.db.commit()

    def get_chunk(self, cid: str) -> dict | None:
        cur = self.db.execute("SELECT * FROM chunks WHERE id = ?", (cid,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def active_chunks(self) -> list[dict]:
        cur = self.db.execute("SELECT * FROM chunks WHERE status='active'")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def log_trace(self, question: str, hops: int, detail_json: str) -> None:
        self.db.execute(
            "INSERT INTO traces (ts, question, hops, detail) VALUES (?, ?, ?, ?)",
            (time.time(), question, hops, detail_json),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
