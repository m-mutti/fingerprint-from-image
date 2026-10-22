"""``runs/registry.sqlite``: one row per step execution (train / eval / export / probe).

Columns hold key, step, status, name, config, git commit, metrics, artifact paths,
licence lineage and failure reason. The leaderboard (``training.report``) and the
sweep scheduler (``training.sweep``) read only from here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from training.keys import runs_root

__all__ = ["STATUSES", "Registry"]

STATUSES = ("pending", "running", "done", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS steps (
    key TEXT NOT NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    name TEXT,
    parent_key TEXT,
    config TEXT,
    git_commit TEXT,
    metrics TEXT,
    artifacts TEXT,
    licence TEXT,
    reason TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    PRIMARY KEY (step, key)
);
CREATE INDEX IF NOT EXISTS idx_steps_status ON steps(step, status);
"""

_JSON_COLUMNS = ("config", "metrics", "artifacts", "licence")


class Registry:
    """Thin SQLite wrapper (one connection per call; safe for one writer at a time)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or runs_root() / "registry.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def upsert(self, step: str, key: str, status: str, **fields: Any) -> None:
        """Insert or update a row; JSON columns accept Python objects."""

        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        values = {k: (json.dumps(v, default=str) if k in _JSON_COLUMNS and v is not None else v)
                  for k, v in fields.items()}
        now = time.time()
        with self._connect() as db:
            exists = db.execute("SELECT 1 FROM steps WHERE step=? AND key=?",
                                (step, key)).fetchone()
            if exists:
                sets = ", ".join(f"{k}=?" for k in values)
                db.execute(f"UPDATE steps SET status=?, updated=?{', ' + sets if sets else ''} "
                           "WHERE step=? AND key=?", (status, now, *values.values(), step, key))
            else:
                cols = ", ".join(["key", "step", "status", "created", "updated", *values])
                marks = ", ".join("?" * (5 + len(values)))
                db.execute(f"INSERT INTO steps ({cols}) VALUES ({marks})",
                           (key, step, status, now, now, *values.values()))

    def get(self, step: str, key: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM steps WHERE step=? AND key=?", (step, key)).fetchone()
        return self._decode(row) if row else None

    def rows(self, step: str | None = None, status: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM steps WHERE 1=1", []
        if step:
            query, args = query + " AND step=?", [*args, step]
        if status:
            query, args = query + " AND status=?", [*args, status]
        with self._connect() as db:
            return [self._decode(r) for r in db.execute(query + " ORDER BY created", args)]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        out = dict(row)
        for column in _JSON_COLUMNS:
            if out.get(column):
                out[column] = json.loads(out[column])
        return out
