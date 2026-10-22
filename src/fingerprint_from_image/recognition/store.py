"""SQLite persistence for fixed-length fingerprint embeddings (schema v4).

Schema v4 stores one L2-normalised float32 vector per enrolled fingertip
instead of RootSIFT keypoints. A database is bound to exactly one embedding
model version and dimension; v3 (RootSIFT) databases and databases created by
a different model are refused rather than mixed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import APPLICATION_ID
from ..errors import FingerprintDatabaseError

__all__ = ["EMBEDDING_SCHEMA_VERSION", "EmbeddingStore", "GalleryMatrix"]

EMBEDDING_SCHEMA_VERSION = 4
_LEGACY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class GalleryMatrix:
    """A read-only snapshot of every enrolled embedding, aligned by row."""

    ids: np.ndarray            # (N,) int64
    identities: tuple[str, ...]
    fingers: tuple[str, ...]
    vectors: np.ndarray        # (N, D) float32, unit norm

    def __len__(self) -> int:
        return int(self.ids.shape[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vector_digest(model_version: str, blob: bytes) -> str:
    digest = hashlib.sha256(model_version.encode("utf-8"))
    digest.update(blob)
    return digest.hexdigest()


def _clean_label(label: str, kind: str) -> str:
    cleaned = label.strip()
    if not cleaned:
        raise ValueError(f"{kind} cannot be empty")
    if len(cleaned) > 255:
        raise ValueError(f"{kind} cannot exceed 255 characters")
    return cleaned


class EmbeddingStore:
    """Versioned SQLite store bound to one embedding model."""

    def __init__(self, path: str | Path, *, model_version: str, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.path = Path(path)
        self.model_version = model_version
        self.dim = int(dim)
        self._gallery: GalleryMatrix | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
            except FileExistsError:
                pass
        try:
            self.connection = sqlite3.connect(str(self.path), timeout=30.0)
            self.connection.row_factory = sqlite3.Row
            for pragma in (
                "foreign_keys = ON",
                "busy_timeout = 30000",
                "secure_delete = ON",
                "journal_size_limit = 0",
            ):
                self.connection.execute(f"PRAGMA {pragma}")
            self._initialize()
        except sqlite3.Error as exc:
            self.connection.close()
            raise FingerprintDatabaseError(
                f"Could not open fingerprint database {self.path}: {exc}"
            ) from exc
        except Exception:
            if hasattr(self, "connection"):
                self.connection.close()
            raise

    def __enter__(self) -> EmbeddingStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        application_id = int(self.connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        has_objects = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        if application_id not in (0, APPLICATION_ID):
            raise FingerprintDatabaseError(f"{self.path} belongs to a different application")
        if has_objects:
            if application_id != APPLICATION_ID:
                raise FingerprintDatabaseError(
                    f"Refusing to modify non-fingerprint SQLite database: {self.path}"
                )
            if user_version == _LEGACY_SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    f"{self.path} is a v3 RootSIFT database; enroll again into a new "
                    "database to use the learned (schema v4) pipeline"
                )
            if user_version != EMBEDDING_SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    f"Database schema version {user_version} is not supported; "
                    f"expected {EMBEDDING_SCHEMA_VERSION}"
                )
            self._check_binding()
            return
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE identities (name TEXT PRIMARY KEY, created_at TEXT NOT NULL);
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL REFERENCES identities(name) ON DELETE CASCADE,
                    finger TEXT NOT NULL,
                    source_path TEXT,
                    source_digest TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    quality REAL NOT NULL CHECK(quality BETWEEN 0 AND 1),
                    vector BLOB NOT NULL,
                    vector_digest TEXT NOT NULL
                );
                CREATE INDEX idx_embeddings_identity ON embeddings(identity, finger);
                """
            )
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(EMBEDDING_SCHEMA_VERSION)),
                    ("model_version", self.model_version),
                    ("dim", str(self.dim)),
                    ("created_at", _utc_now()),
                ),
            )
            self.connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self.connection.execute(f"PRAGMA user_version = {EMBEDDING_SCHEMA_VERSION}")

    def _check_binding(self) -> None:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("model_version") != self.model_version:
            raise FingerprintDatabaseError(
                f"Database was built with model {metadata.get('model_version')!r}, not "
                f"{self.model_version!r}; rebuild it to change models"
            )
        if metadata.get("dim") != str(self.dim):
            raise FingerprintDatabaseError("Database embedding dimension does not match")

    def add(
        self,
        identity: str,
        finger: str,
        vector: np.ndarray,
        *,
        quality: float,
        source_digest: str,
        source_path: str | None = None,
    ) -> int | None:
        """Store one embedding; return its id, or ``None`` for an exact duplicate."""

        identity = _clean_label(identity, "identity")
        finger = _clean_label(finger, "finger")
        array = np.ascontiguousarray(np.asarray(vector, dtype="<f4").reshape(-1))
        if array.shape[0] != self.dim or not np.isfinite(array).all():
            raise ValueError(f"Embedding must be {self.dim} finite values")
        norm = float(np.linalg.norm(array))
        if not 0.99 <= norm <= 1.01:
            raise ValueError("Embedding must be L2-normalised")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        blob = array.tobytes()
        try:
            with self.connection:
                existing = self.connection.execute(
                    "SELECT identity, finger FROM embeddings WHERE source_digest = ?",
                    (source_digest,),
                ).fetchone()
                if existing is not None:
                    if (existing["identity"], existing["finger"]) == (identity, finger):
                        return None
                    raise FingerprintDatabaseError(
                        f"Source already enrolled as {existing['identity']}/"
                        f"{existing['finger']}; refusing enrollment as {identity}/{finger}"
                    )
                self.connection.execute(
                    "INSERT OR IGNORE INTO identities(name, created_at) VALUES (?, ?)",
                    (identity, _utc_now()),
                )
                cursor = self.connection.execute(
                    """
                    INSERT INTO embeddings(identity, finger, source_path, source_digest,
                        created_at, quality, vector, vector_digest)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity,
                        finger,
                        source_path,
                        source_digest,
                        _utc_now(),
                        float(quality),
                        sqlite3.Binary(blob),
                        _vector_digest(self.model_version, blob),
                    ),
                )
        except sqlite3.Error as exc:
            raise FingerprintDatabaseError(f"Could not store embedding: {exc}") from exc
        self._gallery = None
        return int(cursor.lastrowid)

    def gallery(self) -> GalleryMatrix:
        """Return (and cache) every embedding as one matrix for vectorised scoring."""

        if self._gallery is not None:
            return self._gallery
        rows = self.connection.execute(
            "SELECT id, identity, finger, vector, vector_digest FROM embeddings ORDER BY id"
        ).fetchall()
        vectors = np.empty((len(rows), self.dim), dtype=np.float32)
        for index, row in enumerate(rows):
            blob = bytes(row["vector"])
            if len(blob) != self.dim * 4 or _vector_digest(
                self.model_version, blob
            ) != row["vector_digest"]:
                raise FingerprintDatabaseError(f"Embedding {row['id']} failed its checksum")
            vectors[index] = np.frombuffer(blob, dtype="<f4")
        self._gallery = GalleryMatrix(
            ids=np.asarray([int(row["id"]) for row in rows], dtype=np.int64),
            identities=tuple(str(row["identity"]) for row in rows),
            fingers=tuple(str(row["finger"]) for row in rows),
            vectors=vectors,
        )
        return self._gallery

    def has_identity(self, identity: str) -> bool:
        query = "SELECT 1 FROM embeddings WHERE identity = ? LIMIT 1"
        return self.connection.execute(query, (identity,)).fetchone() is not None

    def remove_identity(self, identity: str) -> int:
        """Delete an identity and all of its embeddings; return rows removed."""

        with self.connection:
            removed = self.connection.execute(
                "DELETE FROM embeddings WHERE identity = ?", (identity,)
            ).rowcount
            self.connection.execute("DELETE FROM identities WHERE name = ?", (identity,))
        self._gallery = None
        return int(removed)

    def stats(self) -> dict[str, Any]:
        per_identity = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT identity, COUNT(*) FROM embeddings GROUP BY identity ORDER BY identity"
            )
        }
        return {
            "database": str(self.path.resolve()),
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "model_version": self.model_version,
            "dim": self.dim,
            "identities": len(per_identity),
            "embeddings": sum(per_identity.values()),
            "embeddings_per_identity": per_identity,
        }
