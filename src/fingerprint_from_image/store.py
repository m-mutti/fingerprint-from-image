"""Versioned SQLite persistence for image-free fingerprint templates."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    APPLICATION_ID,
    FEATURE_CONFIG_FIELDS,
    FEATURE_VERSION,
    SCHEMA_VERSION,
    MatcherConfig,
)
from .digests import _extraction_config_signature, _feature_digest, _utc_now
from .errors import FingerprintDatabaseError
from .extraction import EXTRACTION_VERSION, ExtractionConfig
from .templates import FingerprintTemplate


class FingerprintStore:
    """Versioned SQLite persistence for image-free fingerprint templates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        try:
            self.connection = sqlite3.connect(str(self.path), timeout=30.0)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.connection.execute("PRAGMA secure_delete = ON")
            self.connection.execute("PRAGMA journal_size_limit = 0")
            self._initialize()
        except sqlite3.Error as exc:
            if hasattr(self, "connection"):
                self.connection.close()
            raise FingerprintDatabaseError(
                f"Could not open fingerprint database {self.path}: {exc}"
            ) from exc
        except Exception:
            if hasattr(self, "connection"):
                self.connection.close()
            raise

    def __enter__(self) -> FingerprintStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        application_id = int(self.connection.execute("PRAGMA application_id").fetchone()[0])
        existing_objects = {
            (str(row[0]), str(row[1]))
            for row in self.connection.execute(
                "SELECT type, name FROM sqlite_master"
            )
            if not str(row[1]).startswith("sqlite_")
        }
        if application_id not in (0, APPLICATION_ID):
            raise FingerprintDatabaseError(
                f"{self.path} belongs to a different SQLite application"
            )
        existing_tables = {
            name for object_type, name in existing_objects if object_type == "table"
        }
        if existing_objects:
            required = {"fingerprint_metadata", "identities", "templates"}
            if application_id != APPLICATION_ID or not required.issubset(existing_tables):
                raise FingerprintDatabaseError(
                    f"Refusing to modify non-fingerprint SQLite database: {self.path}"
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in self.connection.execute(
                    "SELECT key, value FROM fingerprint_metadata"
                )
            }
            try:
                schema = int(metadata.get("schema_version", "-1"))
            except ValueError as exc:
                raise FingerprintDatabaseError("Invalid database schema metadata") from exc
            if schema != SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    f"Database schema version {schema} is not supported; "
                    f"expected {SCHEMA_VERSION}"
                )
            if metadata.get("feature_version") != FEATURE_VERSION:
                raise FingerprintDatabaseError(
                    "Database contains an incompatible feature representation"
                )
            if metadata.get("extraction_version") != EXTRACTION_VERSION:
                raise FingerprintDatabaseError(
                    "Database contains templates from an incompatible hand extractor"
                )
            user_version = int(
                self.connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if user_version != SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    "Database schema metadata and SQLite user_version disagree"
                )
            return

        if application_id != 0:
            raise FingerprintDatabaseError(
                f"{self.path} is not an empty SQLite database"
            )
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                CREATE TABLE fingerprint_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE identities (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL REFERENCES identities(name) ON DELETE CASCADE,
                    finger TEXT NOT NULL,
                    source_path TEXT,
                    source_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    height INTEGER NOT NULL CHECK(height > 0),
                    width INTEGER NOT NULL CHECK(width > 0),
                    quality REAL NOT NULL CHECK(quality BETWEEN 0 AND 1),
                    roi_fraction REAL NOT NULL CHECK(roi_fraction BETWEEN 0 AND 1),
                    keypoint_count INTEGER NOT NULL CHECK(keypoint_count > 0),
                    descriptor_size INTEGER NOT NULL CHECK(descriptor_size = 128),
                    keypoints BLOB NOT NULL,
                    descriptors BLOB NOT NULL,
                    feature_digest TEXT NOT NULL,
                    UNIQUE(identity, finger, source_digest)
                )
                """
            )
            for name, columns, unique in (
                ("idx_templates_identity", "identity", False),
                ("idx_templates_identity_finger", "identity, finger", False),
                ("uq_templates_source_digest", "source_digest", True),
                ("uq_templates_feature_digest", "feature_digest", True),
            ):
                self.connection.execute(
                    f"CREATE {'UNIQUE ' if unique else ''}INDEX "
                    f"{name} ON templates({columns})"
                )
            self.connection.executemany(
                "INSERT INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("feature_version", FEATURE_VERSION),
                    ("extraction_version", EXTRACTION_VERSION),
                    ("created_at", _utc_now()),
                ),
            )
            self.connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _validate_label(label: str, kind: str) -> str:
        cleaned = label.strip()
        if not cleaned:
            raise ValueError(f"{kind} cannot be empty")
        if len(cleaned) > 255:
            raise ValueError(f"{kind} cannot exceed 255 characters")
        return cleaned

    def add_template(
        self,
        identity: str,
        finger: str,
        template: FingerprintTemplate,
        *,
        source_digest: str,
        source_path: str | None = None,
        _within_transaction: bool = False,
    ) -> int | None:
        """Add a template, returning its id or ``None`` for a duplicate."""

        bound_keys = int(
            self.connection.execute(
                """
                SELECT COUNT(*)
                FROM fingerprint_metadata
                WHERE key IN ('feature_config', 'extraction_config')
                """
            ).fetchone()[0]
        )
        if bound_keys != 2:
            raise FingerprintDatabaseError(
                "Database configuration is not bound; enroll through "
                "FingerprintMatcher or bind both configurations first"
            )
        identity = self._validate_label(identity, "identity")
        finger = self._validate_label(finger, "finger")
        if template.feature_version != FEATURE_VERSION:
            raise FingerprintDatabaseError("Unsupported template feature version")
        points = np.ascontiguousarray(template.keypoints, dtype="<f4")
        descriptors = np.ascontiguousarray(template.descriptors, dtype="<f4")
        points_blob = points.tobytes(order="C")
        descriptors_blob = descriptors.tobytes(order="C")
        feature_digest = _feature_digest(points_blob, descriptors_blob)
        try:
            transaction = nullcontext() if _within_transaction else self.connection
            with transaction:
                existing = self.connection.execute(
                    """
                    SELECT identity, finger
                    FROM templates
                    WHERE source_digest = ? OR feature_digest = ?
                    LIMIT 1
                    """,
                    (source_digest, feature_digest),
                ).fetchone()
                if existing is not None:
                    enrolled_as = (str(existing["identity"]), str(existing["finger"]))
                    if enrolled_as == (identity, finger):
                        return None
                    raise FingerprintDatabaseError(
                        "The same source or feature template is already enrolled as "
                        f"{enrolled_as[0]}/{enrolled_as[1]}; refusing conflicting "
                        f"enrollment as {identity}/{finger}"
                    )
                self.connection.execute(
                    "INSERT OR IGNORE INTO identities(name, created_at) VALUES (?, ?)",
                    (identity, _utc_now()),
                )
                try:
                    cursor = self.connection.execute(
                        """
                        INSERT INTO templates(
                            identity, finger, source_path, source_digest, created_at,
                            feature_version, height, width, quality, roi_fraction,
                            keypoint_count, descriptor_size, keypoints, descriptors,
                            feature_digest
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identity,
                            finger,
                            source_path,
                            source_digest,
                            _utc_now(),
                            FEATURE_VERSION,
                            int(template.image_shape[0]),
                            int(template.image_shape[1]),
                            float(template.quality),
                            float(template.roi_fraction),
                            int(points.shape[0]),
                            int(descriptors.shape[1]),
                            sqlite3.Binary(points_blob),
                            sqlite3.Binary(descriptors_blob),
                            feature_digest,
                        ),
                    )
                except sqlite3.IntegrityError:
                    raced = self.connection.execute(
                        """
                        SELECT identity, finger
                        FROM templates
                        WHERE source_digest = ? OR feature_digest = ?
                        LIMIT 1
                        """,
                        (source_digest, feature_digest),
                    ).fetchone()
                    if raced is not None and (
                        str(raced["identity"]),
                        str(raced["finger"]),
                    ) == (identity, finger):
                        return None
                    if raced is not None:
                        raise FingerprintDatabaseError(
                            "The same source or feature template was concurrently "
                            f"enrolled as {raced['identity']}/{raced['finger']}; "
                            f"refusing conflicting enrollment as {identity}/{finger}"
                        ) from None
                    raise
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise FingerprintDatabaseError(f"Could not store template: {exc}") from exc

    def _decode_template(self, row: sqlite3.Row) -> FingerprintTemplate:
        count = int(row["keypoint_count"])
        descriptor_size = int(row["descriptor_size"])
        if count < 1 or count > 1_000_000 or descriptor_size != 128:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has invalid array dimensions"
            )
        points_blob = bytes(row["keypoints"])
        descriptors_blob = bytes(row["descriptors"])
        if len(points_blob) != count * 2 * 4:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has a corrupt keypoint array"
            )
        if len(descriptors_blob) != count * descriptor_size * 4:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has a corrupt descriptor array"
            )
        if _feature_digest(points_blob, descriptors_blob) != row["feature_digest"]:
            raise FingerprintDatabaseError(
                f"Template {row['id']} failed its feature checksum"
            )
        points = np.frombuffer(points_blob, dtype="<f4").reshape(count, 2).copy()
        descriptors = (
            np.frombuffer(descriptors_blob, dtype="<f4")
            .reshape(count, descriptor_size)
            .copy()
        )
        return FingerprintTemplate(
            keypoints=points,
            descriptors=descriptors,
            image_shape=(int(row["height"]), int(row["width"])),
            quality=float(row["quality"]),
            roi_fraction=float(row["roi_fraction"]),
            identity=str(row["identity"]),
            finger=str(row["finger"]),
            template_id=int(row["id"]),
            source=str(row["source_path"]) if row["source_path"] else None,
            feature_version=str(row["feature_version"]),
        )

    def templates(self, identity: str | None = None) -> Iterator[FingerprintTemplate]:
        if identity is None:
            cursor = self.connection.execute("SELECT * FROM templates ORDER BY id")
        else:
            cursor = self.connection.execute(
                "SELECT * FROM templates WHERE identity = ? ORDER BY id", (identity,)
            )
        for row in cursor:
            yield self._decode_template(row)

    def identities(self) -> list[str]:
        return [
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM identities WHERE EXISTS "
                "(SELECT 1 FROM templates WHERE templates.identity = identities.name) "
                "ORDER BY name"
            )
        ]

    def has_identity(self, identity: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM templates WHERE identity = ? LIMIT 1", (identity,)
            ).fetchone()
            is not None
        )

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM templates")
            self.connection.execute("DELETE FROM identities")
        # Rebuild the file so freed template pages are not retained in its
        # freelist. Filesystem snapshots and storage media may still retain
        # historical blocks, so this is not a forensic-erasure guarantee.
        self.connection.execute("VACUUM")

    def stats(self) -> dict[str, Any]:
        template_count = int(
            self.connection.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
        )
        identity_count = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT identity) FROM templates"
            ).fetchone()[0]
        )
        per_identity = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT identity, COUNT(*) FROM templates "
                "GROUP BY identity ORDER BY identity"
            )
        }
        per_finger = {
            f"{row[0]}/{row[1]}": int(row[2])
            for row in self.connection.execute(
                "SELECT identity, finger, COUNT(*) FROM templates "
                "GROUP BY identity, finger ORDER BY identity, finger"
            )
        }
        return {
            "database": str(self.path.resolve()),
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "identities": identity_count,
            "templates": template_count,
            "templates_per_identity": per_identity,
            "templates_per_finger": per_finger,
        }

    def bind_feature_config(self, config: MatcherConfig) -> None:
        """Record or validate settings that affect persisted feature vectors."""

        config_dict = asdict(config)
        signature = json.dumps(
            {name: config_dict[name] for name in FEATURE_CONFIG_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                ("feature_config", signature),
            )
        row = self.connection.execute(
            "SELECT value FROM fingerprint_metadata WHERE key = 'feature_config'"
        ).fetchone()
        if row is None:
            raise FingerprintDatabaseError("Database feature configuration is missing")
        if str(row[0]) != signature:
            raise FingerprintDatabaseError(
                "Matcher feature settings differ from this database; use the "
                "original settings or rebuild the database"
            )

    def bind_extraction_config(self, config: ExtractionConfig) -> None:
        """Record or validate the hand-crop settings used by this database."""

        signature = _extraction_config_signature(config)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                ("extraction_config", signature),
            )
        row = self.connection.execute(
            "SELECT value FROM fingerprint_metadata WHERE key = 'extraction_config'"
        ).fetchone()
        if row is None:
            raise FingerprintDatabaseError("Database extraction configuration is missing")
        if str(row[0]) != signature:
            raise FingerprintDatabaseError(
                "Hand extraction settings differ from this database; use the "
                "original settings or rebuild the database"
            )
