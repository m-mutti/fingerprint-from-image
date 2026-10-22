"""``fingerprint-match`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import MatcherConfig
from ..errors import FingerprintDatabaseError, FingerprintMatchingError
from ..extraction import FingerprintExtractionError
from ..matcher import FingerprintMatcher
from ..store import FingerprintStore


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db", default="fingerprints.db", help="SQLite template database"
    )


def _add_image_kind_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hand", action="store_true", help="input is a complete hand image")
    group.add_argument(
        "--cropped", action="store_true", help="input is one cropped fingerprint"
    )


def _kind_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "hand", False):
        return "hand"
    if getattr(args, "cropped", False):
        return "cropped"
    # Matching an already-extracted print is the safe default. Hand extraction
    # is intentionally explicit because a connected hand silhouette can look
    # like one contour and cannot be reliably inferred from crop count alone.
    return "cropped"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll and match fingerprints using RootSIFT and affine RANSAC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="incrementally enroll an identity-organized dataset"
    )
    build.add_argument("dataset", help="dataset root (identity/finger/sample or identity/capture)")
    build.add_argument(
        "--replace", action="store_true", help="clear enrolled identities before building"
    )
    _add_database_argument(build)

    enroll = subparsers.add_parser("enroll", help="incrementally add images for an identity")
    enroll.add_argument("identity")
    enroll.add_argument("images", nargs="+")
    enroll.add_argument("--finger", default="unknown", help="label for cropped samples")
    _add_image_kind_arguments(enroll)
    _add_database_argument(enroll)

    for command in ("identify", "match"):
        identify = subparsers.add_parser(
            command,
            help="search an image against the database"
            + (" (alias of identify)" if command == "match" else ""),
        )
        identify.add_argument("image")
        identify.add_argument("--top-k", type=int, default=5)
        identify.add_argument("--threshold", type=float)
        _add_image_kind_arguments(identify)
        _add_database_argument(identify)

    verify = subparsers.add_parser("verify", help="test an image against a claimed identity")
    verify.add_argument("identity")
    verify.add_argument("image")
    verify.add_argument("--threshold", type=float)
    _add_image_kind_arguments(verify)
    _add_database_argument(verify)

    stats = subparsers.add_parser("stats", help="show database counts and versions")
    _add_database_argument(stats)
    return parser


def _json_dump(value: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), file=stream)


def _restore_staged_database(source: Path, destination: Path) -> None:
    """Transactionally copy a staged SQLite image into an existing DB file.

    SQLite keeps a write transaction open on the destination during backup and
    rolls it back if copying does not finish. Writing the existing file also
    preserves its filesystem identity and its POSIX mode or Windows DACL.
    """

    source_connection = sqlite3.connect(str(source), timeout=30.0)
    destination_connection = sqlite3.connect(str(destination), timeout=30.0)
    try:
        destination_connection.execute("PRAGMA busy_timeout = 30000")
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = MatcherConfig()
    if getattr(args, "threshold", None) is not None:
        threshold = float(args.threshold)
        if not 0.0 < threshold <= 1.0:
            _json_dump({"ok": False, "error": "threshold must be in (0, 1]"}, stream=sys.stderr)
            return 2
        config = replace(
            config,
            finger_match_threshold=threshold,
            identity_accept_threshold=threshold,
        )

    try:
        if args.command in {"identify", "match", "verify", "stats"}:
            database_path = Path(args.db)
            if not database_path.is_file():
                raise FingerprintDatabaseError(
                    f"Fingerprint database does not exist: {database_path}"
                )
        if args.command == "build" and args.replace and Path(args.db).is_file():
            database_path = Path(args.db)
            # Refuse to overwrite an unrelated or unsupported file. Keeping the
            # existing connection closed during the backup also avoids Windows
            # sharing violations.
            with FingerprintStore(database_path):
                pass
            staging_path = database_path.with_name(
                f".{database_path.name}.{uuid.uuid4().hex}.staging"
            )
            try:
                with FingerprintMatcher(staging_path, config=config) as matcher:
                    report = matcher.enroll_directory(args.dataset)
                    payload = {
                        "ok": report.failed == 0,
                        "command": "build",
                        "dataset": str(Path(args.dataset).resolve()),
                        "enrollment": report.to_dict(),
                        "stats": matcher.stats(),
                        "replacement_committed": False,
                    }
                if report.failed:
                    _json_dump(payload)
                    return 1
                _restore_staged_database(staging_path, database_path)
                with FingerprintStore(database_path):
                    pass
                payload["stats"]["database"] = str(database_path.resolve())
                payload["replacement_committed"] = True
                _json_dump(payload)
                return 0
            finally:
                # Only exact paths derived from the generated staging name are
                # eligible for cleanup; no directory or wildcard is involved.
                for temporary in (
                    staging_path,
                    Path(f"{staging_path}-journal"),
                    Path(f"{staging_path}-wal"),
                    Path(f"{staging_path}-shm"),
                ):
                    temporary.unlink(missing_ok=True)
        with FingerprintMatcher(args.db, config=config) as matcher:
            if args.command == "build":
                report = matcher.enroll_directory(args.dataset)
                payload = {
                    "ok": report.failed == 0,
                    "command": "build",
                    "dataset": str(Path(args.dataset).resolve()),
                    "enrollment": report.to_dict(),
                    "stats": matcher.stats(),
                }
                _json_dump(payload)
                return 0 if report.failed == 0 else 1
            if args.command == "enroll":
                report = matcher.enroll(
                    args.identity,
                    [Path(image) for image in args.images],
                    finger=args.finger,
                    kind=_kind_from_args(args),
                )
                _json_dump(
                    {
                        "ok": report.failed == 0,
                        "command": "enroll",
                        "identity": args.identity,
                        "enrollment": report.to_dict(),
                    }
                )
                return 0 if report.failed == 0 else 1
            if args.command in {"identify", "match"}:
                results = matcher.identify(
                    args.image, top_k=args.top_k, kind=_kind_from_args(args)
                )
                _json_dump(
                    {
                        "ok": True,
                        "command": args.command,
                        "query": str(Path(args.image).resolve()),
                        "matches": [item.to_dict() for item in results],
                    }
                )
                return 0
            if args.command == "verify":
                result = matcher.verify(
                    args.identity, args.image, kind=_kind_from_args(args)
                )
                _json_dump(
                    {
                        "ok": True,
                        "command": "verify",
                        "query": str(Path(args.image).resolve()),
                        "claimed_identity": args.identity,
                        "match": result.to_dict(),
                    }
                )
                return 0 if result.accepted else 1
            if args.command == "stats":
                _json_dump({"ok": True, "command": "stats", "stats": matcher.stats()})
                return 0
    except (FingerprintMatchingError, FingerprintExtractionError, ValueError, OSError) as exc:
        _json_dump({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
