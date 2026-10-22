"""Decoded-pixel content hashes for every image the pipeline touches (audit C2).

A file's hash is SHA-256 over its decoded pixels (mode, size, raw bytes), so re-encoded
or copied files (e.g. FVC2004 DB_B copies under ``data/fvc_full``) still match. Hashes
are cached in ``runs/hashcache/pixels.json`` keyed by ``path|size|mtime_ns``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from training.keys import runs_root

__all__ = ["digest", "pixel_hash", "pixel_hashes"]


def pixel_hash(path: str | Path) -> str:
    """SHA-256 of the decoded pixels (first frame) including mode and size."""

    with Image.open(path) as image:
        image.load()
        header = f"{image.mode}|{image.size[0]}x{image.size[1]}|".encode()
        return hashlib.sha256(header + image.tobytes()).hexdigest()


def _stat_key(path: str) -> str:
    stat = Path(path).stat()
    return f"{Path(path).resolve().as_posix().lower()}|{stat.st_size}|{stat.st_mtime_ns}"


def pixel_hashes(paths: Iterable[str], *, cache: Path | None = None,
                 workers: int = 12) -> dict[str, str]:
    """``{path: pixel hash}`` for ``paths``, computing only what the cache lacks."""

    cache = cache or runs_root() / "hashcache" / "pixels.json"
    known: dict[str, str] = (json.loads(cache.read_text(encoding="utf-8"))
                             if cache.is_file() else {})
    unique = sorted(set(paths))
    keys = {p: _stat_key(p) for p in unique}
    missing = [p for p in unique if keys[p] not in known]
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for path, value in zip(missing, pool.map(pixel_hash, missing, chunksize=64),
                                   strict=True):
                known[keys[path]] = value
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(known), encoding="utf-8")
        tmp.replace(cache)
    return {p: known[keys[p]] for p in unique}


def digest(items: Iterable[object]) -> str:
    """Order-independent SHA-256 over a collection of JSON-serialisable items."""

    payload = json.dumps(sorted(json.dumps(i, sort_keys=True) for i in items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
