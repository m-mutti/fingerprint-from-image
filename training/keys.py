"""Content-addressed pipeline steps: keys, ``done.json`` records and skip logic.

A step's key is the SHA-256 of a canonical JSON document of its exact inputs.
Its outputs live in ``runs/<step>/<key[:12]>/`` next to ``done.json``, which lists
the inputs, output files with their SHA-256, timings and environment. A step whose
``done.json`` exists and whose outputs still hash correctly is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PIPELINE_VERSION",
    "StepDir",
    "canonical_json",
    "code_version",
    "content_key",
    "source_digest",
    "file_sha256",
    "runs_root",
]

PIPELINE_VERSION = "pipeline-v1"


def runs_root() -> Path:
    """``$FFI_RUNS`` or ``<FFI_ROOT>/runs`` (``FFI_ROOT`` defaults to the repository root)."""

    if os.environ.get("FFI_RUNS"):
        return Path(os.environ["FFI_RUNS"])
    root = Path(os.environ.get("FFI_ROOT", Path(__file__).resolve().parents[1]))
    return root / "runs"


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, tuples as lists, paths as POSIX."""

    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default,
                      ensure_ascii=True)


def content_key(step: str, inputs: Mapping[str, Any]) -> str:
    """SHA-256 hex key of ``{"step", "pipeline", **inputs}``."""

    payload = {"step": step, "pipeline": PIPELINE_VERSION, "inputs": dict(inputs)}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(files: Iterable[Path], root: Path) -> str:
    """SHA-256 over (relative POSIX path, content hash) of ``files`` (sorted)."""

    entries = sorted((f.relative_to(root).as_posix(), file_sha256(f)) for f in files)
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def code_version(repo: str | Path | None = None,
                 patterns: Iterable[str] = ("training/*.py", "src/**/*.py")) -> dict:
    """Content hash of the Python sources that affect training (tracked *and* untracked).

    Files come from ``git ls-files -co --exclude-standard`` restricted to ``patterns``;
    docs/YAML edits do not change the key, uncommitted ``.py`` edits do. The commit id
    is recorded for provenance but is not part of the hash.
    """

    root = Path(repo) if repo else Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  check=False).stdout
        except OSError:
            return ""

    listed = git("ls-files", "-co", "--exclude-standard", "--", *patterns).splitlines()
    files = [root / name for name in listed if name.endswith(".py") and (root / name).is_file()]
    if not files:  # not a git checkout: fall back to globbing
        files = [f for pattern in patterns for f in root.glob(pattern) if f.is_file()]
    status = git("status", "--porcelain", "--", *patterns)
    return {"py_sha256": source_digest(files, root), "files": len(files),
            "commit": git("rev-parse", "HEAD").strip() or "unknown",
            "dirty": bool(status.strip())}


@dataclass(frozen=True)
class StepDir:
    """``runs/<step>/<key[:12]>/`` with ``done.json`` bookkeeping."""

    step: str
    key: str
    root: Path

    @classmethod
    def for_inputs(cls, step: str, inputs: Mapping[str, Any], root: Path | None = None) -> StepDir:
        return cls(step, content_key(step, inputs), root or runs_root())

    @property
    def path(self) -> Path:
        return self.root / self.step / self.key[:12]

    @property
    def done_file(self) -> Path:
        return self.path / "done.json"

    def record(self) -> dict | None:
        if not self.done_file.is_file():
            return None
        return json.loads(self.done_file.read_text(encoding="utf-8"))

    def is_done(self, verify: bool = True) -> bool:
        """True when ``done.json`` exists, matches this key and every output still hashes."""

        record = self.record()
        if record is None or record.get("key") != self.key:
            return False
        if not verify:
            return True
        for rel, digest in record.get("outputs", {}).items():
            target = self.path / rel
            if not target.is_file() or file_sha256(target) != digest:
                return False
        return True

    def mark_done(self, inputs: Mapping[str, Any], outputs: Iterable[str],
                  extra: Mapping[str, Any] | None = None, started: float | None = None) -> dict:
        """Hash ``outputs`` (paths relative to the step dir) and write ``done.json`` atomically."""

        record = {
            "step": self.step, "key": self.key, "pipeline": PIPELINE_VERSION,
            "inputs": dict(inputs),
            "outputs": {rel: file_sha256(self.path / rel) for rel in outputs},
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - started, 1) if started else None,
            "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
            **dict(extra or {}),
        }
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.done_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.done_file)
        return record

    def invalidate(self) -> None:
        self.done_file.unlink(missing_ok=True)
