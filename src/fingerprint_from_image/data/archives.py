"""Resumable downloads and defensive archive extraction (standard library only).

Archives are treated as untrusted: member paths are checked for absolute
paths, drive letters and ``..``; symlinks are refused; only whitelisted file
extensions are written (images and documentation, never executables); and the
total uncompressed size, member count and per-member compression ratio are
bounded to stop zip bombs.  7-Zip archives are listed and extracted with the
``7z`` command-line tool after the same checks on its ``-slt`` listing.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

__all__ = [
    "DOC_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "ExtractLimits",
    "UnsafeArchiveError",
    "fetch_resumable",
    "file_digest",
    "is_safe_member",
    "list_7z",
    "safe_extract_7z",
    "safe_extract_zip",
]

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".wsq", ".pgm"})
DOC_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".csv", ".json"})
_SEVEN_ZIP_CANDIDATES = ("7z", "7za", r"C:\Program Files\7-Zip\7z.exe")


class UnsafeArchiveError(RuntimeError):
    """An archive member failed a safety check."""


@dataclass(frozen=True)
class ExtractLimits:
    """Bounds applied before anything is written."""

    max_total_bytes: int = 120 * 1024**3
    max_members: int = 2_000_000
    max_ratio: float = 200.0  # uncompressed / compressed, per member
    allowed_extensions: frozenset[str] = IMAGE_EXTENSIONS | DOC_EXTENSIONS


DEFAULT_LIMITS = ExtractLimits()


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, streamed in 1 MiB chunks."""

    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_resumable(url: str, destination: str | Path, *, sha256: str | None = None,
                    md5: str | None = None, size: int | None = None,
                    timeout: float = 120.0) -> Path:
    """Download ``url`` to ``destination`` with HTTP Range resume; verify if possible.

    A finished file whose digest matches is left untouched (idempotent).  A
    partial download lives at ``<destination>.part`` and is resumed.
    """

    target = Path(destination)
    if target.is_file() and _verified(target, sha256, md5, size):
        logger.info("ok      %s", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.is_file() and not partial.exists():
        target.replace(partial)  # unverified full file: try to resume / re-check it
    offset = partial.stat().st_size if partial.exists() else 0
    if size is None or offset < size:
        request = urllib.request.Request(url, headers={"User-Agent": "fingerprint-data/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        logger.info("fetch   %s (from byte %d)", url, offset)
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            mode = "ab" if offset and response.status == 206 else "wb"
            with partial.open(mode) as sink:
                shutil.copyfileobj(response, sink, length=1 << 20)
    if not _verified(partial, sha256, md5, size):
        raise RuntimeError(f"{url}: downloaded file failed verification ({partial})")
    partial.replace(target)
    return target


def _verified(path: Path, sha256: str | None, md5: str | None, size: int | None) -> bool:
    if size is not None and path.stat().st_size != size:
        return False
    if sha256 is not None:
        return file_digest(path, "sha256") == sha256.lower()
    if md5 is not None:
        return file_digest(path, "md5") == md5.lower()
    return size is not None


def is_safe_member(name: str) -> bool:
    """True if ``name`` is a relative path without ``..``, drive or root."""

    if not name or name.startswith(("/", "\\")) or "\x00" in name:
        return False
    windows = PureWindowsPath(name)
    if windows.drive or windows.root:
        return False
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return bool(parts) and ".." not in parts


def _wanted(name: str, limits: ExtractLimits, skip_parts: frozenset[str]) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if skip_parts.intersection(parts) or any(p.startswith("._") for p in parts):
        return False
    return PurePosixPath(name).suffix.lower() in limits.allowed_extensions


def safe_extract_zip(zip_path: str | Path, target: str | Path, *,
                     limits: ExtractLimits = DEFAULT_LIMITS,
                     skip_parts: frozenset[str] = frozenset({"__MACOSX", ".git"})) -> int:
    """Validate every member of a zip, then extract the wanted ones; return count."""

    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > limits.max_members:
            raise UnsafeArchiveError(f"{zip_path}: {len(members)} members")
        chosen = []
        for member in members:
            if not is_safe_member(member.filename):
                raise UnsafeArchiveError(f"{zip_path}: unsafe member {member.filename!r}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise UnsafeArchiveError(f"{zip_path}: symlink member {member.filename!r}")
            if member.file_size > limits.max_ratio * max(member.compress_size, 1024):
                raise UnsafeArchiveError(f"{zip_path}: suspicious ratio {member.filename!r}")
            if _wanted(member.filename, limits, skip_parts):
                chosen.append(member)
        total = sum(m.file_size for m in chosen)
        if total > limits.max_total_bytes:
            raise UnsafeArchiveError(f"{zip_path}: {total} bytes exceeds limit")
        written = 0
        for member in chosen:
            output = (root / member.filename.replace("\\", "/")).resolve()
            if resolved not in output.parents:
                raise UnsafeArchiveError(f"{zip_path}: escapes target {member.filename!r}")
            if output.is_file() and output.stat().st_size == member.file_size:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as sink:
                copied = 0
                while chunk := source.read(1 << 20):
                    copied += len(chunk)
                    if copied > member.file_size:
                        raise UnsafeArchiveError(f"{zip_path}: member grew {member.filename!r}")
                    sink.write(chunk)
            written += 1
    return written


def _seven_zip() -> str:
    for candidate in _SEVEN_ZIP_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    raise FileNotFoundError("7-Zip (7z) is required to extract .7z archives")


def list_7z(archive: str | Path) -> list[dict[str, str]]:
    """Parse ``7z l -slt`` into one dict per entry (keys: Path, Size, Attributes...)."""

    output = subprocess.run(  # noqa: S603 - fixed executable, argument list
        [_seven_zip(), "l", "-slt", "-ba", str(archive)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        key, _, value = line.partition(" = ")
        current[key.strip()] = value
    if current:
        entries.append(current)
    return [e for e in entries if "Path" in e]


def safe_extract_7z(archive: str | Path, target: str | Path, *,
                    limits: ExtractLimits = DEFAULT_LIMITS) -> int:
    """Validate a 7z listing, then extract only whitelisted extensions; return count."""

    entries = list_7z(archive)
    files = [e for e in entries if "D" not in e.get("Attributes", "")]
    if len(files) > limits.max_members:
        raise UnsafeArchiveError(f"{archive}: {len(files)} members")
    total = 0
    for entry in files:
        name = entry["Path"]
        if not is_safe_member(name):
            raise UnsafeArchiveError(f"{archive}: unsafe member {name!r}")
        tokens = entry.get("Attributes", "").split()
        if any(tok.startswith("l") and len(tok) == 10 for tok in tokens):
            raise UnsafeArchiveError(f"{archive}: link member {name!r}")
        if _wanted(name, limits, frozenset({"__MACOSX"})):
            total += int(entry.get("Size") or 0)
    if total > limits.max_total_bytes:
        raise UnsafeArchiveError(f"{archive}: {total} bytes exceeds limit")
    Path(target).mkdir(parents=True, exist_ok=True)
    patterns = [f"-ir!*{ext}" for ext in sorted(limits.allowed_extensions)]
    subprocess.run(  # noqa: S603 - fixed executable, argument list
        [_seven_zip(), "x", "-y", "-aos", f"-o{target}", str(archive), *patterns,
         "-xr!__MACOSX"],
        check=True, capture_output=True,
    )
    return sum(1 for e in files if _wanted(e["Path"], limits, frozenset({"__MACOSX"})))
