from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"
_spec = importlib.util.spec_from_file_location("download_datasets", _SCRIPT)
assert _spec is not None and _spec.loader is not None
download = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("download_datasets", download)  # dataclasses resolve the module
_spec.loader.exec_module(download)


def test_archive_table_is_complete() -> None:
    assert len(download.FVC_ARCHIVES) == 12
    entry = download.FVC_ARCHIVES["FVC2004_DB3_B"]
    assert entry.url == "http://bias.csr.unibo.it/fvc2004/Downloads/DB3_B.zip"
    assert all(len(a.sha256) == 64 for a in download.FVC_ARCHIVES.values())


def test_safe_extract_skips_git_and_rejects_traversal(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("socofing/train/Real/1__M_Left_index_finger.BMP", b"x")
        archive.writestr("socofing/.git/config", b"[core]")
    assert download.safe_extract(good, tmp_path / "out", skip_parts=frozenset({".git"})) == 1
    assert not (tmp_path / "out" / "socofing" / ".git").exists()
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../escape.txt", b"x")
    with pytest.raises(RuntimeError):
        download.safe_extract(evil, tmp_path / "out2")


def test_fetch_is_idempotent_for_verified_files(tmp_path: Path) -> None:
    payload = b"archive-bytes"
    target = tmp_path / "a.zip"
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    archive = download.Archive("a", "http://invalid.invalid/a.zip", digest)
    assert download.fetch(archive, target) == target  # no network access needed
