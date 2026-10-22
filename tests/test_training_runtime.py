"""Audit-MVP tests: typed config, pixel-content guard, GPU lock, phase heartbeats."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.guards import ContaminationError, check_content  # noqa: E402


def test_typed_config_coercion() -> None:
    from training.config import load_config

    cfg = load_config(None, {"optim.lr": "5e-4", "loss.scale": 45, "seed": "2",
                             "data.manifest_train": {"FVC2000": 1}})
    assert cfg.optim.lr == 5e-4 and isinstance(cfg.loss.scale, float) and cfg.seed == 2
    assert cfg.data.manifest_train == {"FVC2000": 1.0} and cfg.data.fvc_train_sets == ()
    with pytest.raises(KeyError):
        load_config(None, {"optim.lr_typo": 1})


def test_pixel_content_guard_catches_reencoded_copies(tmp_path: Path) -> None:
    from PIL import Image
    from training.content import pixel_hashes

    pixels = (np.arange(64 * 48) % 251).astype(np.uint8).reshape(48, 64)
    Image.fromarray(pixels).save(tmp_path / "held.png", compress_level=1)
    Image.fromarray(pixels).save(tmp_path / "copy.png", compress_level=9)  # other bytes
    Image.fromarray(255 - pixels).save(tmp_path / "other.png")
    names = ("held.png", "copy.png", "other.png")
    hashes = pixel_hashes([str(tmp_path / n) for n in names], cache=tmp_path / "c.json",
                          workers=1)
    held, copy, other = (hashes[str(tmp_path / n)] for n in names)
    assert (tmp_path / "held.png").read_bytes() != (tmp_path / "copy.png").read_bytes()
    assert held == copy != other
    check_content({"o": other}, {}, {"h": held})
    with pytest.raises(ContaminationError):
        check_content({"c": copy}, {}, {"h": held})
    with pytest.raises(ContaminationError):
        check_content({"c": copy}, {"v": held}, {})


def test_gpu_lock_is_exclusive(tmp_path: Path) -> None:
    from training.runtime import GpuLock

    first = GpuLock(tmp_path / "gpu.lock")
    first.acquire()
    second = GpuLock(tmp_path / "gpu.lock", poll=0.05)
    acquired = threading.Event()
    thread = threading.Thread(target=lambda: (second.acquire(), acquired.set()), daemon=True)
    thread.start()
    assert not acquired.wait(0.4)  # blocked while the first holder runs
    first.release()
    assert acquired.wait(5)
    second.release()


def test_heartbeat_phase_beats_and_respects_cap(tmp_path: Path) -> None:
    from training.runtime import Heartbeat

    beat = Heartbeat(tmp_path / "hb", interval=0.05)
    with beat.phase("cache"):
        first = float((tmp_path / "hb").read_text())
        time.sleep(0.3)
        assert float((tmp_path / "hb").read_text()) > first
    with beat.phase("epoch", max_seconds=0.1):
        time.sleep(0.4)
        last = float((tmp_path / "hb").read_text())
        time.sleep(0.2)
        assert float((tmp_path / "hb").read_text()) == last  # capped: no more beats


def test_atomic_replace(tmp_path: Path) -> None:
    from training.runtime import atomic_replace

    (tmp_path / "a.tmp").write_text("new")
    (tmp_path / "a").write_text("old")
    atomic_replace(tmp_path / "a.tmp", tmp_path / "a")
    assert (tmp_path / "a").read_text() == "new" and not (tmp_path / "a.tmp").exists()
