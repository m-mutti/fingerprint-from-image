"""Pipeline unit tests: keys/skip, guards, registry, P×K sampler, queue transitions.

They need numpy, Pillow and PyYAML only (no torch, no GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
TRAINING = Path(__file__).resolve().parents[1]
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from training.config import OptimConfig  # noqa: E402
from training.data import Sample, cache_inputs  # noqa: E402
from training.guards import (  # noqa: E402
    ContaminationError,
    check_selection_rows,
    check_train_samples,
    heldout_from_rows,
)
from training.keys import StepDir, canonical_json, content_key  # noqa: E402
from training.queue import Queue, classify_failure  # noqa: E402
from training.registry import Registry  # noqa: E402
from training.sampler import PKSampler  # noqa: E402


def test_content_key_is_order_independent_and_sensitive() -> None:
    a = content_key("train", {"x": 1, "y": [1, 2], "p": Path("a/b")})
    b = content_key("train", {"p": Path("a/b"), "y": [1, 2], "x": 1})
    assert a == b and len(a) == 64
    assert a != content_key("train", {"x": 2, "y": [1, 2], "p": Path("a/b")})
    assert a != content_key("eval", {"x": 1, "y": [1, 2], "p": Path("a/b")})
    assert canonical_json({"b": (1, 2), "a": {3}}) == '{"a":[3],"b":[1,2]}'


def test_step_dir_skip_and_invalidate(tmp_path: Path) -> None:
    step = StepDir.for_inputs("train", {"seed": 0}, tmp_path)
    assert not step.is_done()
    step.path.mkdir(parents=True)
    (step.path / "best.pt").write_bytes(b"weights")
    step.mark_done({"seed": 0}, ["best.pt"], {"metric": 1.0})
    assert step.is_done() and step.record()["metric"] == 1.0
    assert StepDir.for_inputs("train", {"seed": 0}, tmp_path).is_done()  # same key → skip
    (step.path / "best.pt").write_bytes(b"tampered")
    assert not step.is_done() and step.is_done(verify=False)
    step.invalidate()
    assert not step.is_done(verify=False)


def _rows() -> list[dict]:
    return [
        {"path": "D:/fvc/FVC2004_DB1_B/101_1.tif", "identity": "FVC2004_DB1:101:unknown",
         "split": "heldout", "sha256": "aa"},
        {"path": "D:/fvc/FVC2002_DB1_B/101_1.tif", "identity": "FVC2002_DB1:101:unknown",
         "split": "val", "sha256": "bb"},
    ]


def test_heldout_guard_rejects_contaminated_training_data() -> None:
    heldout = heldout_from_rows(_rows())
    clean = [Sample("D:/fvc/FVC2002_DB1_A/1_1.tif", "FVC2002_DB1:1:unknown", "FVC2002_DB1", "s")]
    summary = check_train_samples(clean, heldout)
    assert summary["heldout_overlap"] == 0 and summary["heldout_digest"] == heldout.digest
    by_path = [Sample("d:/FVC/FVC2004_DB1_B/101_1.tif", "x", "FVC2004_DB1", "s")]
    by_identity = [Sample("elsewhere.tif", "FVC2004_DB1:101:unknown", "FVC2004_DB1", "s")]
    by_hash = [Sample("copy.tif", "y", "other", "s", "aa")]
    for bad in (by_path, by_identity):
        with pytest.raises(ContaminationError):
            check_train_samples(bad, heldout)
    with pytest.raises(ContaminationError):
        check_train_samples(by_hash, heldout, {"copy.tif": "aa"})


def test_selection_guard_accepts_only_val_rows() -> None:
    rows = _rows()
    check_selection_rows([r for r in rows if r["split"] == "val"])
    with pytest.raises(ContaminationError):
        check_selection_rows(rows)


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert("train", "k1", "running", name="m1", config={"lr": 1e-3})
    registry.upsert("train", "k1", "done", metrics={"val_select": 3.5})
    row = registry.get("train", "k1")
    assert row["status"] == "done" and row["config"] == {"lr": 1e-3}
    assert row["metrics"]["val_select"] == 3.5
    assert [r["key"] for r in registry.rows("train", "done")] == ["k1"]
    with pytest.raises(ValueError):
        registry.upsert("train", "k2", "exploded")


def test_pk_sampler_batches_are_identity_and_source_balanced() -> None:
    labels = np.repeat(np.arange(30), 3)  # 30 identities × 3 images
    groups = ["a"] * 45 + ["b"] * 45  # identities 0-14 in a, 15-29 in b
    sampler = PKSampler(labels, groups, {"a": 0.5, "b": 0.5, "c": 1.0}, p=8, k=4, seed=1)
    batch = sampler.batch()
    assert batch.shape == (32,)
    ids = labels[batch].reshape(8, 4)
    assert (ids == ids[:, :1]).all()  # K images of one identity per row
    state = sampler.state()
    first = sampler.batch()
    sampler.load_state(state)
    np.testing.assert_array_equal(first, sampler.batch())
    counts = np.array([np.isin(labels[sampler.batch()], np.arange(15)).mean()
                       for _ in range(200)])
    assert 0.4 < counts.mean() < 0.6
    with pytest.raises(ValueError):
        PKSampler(labels, groups, {"z": 1.0}, p=2, k=2)


def test_lr_scaling_rules() -> None:
    base = OptimConfig(lr=5e-4, lr_ref_batch=128)
    assert base.scaled_lr(128) == pytest.approx(5e-4)
    assert base.scaled_lr(512) == pytest.approx(1e-3)  # sqrt rule
    assert OptimConfig(lr=5e-4, lr_ref_batch=128, lr_scaling="linear").scaled_lr(256) == 1e-3


def test_cache_key_tracks_content_not_order_of_fields(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    sample = Sample(str(image), "id", "synthetic", "s")
    hashed = Sample(str(image), "id", "FVC2002_DB1", "s", "abc")
    one = cache_inputs([sample, hashed], canvas=320, naive=False)
    assert one == cache_inputs([sample, hashed], canvas=320, naive=False)
    assert one != cache_inputs([sample, hashed], canvas=288, naive=False)
    assert one != cache_inputs([hashed, sample], canvas=320, naive=False)  # row order matters
    image.write_bytes(b"changed!")
    assert one != cache_inputs([sample, hashed], canvas=320, naive=False)


def test_queue_transitions_priority_and_failure_reasons(tmp_path: Path) -> None:
    queue = Queue(tmp_path / "queue")
    queue.submit(name="low", argv=["-c", "pass"], priority=0)
    queue.submit(name="high", argv=["-c", "raise SystemExit(3)"], priority=9)
    queue.submit(name="stall", argv=["-c", "import time; time.sleep(30)"], priority=5)
    assert "high" in queue.next_job().name
    order = []
    while (job := queue.next_job()) is not None:
        order.append(queue.run_one(job, stall_seconds=2, poll=0.2))
    assert [p.parent.name for p in order] == ["failed", "failed", "done"]
    import yaml

    reasons = [yaml.safe_load(p.read_text())["reason"] for p in order[:2]]
    assert reasons == ["exit 3", "stall"]
    queue.submit(name="orphan", argv=["-c", "pass"])
    orphan = queue.jobs("pending")[0]
    orphan.rename(queue.root / "running" / orphan.name)
    assert [p.parent.name for p in queue.recover()] == ["pending"]


def test_classify_failure() -> None:
    assert classify_failure("torch.cuda.OutOfMemoryError: CUDA out of memory", 1) == "oom"
    assert classify_failure("TrainingAborted: divergence: window loss", 1) == "divergence"
    assert classify_failure("ContaminationError: 3 training images", 1) == "contamination"
    assert classify_failure("ValueError: boom", 2) == "exit 2"
