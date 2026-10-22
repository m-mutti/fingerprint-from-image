"""Trainer tests on CPU with a tiny model: exact resume, guards, pair losses, selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.config import load_config  # noqa: E402
from training.losses import multi_similarity_loss, supcon_loss  # noqa: E402
from training.train_utils import set_seed  # noqa: E402
from training.trainer import TrainData, Trainer  # noqa: E402


def _cfg():
    return load_config(None, {
        "input_size": 64, "compile": "none", "log_every": 1, "seed": 3,
        "data.canvas": 80, "model.backbone": "resnet10t", "model.pretrained": False,
        "model.dim": 16, "loss.aux": "supcon", "optim.epochs": 2, "optim.k": 4,
        "optim.samples_per_epoch": 16, "optim.batch_size": 8, "optim.skip_probe": True,
    })


def _data(n_ids: int = 6, per_id: int = 3) -> TrainData:
    rng = np.random.default_rng(0)
    n = n_ids * per_id
    canvases = torch.from_numpy(rng.integers(0, 256, (n, 80, 80), dtype=np.uint8))
    return TrainData(canvases, torch.full((n,), 200.0), torch.tensor([[0, 0, 80, 80]] * n),
                     torch.arange(n_ids).repeat_interleave(per_id),
                     ["a"] * (n // 2) + ["b"] * (n - n // 2))


def _trainer(run_dir: Path) -> Trainer:
    set_seed(3)
    run_dir.mkdir(parents=True, exist_ok=True)
    return Trainer(_cfg(), run_dir, _data(), batch=8, accum=1, shares={"a": 0.5, "b": 0.5},
                   device="cpu")


def test_resume_is_exact(tmp_path: Path) -> None:
    straight = _trainer(tmp_path / "a")
    straight.train_epoch()
    straight.train_epoch()

    first = _trainer(tmp_path / "b")
    first.train_epoch()
    first.save_state()
    first._log({"kind": "step", "step": 99, "loss": 0.0})  # written after the checkpoint
    resumed = _trainer(tmp_path / "b")
    assert resumed.load_state() and resumed.epoch == 1 and resumed.step == first.step
    resumed.train_epoch()

    for (name, a), (_, b) in zip(straight.net.state_dict().items(),
                                 resumed.net.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=name)
    steps = [json.loads(line)["step"] for line in
             (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()]
    assert 99 not in steps and len(steps) == len(set(steps))  # truncated on resume


def test_nonfinite_step_is_skipped_and_bn_restored(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(tmp_path / "nan")
    before = [b.clone() for b in trainer._bn_buffers]
    weights = [p.detach().clone() for p in trainer.net.parameters()]
    original = trainer._loss

    def poisoned(idx):
        loss, x = original(idx)
        return loss * float("nan"), x

    monkeypatch.setattr(trainer, "_loss", poisoned)
    trainer.train_epoch()
    assert trainer.nonfinite == trainer.steps_per_epoch
    for a, b in zip(before, trainer._bn_buffers, strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(weights, trainer.net.parameters(), strict=True):
        torch.testing.assert_close(a, b.detach())


def test_metrics_refuse_nan_scores() -> None:
    from training.metrics import verification_metrics

    with pytest.raises(ValueError):
        verification_metrics(np.array([0.9, np.nan]), np.array([0.1, 0.2]))


def test_pair_losses_prefer_clustered_embeddings() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    centres = torch.eye(3)
    clustered = centres[labels] + 0.01 * torch.randn(6, 3)
    shuffled = centres[torch.tensor([0, 1, 2, 0, 1, 2])]
    for loss in (supcon_loss, multi_similarity_loss):
        assert loss(clustered, labels) < loss(shuffled, labels)
        assert torch.isfinite(loss(clustered, labels))


def test_pooled_selection_metric_uses_one_threshold() -> None:
    from training.validate import selection_metrics

    rng = np.random.default_rng(0)
    scores = {s: {"genuine": rng.normal(0.8, 0.1, 300), "impostor": rng.normal(0.2, 0.1, 3000)}
              for s in ("x", "y")}
    scores["y"]["genuine"] -= 0.3  # a harder sensor shifts the pooled threshold
    out = selection_metrics(scores, ["x", "y"])
    assert set(out) >= {"select", "select_eer", "pooled_eer"}
    alone = selection_metrics({"x": scores["x"]}, ["x"])
    assert out["select"] > alone["select"]


def test_drop_variants_removes_cr_only() -> None:
    from training.manifest import drop_variants

    rows = [{"variant": ""}, {"variant": "altered:CR:Easy"}, {"variant": "altered:Obl:Hard"}]
    assert [r["variant"] for r in drop_variants(rows, ["CR"])] == ["", "altered:Obl:Hard"]
