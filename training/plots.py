"""Training-run plots: loss/LR curves, validation EER, score histograms and DET curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["write_plots"]


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _steps(records: list[dict], out: Path) -> None:
    steps = [r for r in records if r.get("kind") == "step"]
    if not steps:
        return
    x = [r["step"] for r in steps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, [r["loss"] for r in steps], lw=1)
    axes[0].set(title="train loss (margin softmax)", xlabel="step")
    axes[1].plot(x, [r["lr"] for r in steps], lw=1, label="lr")
    ax2 = axes[1].twinx()
    ax2.plot(x, [r["margin"] for r in steps], lw=1, color="tab:orange", label="margin")
    axes[1].set(title="LR (blue) / margin (orange)", xlabel="step")
    axes[2].plot(x, [r["grad_norm"] for r in steps], lw=1)
    axes[2].set(title="grad norm (pre-clip)", xlabel="step", yscale="log")
    fig.tight_layout()
    fig.savefig(out / "train_curves.png", dpi=110)
    plt.close(fig)


def _val(records: list[dict], out: Path) -> None:
    epochs = [r for r in records if r.get("kind") == "epoch" and "val_fvc2002" in r]
    if not epochs:
        return
    x = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [r["val_fvc2002"]["eer_all"] for r in epochs], "o-", label="FVC2002 EER (all)")
    ax.plot(x, [r["val_fvc2002"]["eer_fvc"] for r in epochs], "s--", label="FVC2002 EER (fvc)")
    if "val_soco_eer" in epochs[0]:
        ax.plot(x, [r["val_soco_eer"] for r in epochs], "^-", label="SOCOFing-val EER")
    ax.set(xlabel="epoch", ylabel="EER %", title="validation EER", yscale="log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_eer.png", dpi=110)
    plt.close(fig)


def _det(scores: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    if not scores:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    genuine = np.concatenate([s["genuine"] for s in scores.values()])
    impostor = np.concatenate([s["impostor"] for s in scores.values()])
    bins = np.linspace(-0.5, 1.0, 76)
    axes[0].hist(impostor, bins=bins, alpha=0.6, density=True, label="impostor")
    axes[0].hist(genuine, bins=bins, alpha=0.6, density=True, label="genuine")
    axes[0].set(title="FVC2002 cosine scores (all sets)", xlabel="cosine")
    axes[0].legend()
    for name, s in scores.items():
        thresholds = np.unique(np.concatenate([s["genuine"], s["impostor"]]))
        fmr = [(s["impostor"] >= t).mean() for t in thresholds]
        fnmr = [(s["genuine"] < t).mean() for t in thresholds]
        axes[1].plot(np.clip(fmr, 1e-4, 1), np.clip(fnmr, 1e-4, 1), label=name)
    axes[1].set(xscale="log", yscale="log", xlabel="FMR", ylabel="FNMR", title="DET (FVC2002)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scores_det.png", dpi=110)
    plt.close(fig)


def write_plots(run_dir: Path, scores: dict[str, dict[str, np.ndarray]] | None = None) -> None:
    """Regenerate all PNGs under ``run_dir/plots`` from ``metrics.jsonl`` (+ latest scores)."""

    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    records = _read(run_dir / "metrics.jsonl")
    _steps(records, out)
    _val(records, out)
    _det(scores or {}, out)
