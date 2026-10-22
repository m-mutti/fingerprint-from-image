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
    epochs = [r for r in records if r.get("kind") == "epoch" and "val" in r]
    if not epochs:
        return
    x = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, [r["val"]["select"] for r in epochs], "k-o", lw=2.5,
            label="selection (contact FVC + neurotech)")
    for name in epochs[-1]["val"]["sets"]:
        ax.plot(x, [r["val"]["sets"].get(name, {}).get("eer", np.nan) for r in epochs],
                lw=1, label=name)
    ax.set(xlabel="epoch", ylabel="all-pairs EER %", title="validation EER (manifest val ids)",
           yscale="log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
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
    axes[0].set(title="val cosine scores (selection sets)", xlabel="cosine")
    axes[0].legend()
    for name, s in scores.items():
        gen, imp = np.sort(s["genuine"]), np.sort(s["impostor"])
        thresholds = np.linspace(-1.0, 1.0, 401)
        fmr = 1.0 - np.searchsorted(imp, thresholds, side="left") / len(imp)
        fnmr = np.searchsorted(gen, thresholds, side="left") / len(gen)
        axes[1].plot(np.clip(fmr, 1e-5, 1), np.clip(fnmr, 1e-5, 1), label=name)
    axes[1].set(xscale="log", yscale="log", xlabel="FMR", ylabel="FNMR",
                title="DET (selection val sets)")
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
