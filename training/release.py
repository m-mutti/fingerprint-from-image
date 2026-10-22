"""Final checkpoint choice (protocol v1): prefer ``last`` unless ``best`` is clearly better.

``best.pt`` is chosen by the noisy val metric, so it is kept only if it beats ``last.pt``
on the pooled val FNMR @ FMR 0.1 % by more than 2 bootstrap standard deviations of the
finger-level paired difference; otherwise ``last.pt`` (less selection-biased) is used.
The choice is copied to ``selected.pt``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from training.stats import paired_bootstrap
from training.train_utils import load_model
from training.validate import Validator, torch_embed_fn

__all__ = ["choose_release"]


def choose_release(run_dir: Path, validator: Validator, device: str = "cpu",
                   n_boot: int = 200) -> dict:
    """Compare ``best.pt`` vs ``last.pt`` on val; write ``selected.pt``; return the decision."""

    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    if not best_path.is_file():
        shutil.copyfile(last_path, run_dir / "selected.pt")
        return {"selected": "last", "reason": "no best.pt"}
    embeddings = {}
    for name, path in (("best", best_path), ("last", last_path)):
        model, _ = load_model(path, device)
        embeddings[name] = validator.embeddings(torch_embed_fn(model.cpu()))
    labels = {s: validator.sets[s][1] for s in validator.selection}
    boot = paired_bootstrap(embeddings["last"], embeddings["best"], labels,
                            selection=validator.selection, n=n_boot)
    # diff = metric(best) - metric(last); negative = best is better (lower FNMR)
    keep_best = bool(-boot["mean_diff"] > 2.0 * boot["sd"] and boot["sd"] > 0)
    chosen = best_path if keep_best else last_path
    shutil.copyfile(chosen, run_dir / "selected.pt")
    return {"selected": "best" if keep_best else "last", "bootstrap": boot,
            "rule": "best only if it beats last by > 2 bootstrap SD (pooled val FNMR@0.1%)",
            "mean_best_minus_last": float(np.round(boot["mean_diff"], 4))}
