"""Finger-level paired bootstrap for comparing two checkpoints on the val selection sets.

Identities (fingers) are resampled with replacement within each val set; both models
are scored on the *same* resample, so the CI of the difference reflects how much the
comparison depends on which fingers happen to be in the val set. A change is adopted
only if the CI of ``metric(B) - metric(A)`` lies below zero.

Usage (from Python)::

    from training.stats import paired_bootstrap
    ci = paired_bootstrap(emb_a, emb_b, labels, selection=SELECTION_SETS)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from training.manifest import all_pair_scores
from training.validate import selection_metrics

__all__ = ["paired_bootstrap"]


def _resample(labels: Sequence[str], rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    lab = np.asarray(labels)
    ids = np.unique(lab)
    chosen = rng.choice(ids, size=len(ids), replace=True)
    index, new_labels = [], []
    for copy, ident in enumerate(chosen):  # duplicated fingers become distinct identities
        members = np.nonzero(lab == ident)[0]
        index.extend(members.tolist())
        new_labels.extend([f"{ident}#{copy}"] * len(members))
    return np.asarray(index), new_labels


def paired_bootstrap(emb_a: Mapping[str, np.ndarray], emb_b: Mapping[str, np.ndarray],
                     labels: Mapping[str, Sequence[str]], *, selection: Sequence[str],
                     metric: str = "select", n: int = 200, seed: int = 0) -> dict[str, float]:
    """Mean and 95 % CI of ``metric(B) - metric(A)`` over finger-level resamples."""

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        scores_a, scores_b = {}, {}
        for name in selection:
            index, new_labels = _resample(labels[name], rng)
            for emb, out in ((emb_a, scores_a), (emb_b, scores_b)):
                gen, imp = all_pair_scores(emb[name][index], new_labels)
                out[name] = {"genuine": gen, "impostor": imp}
        diffs.append(selection_metrics(scores_b, selection)[metric]
                     - selection_metrics(scores_a, selection)[metric])
    values = np.asarray(diffs)
    return {"mean_diff": float(values.mean()), "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)), "n": n}
