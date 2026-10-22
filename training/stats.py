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

__all__ = ["fvc_bootstrap", "paired_bootstrap"]


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
    return {"mean_diff": float(values.mean()), "sd": float(values.std(ddof=1)) if n > 1 else 0.0,
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)), "n": n}


def _fvc_pairs_for(fingers: np.ndarray, n_imp: int) -> tuple[np.ndarray, np.ndarray]:
    """Genuine / FVC-impostor index pairs for a (possibly resampled) finger list.

    Copies of the same finger never form impostor pairs.
    """

    gen, imp = [], []
    for a, finger in enumerate(fingers):
        base = int(finger) * n_imp
        gen += [(base + i, base + j) for i in range(n_imp) for j in range(i + 1, n_imp)]
        imp += [(base, int(other) * n_imp) for other in fingers[a + 1 :] if other != finger]
    return np.asarray(gen), np.asarray(imp)


def fvc_bootstrap(scores: Mapping[str, np.ndarray], *, n_imp: int = 8, n: int = 1000,
                  seed: int = 0,
                  reference: Mapping[str, np.ndarray] | None = None) -> dict[str, float]:
    """Finger-level bootstrap of the mean (over DBs) EER with FVC impostors (protocol v1).

    ``scores[db]`` is the (F*n_imp, F*n_imp) similarity matrix of one database, finger-major.
    With ``reference`` (same layout, e.g. SourceAFIS scores), the paired difference
    ``EER(model) - EER(reference)`` is bootstrapped on the same finger resamples.
    """

    from training.metrics import verification_metrics

    rng = np.random.default_rng(seed)
    names = sorted(scores)
    fingers = {db: scores[db].shape[0] // n_imp for db in names}

    def mean_eer(mats: Mapping[str, np.ndarray], draw: Mapping[str, np.ndarray]) -> float:
        eers = []
        for db in names:
            gen, imp = _fvc_pairs_for(draw[db], n_imp)
            mat = mats[db]
            eers.append(verification_metrics(mat[gen[:, 0], gen[:, 1]],
                                             mat[imp[:, 0], imp[:, 1]]).eer)
        return float(np.mean(eers))

    identity = {db: np.arange(fingers[db]) for db in names}
    point = mean_eer(scores, identity)
    samples, diffs = [], []
    for _ in range(n):
        draw = {db: rng.integers(0, fingers[db], fingers[db]) for db in names}
        samples.append(mean_eer(scores, draw))
        if reference is not None:
            diffs.append(samples[-1] - mean_eer(reference, draw))
    out = {"mean_eer": point, "ci_low": float(np.percentile(samples, 2.5)),
           "ci_high": float(np.percentile(samples, 97.5)), "n": n}
    if reference is not None:
        out.update(ref_mean_eer=mean_eer(reference, identity),
                   diff_ci_low=float(np.percentile(diffs, 2.5)),
                   diff_ci_high=float(np.percentile(diffs, 97.5)))
    return out
