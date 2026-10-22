"""Biometric verification and closed-set identification metrics (numpy only).

Scores are *similarities*: a comparison is accepted when ``score >= t``.  For a
threshold ``t``

* ``FMR(t)``  = fraction of impostor scores ``>= t`` (false match rate),
* ``FNMR(t)`` = fraction of genuine scores ``< t`` (false non-match rate).

Operating points are evaluated exactly at every distinct observed score plus
``+inf`` (accept nothing), so ties between genuine and impostor scores are
handled without bias.  Failures to enrol or compare should be passed in as
``-inf`` (the worst possible score): they are always rejected except at the
degenerate "accept everything" threshold.

The FVC definitions (Maio et al., FVC2000/2002/2004) are followed:
``FMR100`` / ``FMR1000`` are the lowest FNMR with ``FMR <= 1%`` / ``0.1%``,
``ZeroFMR`` the lowest FNMR with ``FMR = 0`` and ``ZeroFNMR`` the lowest FMR
with ``FNMR = 0``.  The EER is reported both as the FVC interval midpoint and as
the exact crossing of the linearly interpolated ROC (the segment between two
adjacent operating points is achievable by randomising the decision at the
threshold, so the crossing is an attainable operating point).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

__all__ = [
    "DEFAULT_FMR_TARGETS",
    "IdentificationMetrics",
    "VerificationMetrics",
    "det_points",
    "identification_metrics",
    "operating_points",
    "verification_metrics",
]

DEFAULT_FMR_TARGETS: tuple[float, ...] = (1e-2, 1e-3, 1e-4)


def _clean_scores(scores: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} scores are empty")
    if np.isnan(array).any():
        raise ValueError(f"{name} scores contain NaN; encode failures as -inf")
    if np.isposinf(array).any():
        raise ValueError(f"{name} scores contain +inf")
    return array


def operating_points(
    genuine: np.ndarray | Sequence[float], impostor: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(thresholds, fmr, fnmr)`` at every distinct score and ``+inf``.

    Thresholds are ascending, so FMR is non-increasing and FNMR non-decreasing.
    The first point accepts everything (``FMR = 1, FNMR = 0``) and the last
    (``+inf``) rejects everything (``FMR = 0, FNMR = 1``).
    """

    gen = np.sort(_clean_scores(genuine, "genuine"))
    imp = np.sort(_clean_scores(impostor, "impostor"))
    thresholds = np.append(np.unique(np.concatenate([gen, imp])), np.inf)
    fnmr = np.searchsorted(gen, thresholds, side="left") / gen.size
    fmr = (imp.size - np.searchsorted(imp, thresholds, side="left")) / imp.size
    return thresholds, fmr.astype(np.float64), fnmr.astype(np.float64)


def _fnmr_at_fmr(fmr: np.ndarray, fnmr: np.ndarray, target: float) -> float:
    """Lowest FNMR among operating points with ``FMR <= target``."""

    eligible = fmr <= target + 1e-15
    return float(fnmr[eligible].min())  # the +inf point (FMR = 0) always qualifies


def _subsample(count: int, max_points: int, keep: Sequence[int] = ()) -> np.ndarray:
    if count <= max_points:
        return np.arange(count)
    picked = np.linspace(0, count - 1, max_points).round().astype(np.int64)
    extra = np.asarray([k for k in keep if 0 <= k < count], dtype=np.int64)
    return np.unique(np.concatenate([picked, extra]))


@dataclass(frozen=True)
class VerificationMetrics:
    """Summary of a one-to-one verification experiment (rates are fractions)."""

    n_genuine: int
    n_impostor: int
    eer: float
    eer_threshold: float
    eer_fvc: float
    eer_low: float
    eer_high: float
    fmr100: float
    fmr1000: float
    zero_fmr: float
    zero_fnmr: float
    fnmr_at_fmr: dict[float, float]
    thresholds: np.ndarray = field(repr=False, compare=False)
    fmr: np.ndarray = field(repr=False, compare=False)
    fnmr: np.ndarray = field(repr=False, compare=False)

    @property
    def tar_at_far(self) -> dict[float, float]:
        """True accept rate (``1 - FNMR``) at each FMR target."""

        return {target: 1.0 - value for target, value in self.fnmr_at_fmr.items()}

    @property
    def fmr_resolution(self) -> float:
        """Smallest non-zero FMR the impostor set can express (``1 / n_impostor``)."""

        return 1.0 / self.n_impostor

    def resolvable(self, target: float) -> bool:
        """Whether ``FMR <= target`` differs from ``FMR = 0`` for this impostor count."""

        return self.n_impostor * target >= 1.0

    def roc(self) -> tuple[np.ndarray, np.ndarray]:
        """``(FMR, TAR)`` arrays, FMR descending."""

        return self.fmr.copy(), 1.0 - self.fnmr

    def to_dict(self, *, max_points: int = 512) -> dict[str, Any]:
        """JSON-ready summary with a subsampled ROC (``+inf`` written as ``null``)."""

        crossing = int(np.searchsorted(self.thresholds, self.eer_threshold))
        index = _subsample(self.thresholds.size, max_points, (crossing - 1, crossing))
        thresholds = [None if not np.isfinite(t) else float(t) for t in self.thresholds[index]]
        return {
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold if np.isfinite(self.eer_threshold) else None,
            "eer_fvc": self.eer_fvc,
            "eer_low": self.eer_low,
            "eer_high": self.eer_high,
            "fmr100": self.fmr100,
            "fmr1000": self.fmr1000,
            "zero_fmr": self.zero_fmr,
            "zero_fnmr": self.zero_fnmr,
            "fmr_resolution": self.fmr_resolution,
            "fnmr_at_fmr": {f"{t:g}": v for t, v in self.fnmr_at_fmr.items()},
            "tar_at_far": {f"{t:g}": v for t, v in self.tar_at_far.items()},
            "resolvable": {f"{t:g}": self.resolvable(t) for t in self.fnmr_at_fmr},
            "roc": {
                "threshold": thresholds,
                "fmr": self.fmr[index].tolist(),
                "fnmr": self.fnmr[index].tolist(),
            },
        }


def verification_metrics(
    genuine: np.ndarray | Sequence[float],
    impostor: np.ndarray | Sequence[float],
    *,
    fmr_targets: Sequence[float] = DEFAULT_FMR_TARGETS,
) -> VerificationMetrics:
    """Compute EER, FMR100/1000, ZeroFMR/ZeroFNMR and FNMR at FMR targets.

    Raises ``ValueError`` for empty inputs, NaN, or ``+inf`` scores.
    """

    thresholds, fmr, fnmr = operating_points(genuine, impostor)
    diff = fnmr - fmr  # non-decreasing from -1 (accept all) to +1 (reject all)
    k = int(np.argmax(diff >= 0.0))  # first point with FNMR >= FMR; k >= 1
    if diff[k] == 0.0:
        eer = float(fnmr[k])
        eer_threshold = float(thresholds[k])
    else:
        d0, d1 = float(diff[k - 1]), float(diff[k])
        alpha = d0 / (d0 - d1)
        eer = float(fmr[k - 1] + alpha * (fmr[k] - fmr[k - 1]))
        eer_threshold = float(thresholds[k])
    # FVC: t1 = largest t with FNMR <= FMR, t2 = smallest t with FNMR >= FMR.
    t1 = int(np.nonzero(diff <= 0.0)[0][-1])
    t2 = k
    if fnmr[t1] + fmr[t1] <= fmr[t2] + fnmr[t2]:
        low, high = float(fnmr[t1]), float(fmr[t1])
    else:
        low, high = float(fmr[t2]), float(fnmr[t2])
    targets = tuple(float(t) for t in fmr_targets)
    return VerificationMetrics(
        n_genuine=int(np.asarray(genuine).size),
        n_impostor=int(np.asarray(impostor).size),
        eer=eer,
        eer_threshold=eer_threshold,
        eer_fvc=(low + high) / 2.0,
        eer_low=min(low, high),
        eer_high=max(low, high),
        fmr100=_fnmr_at_fmr(fmr, fnmr, 1e-2),
        fmr1000=_fnmr_at_fmr(fmr, fnmr, 1e-3),
        zero_fmr=_fnmr_at_fmr(fmr, fnmr, 0.0),
        zero_fnmr=float(fmr[fnmr <= 0.0].min()),
        fnmr_at_fmr={t: _fnmr_at_fmr(fmr, fnmr, t) for t in targets},
        thresholds=thresholds,
        fmr=fmr,
        fnmr=fnmr,
    )


def det_points(
    metrics: VerificationMetrics, *, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Probit-transformed ``(FMR, FNMR)`` for a DET plot, rates clipped to ``[eps, 1-eps]``."""

    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    normal = NormalDist()
    probit = np.vectorize(normal.inv_cdf, otypes=[np.float64])
    return (
        probit(np.clip(metrics.fmr, eps, 1.0 - eps)),
        probit(np.clip(metrics.fnmr, eps, 1.0 - eps)),
    )


TieMode = Literal["pessimistic", "optimistic", "expected"]


@dataclass(frozen=True)
class IdentificationMetrics:
    """Closed-set identification (CMC) summary; rates are fractions."""

    n_queries: int
    n_excluded: int
    n_gallery_identities: int
    cmc: np.ndarray = field(repr=False, compare=False)

    def rank(self, k: int) -> float:
        """Identification rate at rank ``k`` (1-based); saturates beyond the gallery size."""

        if k < 1:
            raise ValueError("rank must be >= 1")
        return float(self.cmc[min(k, self.cmc.size) - 1])

    @property
    def rank1(self) -> float:
        return self.rank(1)

    @property
    def rank5(self) -> float:
        return self.rank(5)

    @property
    def rank10(self) -> float:
        return self.rank(10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_excluded": self.n_excluded,
            "n_gallery_identities": self.n_gallery_identities,
            "rank1": self.rank1,
            "rank5": self.rank5,
            "rank10": self.rank10,
            "cmc": self.cmc.tolist(),
        }


def identification_metrics(
    score_matrix: np.ndarray,
    query_labels: Sequence[Any] | np.ndarray,
    gallery_labels: Sequence[Any] | np.ndarray,
    *,
    ties: TieMode = "pessimistic",
) -> IdentificationMetrics:
    """Cumulative match characteristic for a ``(Q, G)`` similarity matrix.

    Gallery templates sharing a label are fused by their maximum score, so ranks
    are over *identities*.  Queries whose label is absent from the gallery are
    excluded (reported in ``n_excluded``).  NaN scores are treated as ``-inf``.
    ``ties`` decides how gallery identities scoring exactly the mate score
    count: ``pessimistic`` (ranked ahead of the mate), ``optimistic`` (behind),
    or ``expected`` (uniformly random tie-break; the CMC is its expectation).
    """

    scores = np.asarray(score_matrix, dtype=np.float64)
    queries = np.asarray(query_labels)
    gallery = np.asarray(gallery_labels)
    if scores.ndim != 2 or scores.shape != (queries.size, gallery.size):
        raise ValueError("score_matrix must have shape (len(query_labels), len(gallery_labels))")
    if ties not in ("pessimistic", "optimistic", "expected"):
        raise ValueError("ties must be 'pessimistic', 'optimistic', or 'expected'")
    if gallery.size == 0:
        raise ValueError("gallery is empty")
    scores = np.where(np.isnan(scores), -np.inf, scores)
    identities, inverse = np.unique(gallery, return_inverse=True)
    fused = np.full((queries.size, identities.size), -np.inf)
    for column, identity_index in enumerate(inverse.reshape(-1)):
        fused[:, identity_index] = np.maximum(fused[:, identity_index], scores[:, column])

    lookup = {label: index for index, label in enumerate(identities.tolist())}
    mate_index = np.asarray([lookup.get(label, -1) for label in queries.tolist()], dtype=np.int64)
    valid = mate_index >= 0
    n_valid = int(valid.sum())
    n_ids = identities.size
    if n_valid == 0:
        raise ValueError("no query label is present in the gallery")
    rows = fused[valid]
    mate = rows[np.arange(n_valid), mate_index[valid]][:, None]
    better = (rows > mate).sum(axis=1)
    tied = (rows == mate).sum(axis=1) - 1  # exclude the mate itself
    ranks = np.arange(1, n_ids + 1)[None, :]
    if ties == "pessimistic":
        hit = (better + tied + 1)[:, None] <= ranks
        cmc = hit.mean(axis=0)
    elif ties == "optimistic":
        cmc = ((better + 1)[:, None] <= ranks).mean(axis=0)
    else:
        # Mate position is uniform over better+1 .. better+tied+1.
        reach = (ranks - better[:, None]) / (tied[:, None] + 1.0)
        cmc = np.clip(reach, 0.0, 1.0).mean(axis=0)
    return IdentificationMetrics(
        n_queries=n_valid,
        n_excluded=int(queries.size - n_valid),
        n_gallery_identities=int(n_ids),
        cmc=cmc.astype(np.float64),
    )
