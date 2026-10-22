"""Numpy verification metrics and FVC pair protocols (self-contained)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

__all__ = ["VerificationMetrics", "fvc_pairs", "verification_metrics"]


@dataclass(frozen=True)
class VerificationMetrics:
    """Error rates in percent; ``fmr100`` = FNMR at FMR ≤ 1 %, ``fmr1000`` at ≤ 0.1 %."""

    eer: float
    fmr100: float
    fmr1000: float
    zero_fmr: float
    n_genuine: int
    n_impostor: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def fvc_pairs(
    n_fingers: int = 10, n_impressions: int = 8, *, impostor: Literal["fvc", "all"] = "all"
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs into a ``finger * n_impressions + impression`` ordering.

    Genuine: every unordered same-finger impression pair. Impostor ``"fvc"``:
    first impression vs first impression across fingers; ``"all"``: every
    cross-finger pair.
    """

    genuine = [
        (f * n_impressions + a, f * n_impressions + b)
        for f in range(n_fingers)
        for a in range(n_impressions)
        for b in range(a + 1, n_impressions)
    ]
    if impostor == "fvc":
        imp = [(f * n_impressions, g * n_impressions)
               for f in range(n_fingers) for g in range(f + 1, n_fingers)]
    else:
        total = n_fingers * n_impressions
        imp = [(i, j) for i in range(total) for j in range(i + 1, total)
               if i // n_impressions != j // n_impressions]
    return np.asarray(genuine, dtype=np.int64), np.asarray(imp, dtype=np.int64)


def _fnmr_at(genuine: np.ndarray, impostor: np.ndarray, max_fmr: float) -> float:
    """Lowest FNMR over thresholds whose FMR ≤ ``max_fmr`` (score ≥ t accepts)."""

    imp = np.sort(impostor)[::-1]
    allowed = int(np.floor(max_fmr * len(imp) + 1e-9))  # impostors allowed above threshold
    threshold = imp[allowed] if allowed < len(imp) else -np.inf
    # accept strictly above the first disallowed impostor score
    return float(np.mean(genuine <= threshold) * 100.0)


def verification_metrics(genuine: np.ndarray, impostor: np.ndarray) -> VerificationMetrics:
    """EER (linear interpolation at the FMR/FNMR crossing), FMR100, FMR1000, ZeroFMR."""

    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if not (np.isfinite(genuine).all() and np.isfinite(impostor).all()):
        raise ValueError("non-finite scores (NaN/inf embeddings) — refusing to compute metrics")
    thresholds = np.unique(np.concatenate([genuine, impostor, [-np.inf, np.inf]]))
    g_sorted, i_sorted = np.sort(genuine), np.sort(impostor)
    fnmr = np.searchsorted(g_sorted, thresholds, side="left") / len(genuine)
    fmr = 1.0 - np.searchsorted(i_sorted, thresholds, side="left") / len(impostor)
    diff = fnmr - fmr  # increases with threshold
    k = int(np.searchsorted(diff, 0.0, side="left"))
    if k == 0:
        eer = (fnmr[0] + fmr[0]) / 2
    else:
        d0, d1 = diff[k - 1], diff[min(k, len(diff) - 1)]
        w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
        eer_fnmr = fnmr[k - 1] + w * (fnmr[min(k, len(diff) - 1)] - fnmr[k - 1])
        eer_fmr = fmr[k - 1] + w * (fmr[min(k, len(diff) - 1)] - fmr[k - 1])
        eer = (eer_fnmr + eer_fmr) / 2
    return VerificationMetrics(
        eer=float(eer * 100.0),
        fmr100=_fnmr_at(genuine, impostor, 0.01),
        fmr1000=_fnmr_at(genuine, impostor, 0.001),
        zero_fmr=_fnmr_at(genuine, impostor, 0.0),
        n_genuine=len(genuine),
        n_impostor=len(impostor),
    )
