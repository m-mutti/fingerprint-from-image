"""Ridge-period estimator study (SOCOFing failure rate, FVC within-finger stability).

Compares the v1 global-spectrum estimator, v1 + print-size prior, and the
runtime Welch estimator (``preprocess.estimate_ridge_period``) with and without
the prior. Reproduces the estimator table in ``training/EXPERIMENTS.md``::

    python -m training.period_study                 # FVC2000/2002 only (default)
    python -m training.period_study --include-2004  # also the held-out FVC2004 (audit only)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fingerprint_from_image.embedding import preprocess as pp
from training.config import ROOT
from training.data import load_gray

__all__ = ["main"]

_NB = 64


def _peak(power: np.ndarray, n_y: int, n_x: int, max_period: float, rfft: bool = True) -> float:
    fy = np.fft.fftfreq(n_y)[:, None]
    fx = (np.fft.rfftfreq(n_x) if rfft else np.fft.fftfreq(n_x))[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    low, high = 1 / max_period, 1 / pp.MIN_PERIOD
    valid = (radius >= low) & (radius < high)
    edges = np.geomspace(low, high, _NB + 1)
    idx = np.clip(np.digitize(radius[valid], edges) - 1, 0, _NB - 1)
    prof = np.bincount(idx, weights=power[valid], minlength=_NB)[:_NB]
    prof = np.convolve(np.pad(prof, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    k = int(np.argmax(prof))
    lo, hi = max(k - 2, 0), min(k + 3, _NB)
    centres = np.log(np.sqrt(edges[:-1] * edges[1:]))[lo:hi]
    return float(1.0 / np.exp(np.sum(centres * prof[lo:hi]) / np.sum(prof[lo:hi])))


def global_spectrum(gray: np.ndarray, prior: bool = False) -> float:
    """v1: one Hann-windowed spectrum of the whole image."""

    g = gray.astype(np.float32)
    h, w = g.shape
    max_period = min(pp.MAX_PERIOD, max(h, w) / pp._MIN_RIDGES) if prior else pp.MAX_PERIOD
    power = np.abs(np.fft.rfft2((g - g.mean()) * np.outer(np.hanning(h), np.hanning(w)))) ** 2
    return _peak(power, h, w, max_period)


def welch_no_prior(gray: np.ndarray) -> float:
    """Runtime Welch averaging but with the full period range (no size prior)."""

    blocks, side = pp._welch_blocks(gray.astype(np.float32))
    n = 2 * side
    window = np.outer(np.hanning(side), np.hanning(side))
    power = np.abs(np.fft.rfft2(blocks * window, s=(n, n))) ** 2
    power /= power.sum(axis=(1, 2), keepdims=True) + 1e-12
    return _peak(power.mean(axis=0), n, n, pp.MAX_PERIOD)


VARIANTS = {
    "global (v1)": global_spectrum,
    "global + prior": lambda g: global_spectrum(g, prior=True),
    "welch + prior (runtime)": pp.estimate_ridge_period,
    "welch, no prior": welch_no_prior,
}


def _run(args: tuple[str, str]) -> list[float]:
    gray = load_gray(*args)
    return [fn(gray) for fn in VARIANTS.values()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-2004", action="store_true",
                        help="also read the held-out FVC2004 sets (never for decisions)")
    args = parser.parse_args()
    data = ROOT / "data"
    base = data / "socofing" / "socofing" / "train"
    soco = sorted(str(p) for p in base.glob("Real/*.BMP"))[::4]
    soco += sorted(str(p) for p in base.glob("Altered/*/*.BMP"))[::12]
    years = ("2000", "2002", "2004") if args.include_2004 else ("2000", "2002")
    sets = [f"FVC{y}_DB{i}_B" for y in years for i in range(1, 5)]
    fvc = [str(data / "fvc" / s / f"{f}_{i}.tif") for s in sets
           for f in range(101, 111) for i in range(1, 9)]
    with ProcessPoolExecutor(16) as pool:
        s = np.array(list(pool.map(_run, [(p, "socofing") for p in soco], chunksize=32)))
        f = np.array(list(pool.map(_run, [(p, "fvc") for p in fvc], chunksize=8)))
    altered = np.array(["Altered" in p for p in soco])
    print("| estimator | SOCOFing Real fail | Altered fail | FVC rel. std | FVC p95 ratio |")  # noqa: T201
    for k, name in enumerate(VARIANTS):
        fail = s[:, k] > 4.5
        per = f[:, k].reshape(len(sets), 10, 8)
        ratio = per.max(2) / per.min(2)
        rel = (per.std(2) / per.mean(2)).mean()
        print(f"| {name} | {fail[~altered].mean() * 100:.2f} % | "  # noqa: T201
              f"{fail[altered].mean() * 100:.2f} % | {rel:.3f} | "
              f"{np.percentile(ratio, 95):.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
