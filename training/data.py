"""Dataset indexing, subject-disjoint splits and the preprocessed canvas cache.

Every training image is converted once with the *runtime* preprocessing
(:mod:`fingerprint_from_image.embedding.preprocess`): ridge-period estimation,
rescaling to ``TARGET_PERIOD`` and a crop around the foreground centroid. The
crop is a larger ``canvas`` (default 320 px) so GPU augmentation can rotate,
scale and translate before cutting the final ``INPUT_SIZE`` window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fingerprint_from_image.embedding import preprocess as pp
from fingerprint_from_image.io import load_image
from training.keys import StepDir, canonical_json

__all__ = [
    "Sample",
    "build_cache",
    "cache_inputs",
    "image_fingerprint",
    "calibration_inputs",
    "fvc_samples",
    "load_gray",
    "model_input",
    "socofing_samples",
    "split_from_file",
    "split_socofing",
]

logger = logging.getLogger(__name__)
_SOCO_TRIM = (2, 4)  # SOCOFing frame: 2 px top/left, 4 px bottom/right


@dataclass(frozen=True)
class Sample:
    """One image file with its finger identity."""

    path: str
    label: str  # finger identity key
    source: str  # "socofing" | FVC set name | "synthetic"
    subject: str
    sha256: str | None = None  # image content hash from the manifest, when known


def load_gray(path: str, source: str) -> np.ndarray:
    """Load a grayscale uint8 image (16-bit/LA/RGBA handled); strips the SOCOFing frame."""

    gray = load_image(path, mode="L")
    if source.lower() == "socofing":
        top, bottom = _SOCO_TRIM
        gray = gray[top:-bottom, top:-bottom]
    return gray


def socofing_samples(root: Path, *, altered: bool = True) -> list[Sample]:
    """Index SOCOFing Real (+ Altered) images from all shipped splits."""

    base = root / "socofing" / "socofing"
    patterns = ["*/Real/*.BMP"] + (["*/Altered/*/*.BMP"] if altered else [])
    samples: list[Sample] = []
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            stem = path.stem  # 100__M_Left_index_finger[_CR]
            subject, rest = stem.split("__", 1)
            parts = rest.split("_")  # M, Left, index, finger, [CR]
            label = f"soco_{subject}_{parts[1]}_{parts[2]}"
            samples.append(Sample(str(path), label, "socofing", f"soco_{subject}"))
    return samples


def split_socofing(
    samples: Sequence[Sample], n_val_subjects: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Subject-disjoint split: ``n_val_subjects`` random subjects go to validation."""

    subjects = sorted({s.subject for s in samples})
    rng = np.random.default_rng(seed)
    val_subjects = set(rng.permutation(subjects)[:n_val_subjects].tolist())
    train = [s for s in samples if s.subject not in val_subjects]
    val = [s for s in samples if s.subject in val_subjects]
    return train, val


def split_from_file(samples: Sequence[Sample], path: Path) -> dict[str, list[Sample]]:
    """Subject split from the shared benchmark JSON (``subjects: {train, val, test}``)."""

    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    owner = {f"soco_{sid}": part for part, ids in spec["subjects"].items() for sid in ids}
    parts: dict[str, list[Sample]] = {part: [] for part in spec["subjects"]}
    for sample in samples:
        parts[owner[sample.subject]].append(sample)
    return parts


def fvc_samples(root: Path, sets: Sequence[str]) -> list[Sample]:
    """Index FVC ``{finger}_{impression}.tif`` files; label = set + finger."""

    samples: list[Sample] = []
    for name in sets:
        for path in sorted((root / "fvc" / name).glob("*.tif")):
            finger, _ = path.stem.split("_")
            samples.append(Sample(str(path), f"{name}_{finger}", name, f"{name}_{finger}"))
    return samples


def _naive_prepare(gray: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, float], float]:
    """Ablation: fit the longer side to 90% of ``size`` (no ridge-scale normalisation)."""

    height, width = gray.shape
    scale = 0.9 * size / max(height, width)
    image = Image.fromarray(gray.astype(np.float32)).resize(
        (max(8, round(width * scale)), max(8, round(height * scale))), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    center, background = pp.foreground_center(array)
    return array, center, background


def model_input(gray: np.ndarray, *, naive: bool = False, size: int = pp.INPUT_SIZE) -> np.ndarray:
    """Evaluation-time preprocessing (runtime path, or the naive ablation)."""

    if not naive:
        return pp.preprocess(gray, size)
    image, center, background = _naive_prepare(gray, size)
    return pp.normalize_intensity(pp.crop_around(image, center, size, background))


def _period_one(args: tuple[str, str]) -> float:
    path, source = args
    return pp.estimate_ridge_period(pp.trim_uniform_border(load_gray(path, source)))


def _canvas_one(args: tuple[str, str, int, bool, float | None]) -> tuple[np.ndarray, ...]:
    path, source, canvas, naive, period = args
    gray = load_gray(path, source)
    if naive:
        image, center, background = _naive_prepare(gray, pp.INPUT_SIZE)
        period = float("nan")
    else:
        prepared = pp.prepare(gray, period)
        image, center, background = prepared.image, prepared.center, prepared.background
        period = prepared.period
    window = pp.crop_around(image, center, canvas, background)
    extent = float(max(image.shape))
    top = int(round(center[0] - canvas / 2.0))
    left = int(round(center[1] - canvas / 2.0))
    box = np.asarray([max(0, -top), max(0, -left), min(canvas, image.shape[0] - top),
                      min(canvas, image.shape[1] - left)], dtype=np.int16)
    return np.clip(np.rint(window), 0, 255).astype(np.uint8), background, period, extent, box


def _real_counterpart(path: str) -> str:
    """SOCOFing Altered file stem → its Real file stem (``..._finger_CR`` → ``..._finger``)."""

    return Path(path).stem.rsplit("_", 1)[0]


def image_fingerprint(sample: Sample) -> str:
    """Manifest SHA-256 when known, else ``path|size|mtime_ns`` (generated/legacy files)."""

    if sample.sha256:
        return sample.sha256
    stat = Path(sample.path).stat()
    return f"{Path(sample.path).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"


def cache_inputs(samples: Sequence[Sample], *, canvas: int, naive: bool) -> dict:
    """Content-key inputs of the canvas cache (image list digest + preprocessing params)."""

    images = [[image_fingerprint(s), s.source] for s in samples]
    return {
        "images_sha256": hashlib.sha256(canonical_json(images).encode()).hexdigest(),
        "n_images": len(samples), "canvas": canvas, "naive": naive,
        "preprocess": pp.PREPROCESS_VERSION, "target_period": pp.TARGET_PERIOD,
        "socofing_period_inheritance": True,
    }


def build_cache(
    samples: Sequence[Sample], root: Path | None = None, *, canvas: int, naive: bool,
    workers: int = 12,
) -> tuple[dict[str, np.ndarray], str]:
    """Content-keyed canvas cache → (arrays, cache key); skipped when already built.

    Arrays: ``canvases``, ``background``, ``period``, ``extent``, ``valid_box``.
    SOCOFing Altered images are pixel-aligned edits of their Real image, so they reuse
    the Real image's ridge period (obliteration blobs otherwise corrupt the spectral
    estimate); every other image uses the runtime estimator.
    """

    inputs = cache_inputs(samples, canvas=canvas, naive=naive)
    step = StepDir.for_inputs("cache", inputs, root)
    target = step.path / "canvases.npz"
    if step.is_done(verify=False) and target.is_file():
        with np.load(target) as stored:
            return {name: stored[name] for name in stored.files}, step.key
    started = time.time()
    step.path.mkdir(parents=True, exist_ok=True)
    logger.info("Building canvas cache for %d images -> %s", len(samples), step.path)
    real = {Path(s.path).stem: s for s in samples
            if s.source.lower() == "socofing" and "/Real/" in s.path.replace("\\", "/")}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        real_periods = dict(zip(real, pool.map(
            _period_one, [(s.path, s.source) for s in real.values()], chunksize=64), strict=True))
        jobs = []
        for s in samples:
            inherited = None
            if s.source.lower() == "socofing" and "/Altered/" in s.path.replace("\\", "/"):
                inherited = real_periods.get(_real_counterpart(s.path))
            elif s.source.lower() == "socofing":
                inherited = real_periods.get(Path(s.path).stem)
            jobs.append((s.path, s.source, canvas, naive, inherited))
        results = list(pool.map(_canvas_one, jobs, chunksize=64))
    arrays = {
        "canvases": np.stack([r[0] for r in results]),
        "background": np.asarray([r[1] for r in results], dtype=np.float32),
        "period": np.asarray([r[2] for r in results], dtype=np.float32),
        "extent": np.asarray([r[3] for r in results], dtype=np.float32),
        "valid_box": np.stack([r[4] for r in results]),  # (top, left, bottom, right) real pixels
    }
    small = int((arrays["extent"] < 100).sum())
    logger.info("cache: %d / %d images have rescaled extent < 100 px", small, len(samples))
    np.savez(target, **arrays)
    # verify=False on reuse: hashing a multi-GB cache on every start is not worth it
    step.mark_done(inputs, [], {"small_extent": small, "file": target.name}, started)
    return arrays, step.key


def calibration_inputs(n: int = 256, seed: int = 0) -> np.ndarray:
    """Preprocessed images from the *training* split only (SOCOFing train subjects + FVC2000)."""

    from training.config import ROOT

    data = ROOT / "data"
    split = ROOT / "benchmarks" / "splits" / "socofing_subject_split_seed0.json"
    spec = json.loads(split.read_text(encoding="utf-8"))
    train_subjects = {f"soco_{s}" for s in spec["subjects"]["train"]}
    pool: list[Sample] = [s for s in socofing_samples(data) if s.subject in train_subjects]
    fvc2000 = [f"FVC2000_DB{i}_B" for i in range(1, 5)]
    pool += fvc_samples(data, fvc2000) * 10  # up-weight the multi-sensor images
    rng = np.random.default_rng(seed)
    chosen = [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]
    return np.stack([model_input(load_gray(s.path, s.source)) for s in chosen])[:, None]
