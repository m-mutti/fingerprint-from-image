"""Build ``data/manifests/<slug>.jsonl``, ``all.jsonl`` and ``SUMMARY.md``.

Usage::

    python -m fingerprint_from_image.data.build --data-root data
    python -m fingerprint_from_image.data.build --data-root data --datasets fvc,socofing

Steps: index files -> decode/hash every image (parallel, cached) -> estimate
dpi where undocumented -> assign subject-disjoint splits -> exact + near
duplicate resolution across *all* datasets -> leakage check -> write.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ProbeCache
from .dedup import find_near_duplicate_pairs, resolve_duplicates, verify_pairs
from .indexers import (
    Draft,
    index_fvc,
    index_mendeley_family,
    index_naijafingset,
    index_neurotech,
    index_socofing,
    index_tdfd,
)
from .indexers_extra import index_impose_syn, index_l3sf, index_synthetic, index_unsw_2d3d
from .quality import ImageProbe, probe_image
from .schema import write_manifest
from .splits import assign_subject_splits, check_splits, load_subject_split_file
from .summary import write_summary

logger = logging.getLogger(__name__)

DEFAULT_SOCOFING_SPLIT = Path("benchmarks/splits/socofing_subject_split_seed0.json")
# Calibration: real contact sets with a documented 500 dpi.
_CALIBRATION_DATASETS = ("FVC2000_DB1", "FVC2000_DB3", "FVC2002_DB1", "FVC2002_DB3")


def _indexers(args: argparse.Namespace) -> dict[str, Callable[[Path], Iterable[Draft]]]:
    split_path = args.socofing_split
    return {
        "fvc": index_fvc,
        "socofing": lambda root: index_socofing(root, load_subject_split_file(split_path)),
        "tdfd": index_tdfd,
        "neurotech": index_neurotech,
        "mendeley_family": index_mendeley_family,
        "naijafingset": index_naijafingset,
        "unsw_2d3d": index_unsw_2d3d,
        "impose_syn": index_impose_syn,
        "l3sf": index_l3sf,
        "synth_procedural": index_synthetic,
    }


def _probe_task(path: str) -> ImageProbe:
    return probe_image(path, descriptor=True, ridge_period=True)


def probe_all(drafts: list[Draft], cache: ProbeCache, workers: int) -> list[ImageProbe]:
    """Probe every draft's image, reusing cached results for unchanged files."""

    todo = [(i, d) for i, d in enumerate(drafts) if cache.get(d["path"]) is None]
    if todo:
        logger.info("probing %d images (%d cached)", len(todo), len(drafts) - len(todo))
        items = [d["path"] for _, d in todo]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, probe in enumerate(pool.map(_probe_task, items, chunksize=64), 1):
                cache.put(probe)
                if done % 5000 == 0:
                    logger.info("  %d / %d", done, len(items))
        cache.save()
    return [cache.get(d["path"]) for d in drafts]  # type: ignore[misc]


def _estimate_dpi(rows: list[dict[str, Any]], probes: list[ImageProbe]) -> dict[str, Any]:
    """Replace dpi for ``dpi_source == 'estimate'`` rows using ridge periods.

    dpi = 500 * median_period(group) / median_period(500-dpi calibration sets),
    per (dataset, sensor) group.  Returns the calibration numbers for the report.
    """

    calibration = [p.ridge_period for r, p in zip(rows, probes, strict=True)
                   if r["dataset"] in _CALIBRATION_DATASETS and p.ridge_period]
    if not calibration:
        logger.warning("no calibration ridge periods; estimated dpi left at 500")
        return {}
    reference = float(np.median(calibration))
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, probe in zip(rows, probes, strict=True):
        if row["dpi_source"] == "estimate" and probe.ridge_period:
            groups[(row["dataset"], row["sensor"])].append(probe.ridge_period)
    report: dict[str, Any] = {"reference_period_px_at_500dpi": round(reference, 2)}
    for (dataset, sensor), periods in groups.items():
        dpi = round(500 * float(np.median(periods)) / reference)
        report[f"{dataset} / {sensor}"] = {"median_period_px": round(float(np.median(periods)), 2),
                                           "dpi": dpi, "n": len(periods)}
        for row in rows:
            if row["dataset"] == dataset and row["sensor"] == sensor \
                    and row["dpi_source"] == "estimate":
                row["dpi"] = dpi
                row["dpi_source"] = "estimate:ridge-period vs 500-dpi FVC"
    return report


def _assign_splits(rows: list[dict[str, Any]], seed: int) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, items in by_dataset.items():
        missing = [r for r in items if "split" not in r]
        if not missing:
            continue
        if any("split" in r for r in items):
            units = sorted({r["split_unit"] for r in missing})
            raise ValueError(f"{dataset}: {len(units)} split units lack a split, e.g. {units[:5]}")
        mapping = assign_subject_splits((r["split_unit"] for r in items), seed=seed)
        for row in items:
            row["split"] = mapping[row["split_unit"]]


def build(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    out_dir = (args.out or data_root / "manifests").resolve()
    indexers = _indexers(args)
    names = list(indexers) if args.datasets == "all" else args.datasets.split(",")
    rows: list[dict[str, Any]] = []
    for name in names:
        found = list(indexers[name](data_root))
        logger.info("%-18s %7d files indexed", name, len(found))
        rows.extend(found)
    cache = ProbeCache(out_dir / ".cache" / "probes_v4.npz")
    probes = probe_all(rows, cache, args.workers)
    corrupt = [(r, p) for r, p in zip(rows, probes, strict=True) if not p.ok]
    kept = [(r, p) for r, p in zip(rows, probes, strict=True) if p.ok]
    rows, probes = [r for r, _ in kept], [p for _, p in kept]
    dpi_report = _estimate_dpi(rows, probes)
    for row, probe in zip(rows, probes, strict=True):
        row.update(sha256=probe.sha256, width=probe.width, height=probe.height, mode=probe.mode)
        extra: dict[str, Any] = {}
        if probe.issues:
            extra["issues"] = probe.issues
        if probe.ridge_period:
            extra["ridge_period_px"] = round(probe.ridge_period, 2)
        if extra:
            row["attributes"] = {**row["attributes"], **extra}
    _assign_splits(rows, args.seed)

    originals = [i for i, r in enumerate(rows) if not r["variant"]]
    descriptors = np.stack([np.asarray(probes[i].descriptor) for i in originals])
    identities = [rows[i]["identity"] for i in originals]
    half = descriptors.shape[1] // 2
    best: dict[tuple[int, int], float] = {}
    for part in (descriptors[:, :half], descriptors[:, half:]):
        for i, j, score in find_near_duplicate_pairs(part, identities,
                                                     threshold=args.near_dup_threshold):
            best[(i, j)] = max(score, best.get((i, j), -1.0))
    del descriptors
    candidates = [(originals[i], originals[j], s) for (i, j), s in sorted(best.items())]
    pairs, scored = verify_pairs(rows, candidates, threshold=args.fine_threshold,
                                 workers=args.workers)
    logger.info("near-duplicate candidates %d -> verified %d", len(candidates), len(pairs))
    stats = resolve_duplicates(rows, pairs)
    logger.info("duplicates: %s", stats)
    for row in rows:
        row.pop("split_unit", None)
        row.setdefault("duplicate_group", None)
        row.setdefault("exclude", False)
        row.setdefault("exclude_reason", None)
    problems = check_splits(rows)
    if problems:
        for problem in problems[:50]:
            logger.error("leak: %s", problem)
        return 2

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"].split("_DB")[0] if row["dataset"].startswith("FVC")
                   else row["dataset"]].append(row)
    for slug, items in sorted(by_dataset.items()):
        write_manifest(items, out_dir / f"{slug}.jsonl")
    if args.datasets == "all":
        write_manifest(rows, out_dir / "all.jsonl")
    _write_reports(out_dir, rows, pairs, corrupt, stats, dpi_report, scored)
    write_summary(rows, out_dir / "SUMMARY.md", dpi_report=dpi_report, dup_stats=stats,
                  corrupt=[(r["path"], p.issues) for r, p in corrupt])
    logger.info("wrote %d rows to %s", len(rows), out_dir)
    return 0


def _write_reports(out_dir: Path, rows: list[dict[str, Any]], pairs: list[tuple[int, int, float]],
                   corrupt: list[tuple[Draft, ImageProbe]], stats: dict[str, int],
                   dpi_report: dict[str, Any], scored: list[tuple[int, int, float]]) -> None:
    kept = set(pairs)
    rejected = sorted((t for t in scored if t not in kept), key=lambda t: -t[2])
    report = {
        "rejected_candidates_top": [
            {"a": rows[i]["path"], "b": rows[j]["path"], "fine_cosine": round(s, 4)}
            for i, j, s in rejected[:300]
        ],
        "fine_cosine_histogram_all_candidates": np.histogram(
            [s for _, _, s in scored], bins=[-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        )[0].tolist(),
        "near_duplicate_pairs": [
            {"a": rows[i]["path"], "b": rows[j]["path"], "fine_cosine": round(s, 4),
             "a_split": rows[i]["split"], "b_split": rows[j]["split"]}
            for i, j, s in sorted(pairs, key=lambda t: -t[2])
        ],
        "corrupt": [{"path": r["path"], "issues": p.issues} for r, p in corrupt],
        "dedup_stats": stats,
        "dpi_estimates": dpi_report,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training manifests for all datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=None, help="default: <data-root>/manifests")
    parser.add_argument("--datasets", default="all", help="comma-separated indexer names")
    parser.add_argument("--socofing-split", type=Path, default=DEFAULT_SOCOFING_SPLIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--near-dup-threshold", type=float, default=0.6,
                        help="coarse 64x64 cosine for candidate pairs")
    parser.add_argument("--fine-threshold", type=float, default=0.6,
                        help="160x160 ridge-crop cosine a candidate must reach")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
