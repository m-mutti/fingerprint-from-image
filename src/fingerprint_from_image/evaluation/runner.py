"""FVC-protocol benchmark runner and the ``fingerprint-bench`` entry point.

For each FVC set the runner enrols every image once, scores the genuine pairs
and the impostor pairs (``"all"`` cross-finger pairs by default, from which the
official 45-pair FVC impostor subset is also extracted), and records:

* verification metrics for the ``fvc`` and (if computed) ``all`` impostor sets,
* closed-set identification (gallery = impression 1, probes = impressions 2..n),
* failure-to-enrol (FTE) and failure-to-compare counts — failed comparisons and
  pairs touching an FTE image are scored ``-inf`` (always rejected),
* latency: template extraction ms/image and comparison µs/pair (single process,
  image decoding excluded).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import FVC_DPI, FVC_SETS, FingerprintRecord, fvc_set_dir, load_fvc_set, load_images
from .metrics import IdentificationMetrics, VerificationMetrics
from .metrics import identification_metrics as _identification
from .metrics import verification_metrics as _verification
from .protocols import ImpostorProtocol, fvc_pairs
from .scorers import ClassicScorer, EmbeddingScorer, Scorer, load_embedder

__all__ = [
    "SetResult",
    "average_row",
    "environment_info",
    "format_markdown",
    "main",
    "run_benchmark",
    "run_fvc_set",
    "summary_rows",
]

logger = logging.getLogger(__name__)


@dataclass
class SetResult:
    """Everything measured on one FVC set."""

    name: str
    n_fingers: int
    n_impressions: int
    n_fte: int
    n_failed_comparisons: int
    enroll_ms_per_image: float
    compare_us_per_pair: float
    verification: dict[str, VerificationMetrics]
    identification: IdentificationMetrics | None
    genuine_scores: np.ndarray = field(repr=False)
    impostor_scores: np.ndarray = field(repr=False)
    impostor_protocol: ImpostorProtocol = "all"

    @property
    def n_images(self) -> int:
        return self.n_fingers * self.n_impressions

    @property
    def fte_rate(self) -> float:
        return self.n_fte / self.n_images

    def to_dict(self, *, include_scores: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "set": self.name,
            "dpi": FVC_DPI.get(self.name.rsplit("_", 1)[0]),
            "n_fingers": self.n_fingers,
            "n_impressions": self.n_impressions,
            "n_images": self.n_images,
            "n_fte": self.n_fte,
            "fte_rate": self.fte_rate,
            "n_failed_comparisons": self.n_failed_comparisons,
            "latency": {
                "enroll_ms_per_image": self.enroll_ms_per_image,
                "compare_us_per_pair": self.compare_us_per_pair,
            },
            "verification": {k: v.to_dict() for k, v in self.verification.items()},
            "identification": self.identification.to_dict() if self.identification else None,
        }
        if include_scores:
            result["scores"] = {
                "order": "fvc_pairs(n_fingers, n_impressions, impostor="
                f"{self.impostor_protocol!r}): genuine rows, then impostor rows",
                "genuine": _json_scores(self.genuine_scores),
                "impostor": _json_scores(self.impostor_scores),
            }
        return result


def _json_scores(scores: np.ndarray) -> list[float | None]:
    return [float(s) if np.isfinite(s) else None for s in scores]


def run_fvc_set(
    scorer: Scorer,
    records: Sequence[FingerprintRecord],
    *,
    impostor: ImpostorProtocol = "all",
    name: str | None = None,
    images: Sequence[np.ndarray] | None = None,
) -> SetResult:
    """Run the FVC protocol for one set indexed by :func:`load_fvc_set`."""

    fingers = list(dict.fromkeys(r.finger_id for r in records))
    n_fingers = len(fingers)
    if n_fingers < 2 or len(records) % n_fingers:
        raise ValueError("records must hold an equal number of impressions for >= 2 fingers")
    n_impressions = len(records) // n_fingers
    if any(r.finger_id != fingers[k // n_impressions] for k, r in enumerate(records)):
        raise ValueError("records must be grouped by finger (as returned by load_fvc_set)")
    pixels = list(images) if images is not None else load_images(records)

    start = time.perf_counter()
    templates = scorer.enroll(pixels)
    enroll_seconds = time.perf_counter() - start
    if len(templates) != len(pixels):
        raise ValueError("scorer.enroll must return one template per image")
    enrolled = np.asarray([t is not None for t in templates], dtype=bool)

    pairs = fvc_pairs(n_fingers, n_impressions, impostor)
    every = pairs.all_pairs()
    start = time.perf_counter()
    scores = np.asarray(scorer.compare_pairs(templates, every), dtype=np.float64)
    compare_seconds = time.perf_counter() - start
    attempted = enrolled[every[:, 0]] & enrolled[every[:, 1]]
    genuine = scores[: len(pairs.genuine)]
    impostor_scores = scores[len(pairs.genuine) :]

    verification = {}
    first = (pairs.impostor % n_impressions == 0).all(axis=1)  # impression 1 vs impression 1
    verification["fvc"] = _verification(genuine, impostor_scores[first])
    if impostor == "all":
        verification["all"] = _verification(genuine, impostor_scores)

    identification = None
    if impostor == "all":
        full = np.full((len(pixels), len(pixels)), -np.inf)
        full[every[:, 0], every[:, 1]] = scores
        full[every[:, 1], every[:, 0]] = scores
        gallery = np.arange(n_fingers) * n_impressions
        probes = np.setdiff1d(np.arange(len(pixels)), gallery)
        identification = _identification(
            full[np.ix_(probes, gallery)], probes // n_impressions, gallery // n_impressions
        )

    return SetResult(
        name=name or records[0].dataset,
        n_fingers=n_fingers,
        n_impressions=n_impressions,
        n_fte=int((~enrolled).sum()),
        n_failed_comparisons=int((attempted & ~np.isfinite(scores)).sum()),
        enroll_ms_per_image=1e3 * enroll_seconds / len(pixels),
        compare_us_per_pair=1e6 * compare_seconds / max(1, int(attempted.sum())),
        verification=verification,
        identification=identification,
        genuine_scores=genuine,
        impostor_scores=impostor_scores,
        impostor_protocol=impostor,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}"


def _row(name: str, r: dict[str, Any]) -> str:
    cells = [
        name,
        _pct(r["fte_rate"]),
        _pct(r["eer_fvc_protocol"]),
        _pct(r.get("eer_all")),
        _pct(r.get("fmr100_all")),
        _pct(r.get("fmr1000_all")),
        _pct(r.get("zero_fmr_all")),
        _pct(r.get("rank1")),
        f"{r['enroll_ms_per_image']:.1f}",
        f"{r['compare_us_per_pair']:.1f}",
    ]
    return "| " + " | ".join(cells) + " |"


def summary_rows(results: Sequence[SetResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "set": result.name,
            "fte_rate": result.fte_rate,
            "eer_fvc_protocol": result.verification["fvc"].eer,
            "enroll_ms_per_image": result.enroll_ms_per_image,
            "compare_us_per_pair": result.compare_us_per_pair,
        }
        if "all" in result.verification:
            metrics = result.verification["all"]
            row.update(
                eer_all=metrics.eer,
                fmr100_all=metrics.fmr100,
                fmr1000_all=metrics.fmr1000,
                zero_fmr_all=metrics.zero_fmr,
            )
        if result.identification is not None:
            row["rank1"] = result.identification.rank1
        rows.append(row)
    return rows


def average_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = [k for k in rows[0] if k != "set" and all(k in row for row in rows)]
    return {"set": "Average", **{k: float(np.mean([row[k] for row in rows])) for k in keys}}


def format_markdown(rows: Sequence[dict[str, Any]], scorer_name: str) -> str:
    """Markdown table (rates in %) with an average row over sets."""

    header = (
        f"### `{scorer_name}` on FVC DB_B (FVC protocol)\n\n"
        "| Set | FTE % | EER % (FVC imp.) | EER % (all imp.) | FMR100 % | FMR1000 % "
        "| ZeroFMR % | Rank-1 % | Enrol ms/img | Compare us/pair |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header, *(_row(row["set"], row) for row in rows)]
    if len(rows) > 1:
        average = average_row(rows)
        lines.append(_row("**Average**", average))
    return "\n".join(lines) + "\n"


def environment_info() -> dict[str, Any]:
    from .. import __version__

    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        info["git_commit"] = commit.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


def run_benchmark(
    scorer: Scorer,
    data_root: str | Path,
    sets: Sequence[str] = FVC_SETS,
    *,
    impostor: ImpostorProtocol = "all",
    output_dir: str | Path | None = None,
    tag: str | None = None,
    include_scores: bool = True,
) -> tuple[list[SetResult], str]:
    """Run several FVC sets; optionally write per-set JSON, summary JSON and Markdown."""

    label = tag or scorer.name
    results: list[SetResult] = []
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for set_name in sets:
        records = load_fvc_set(fvc_set_dir(data_root, set_name), set_name)
        logger.info("%s: %d images, scorer %s", set_name, len(records), label)
        result = run_fvc_set(scorer, records, impostor=impostor, name=set_name)
        results.append(result)
        logger.info(
            "%s: EER(fvc)=%.2f%% FTE=%d enrol=%.1fms compare=%.0fus",
            set_name,
            100 * result.verification["fvc"].eer,
            result.n_fte,
            result.enroll_ms_per_image,
            result.compare_us_per_pair,
        )
        if out is not None:
            payload = {"scorer": scorer.describe(), **result.to_dict(include_scores=include_scores)}
            (out / f"{label}_{set_name}.json").write_text(
                json.dumps(payload, indent=1) + "\n", encoding="utf-8"
            )
    rows = summary_rows(results)
    markdown = format_markdown(rows, label)
    if out is not None:
        summary = {
            "scorer": scorer.describe(),
            "environment": environment_info(),
            "impostor_protocol": impostor,
            "sets": rows,
            "average": average_row(rows) if rows else None,
        }
        (out / f"{label}_summary.json").write_text(
            json.dumps(summary, indent=1) + "\n", encoding="utf-8"
        )
        (out / f"{label}_summary.md").write_text(markdown, encoding="utf-8")
    return results, markdown


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingerprint-bench",
        description="Benchmark a fingerprint scorer on FVC2000/2002/2004 DB_B (FVC protocol)",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("FINGERPRINT_DATA_ROOT", "data"),
        help="directory containing fvc/FVC{year}_DB{n}_B (default: $FINGERPRINT_DATA_ROOT or data)",
    )
    parser.add_argument("--sets", default="all", help="comma-separated set names, or 'all'")
    parser.add_argument("--scorer", choices=("classic", "embedding"), default="classic")
    parser.add_argument("--embedder", help="embedding factory as 'package.module:Factory'")
    parser.add_argument("--embedder-kwargs", help="JSON object passed to the embedder factory")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding batch size")
    parser.add_argument("--tag", help="result file prefix (default: scorer name)")
    parser.add_argument(
        "--impostor",
        choices=("all", "fvc"),
        default="all",
        help="'all' scores every cross-finger pair (FVC subset also reported); 'fvc' only 45",
    )
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--no-scores", action="store_true", help="omit raw scores from JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``fingerprint-bench``."""

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sets = list(FVC_SETS) if args.sets == "all" else [s.strip() for s in args.sets.split(",")]
    unknown = [s for s in sets if s not in FVC_SETS]
    if unknown:
        print(f"error: unknown set(s) {unknown}; choose from {list(FVC_SETS)}", file=sys.stderr)
        return 2
    scorer: Scorer
    if args.scorer == "classic":
        scorer = ClassicScorer()
    else:
        if not args.embedder:
            print("error: --scorer embedding requires --embedder", file=sys.stderr)
            return 2
        embedder = load_embedder(args.embedder, args.embedder_kwargs)
        scorer = EmbeddingScorer(embedder, name=args.tag or "embedding", batch_size=args.batch_size)
    try:
        _results, markdown = run_benchmark(
            scorer,
            args.data_root,
            sets,
            impostor=args.impostor,
            output_dir=args.output_dir,
            tag=args.tag,
            include_scores=not args.no_scores,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
