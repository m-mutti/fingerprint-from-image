"""SourceAFIS (.NET, Apache-2.0) comparator for the FVC DB_B benchmark.

Runs the same FVC protocol, metrics and output format as ``fingerprint-bench``,
with SourceAFIS 3.14.0 doing template extraction and matching through the
small driver in ``sourceafis_dotnet/`` (built on first use with
``dotnet build -c Release``; requires the .NET SDK >= 8).

Images are decoded by Pillow and handed to SourceAFIS as raw 8-bit grayscale
with each database's nominal DPI (``FVC_DPI``), so SourceAFIS's own image
decoder is not used.  Reported latency is SourceAFIS's in-process time
(process start-up and JIT warm-up excluded); wall-clock per-call figures,
which include start-up, are stored under ``latency_wall``.

Usage::

    python benchmarks/comparators/sourceafis.py --data-root data --sets all
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fingerprint_from_image.evaluation import FVC_DPI, FVC_SETS, fvc_set_dir, load_fvc_set
from fingerprint_from_image.evaluation.runner import (
    average_row,
    environment_info,
    format_markdown,
    run_fvc_set,
    summary_rows,
)

logger = logging.getLogger("sourceafis")

PROJECT = Path(__file__).resolve().parent / "sourceafis_dotnet"
SOURCEAFIS_VERSION = "3.14.0"


def build_driver(project: Path = PROJECT) -> Path:
    """Build the .NET driver (idempotent) and return the executable path."""

    subprocess.run(
        ["dotnet", "build", "-c", "Release", "--nologo", "-v", "quiet", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("SourceAfisBench.exe", "SourceAfisBench"):
        candidate = project / "bin" / "Release" / "net10.0" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SourceAfisBench executable not found under {project / 'bin'}")


class SourceAfisScorer:
    """:class:`~fingerprint_from_image.evaluation.Scorer` backed by the SourceAFIS driver."""

    name = "sourceafis"

    def __init__(self, executable: Path, dpi: float, work_dir: Path) -> None:
        self.executable = executable
        self.dpi = dpi
        self.work_dir = work_dir
        self.timing: dict[str, float] = {}

    def _run(self, *args: str) -> dict[str, Any]:
        completed = subprocess.run(
            [str(self.executable), *args], check=True, capture_output=True, text=True
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def enroll(self, images: Sequence[np.ndarray]) -> list[Any | None]:
        lines = []
        for index, image in enumerate(images):
            pixels = np.ascontiguousarray(image, dtype=np.uint8)
            (self.work_dir / f"{index}.raw").write_bytes(pixels.tobytes())
            lines.append(f"{index} {pixels.shape[1]} {pixels.shape[0]} {index}.raw")
        (self.work_dir / "images.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
        report = self._run("extract", str(self.work_dir), f"{self.dpi:g}")
        self.timing["enroll_ms_per_image"] = float(report["ms_per_image"])
        return [
            index if (self.work_dir / f"{index}.tpl").is_file() else None
            for index in range(len(images))
        ]

    def compare_pairs(self, templates: Sequence[Any | None], pairs: np.ndarray) -> np.ndarray:
        pairs_path = self.work_dir / "pairs.txt"
        scores_path = self.work_dir / "scores.txt"
        rows = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        pairs_path.write_text("".join(f"{i} {j}\n" for i, j in rows), encoding="ascii")
        report = self._run("match", str(self.work_dir), str(pairs_path), str(scores_path))
        self.timing["compare_us_per_pair"] = float(report["us_per_pair"])
        values = scores_path.read_text(encoding="ascii").split()
        scores = np.asarray([float(v) for v in values], dtype=np.float64)
        if scores.size != len(rows):
            raise RuntimeError(f"driver returned {scores.size} scores for {len(rows)} pairs")
        return scores

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": f"SourceAFIS .NET {SOURCEAFIS_VERSION} (NuGet), raw-pixel input",
            "dpi": self.dpi,
            "score": "FingerprintMatcher(probe=reference).Match(candidate=probe image)",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sets", default="all")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--tag", default="sourceafis")
    parser.add_argument("--impostor", choices=("all", "fvc"), default="all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sets = list(FVC_SETS) if args.sets == "all" else [s.strip() for s in args.sets.split(",")]
    executable = build_driver()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    description: dict[str, Any] = {}
    for set_name in sets:
        records = load_fvc_set(fvc_set_dir(args.data_root, set_name), set_name)
        dpi = float(FVC_DPI[set_name.rsplit("_", 1)[0]])
        with tempfile.TemporaryDirectory(prefix="sourceafis_") as work:
            scorer = SourceAfisScorer(executable, dpi, Path(work))
            result = run_fvc_set(scorer, records, impostor=args.impostor, name=set_name)
        wall = {
            "enroll_ms_per_image": result.enroll_ms_per_image,
            "compare_us_per_pair": result.compare_us_per_pair,
        }
        result.enroll_ms_per_image = scorer.timing["enroll_ms_per_image"]
        result.compare_us_per_pair = scorer.timing["compare_us_per_pair"]
        description = {key: value for key, value in scorer.describe().items() if key != "dpi"}
        eer = 100 * result.verification["fvc"].eer
        logger.info("%s: EER(fvc)=%.2f%% FTE=%d", set_name, eer, result.n_fte)
        payload = {"scorer": scorer.describe(), **result.to_dict(), "latency_wall": wall}
        (out / f"{args.tag}_{set_name}.json").write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8"
        )
        results.append(result)
    rows = summary_rows(results)
    markdown = format_markdown(rows, args.tag)
    summary = {
        "scorer": description,
        "environment": environment_info(),
        "impostor_protocol": args.impostor,
        "sets": rows,
        "average": average_row(rows),
    }
    (out / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    (out / f"{args.tag}_summary.md").write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
