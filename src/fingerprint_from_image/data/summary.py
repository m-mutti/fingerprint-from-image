"""Markdown summary of the manifests (per dataset, per split, per modality)."""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["summarise", "write_summary"]


def _fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return str(size)


def _median_ridge(rows: Sequence[Mapping[str, Any]]) -> str:
    periods = [r["attributes"]["ridge_period_px"] for r in rows
               if r.get("attributes", {}).get("ridge_period_px")]
    return f"{statistics.median(periods):.1f}" if periods else "-"


def summarise(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One dict per dataset with counts over non-excluded rows (plus excluded count)."""

    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    table = []
    for dataset in sorted(by_dataset):
        items = by_dataset[dataset]
        used = [r for r in items if not r.get("exclude")]
        per_identity = Counter(r["identity"] for r in used)
        counts = sorted(per_identity.values()) or [0]
        split_images = Counter(r["split"] for r in used)
        split_ids = {s: len({r["identity"] for r in used if r["split"] == s}) for s in split_images}
        size = 0
        for r in items:
            try:
                size += os.path.getsize(r["path"])
            except OSError:
                pass
        table.append({
            "dataset": dataset,
            "identities": len(per_identity),
            "images": len(used),
            "excluded": len(items) - len(used),
            "imp_min": counts[0],
            "imp_median": statistics.median(counts),
            "imp_max": counts[-1],
            "modality": ", ".join(sorted({r["modality"] for r in used})),
            "dpi": ", ".join(str(round(d)) for d in sorted({float(r["dpi"]) for r in used})),
            "dpi_source": ", ".join(sorted({r["dpi_source"] for r in used})),
            "license": ", ".join(sorted({r["license"] for r in items})),
            "ridge": _median_ridge(used),
            "splits": ", ".join(f"{s} {split_ids[s]}/{split_images[s]}"
                                for s in ("train", "val", "test", "heldout") if s in split_images),
            "disk": _fmt_bytes(size),
        })
    return table


def write_summary(rows: Sequence[Mapping[str, Any]], path: str | Path, *,
                  dpi_report: Mapping[str, Any] | None = None,
                  dup_stats: Mapping[str, int] | None = None,
                  corrupt: Sequence[tuple[str, list[str]]] = ()) -> Path:
    """Write ``SUMMARY.md`` (dataset table, split x modality totals, QA notes)."""

    lines = ["# Fingerprint manifests summary", "",
             "Counts exclude rows with `exclude: true`. Splits column: "
             "`split identities/images`. Ridge period = median dominant ridge "
             "wavelength in pixels (about 9.2 px on 500-dpi FVC sensors), a check on "
             "the effective resolution of each set.", "",
             "| dataset | identities | images | excluded | impr./identity min/med/max | "
             "modality | dpi | dpi source | ridge period px | licence | splits (ids/imgs) | disk |",
             "|---|---:|---:|---:|---|---|---|---|---:|---|---|---:|"]
    for t in summarise(rows):
        lines.append(
            f"| {t['dataset']} | {t['identities']} | {t['images']} | {t['excluded']} | "
            f"{t['imp_min']}/{t['imp_median']}/{t['imp_max']} | {t['modality']} | {t['dpi']} | "
            f"{t['dpi_source']} | {t['ridge']} | {t['license']} | {t['splits']} | {t['disk']} |")
    used = [r for r in rows if not r.get("exclude")]
    lines += ["", "## Totals by split and modality (identities / images)", "",
              "| modality | train | val | test | heldout |", "|---|---|---|---|---|"]
    for modality in sorted({r["modality"] for r in used}):
        cells = []
        for split in ("train", "val", "test", "heldout"):
            sel = [r for r in used if r["modality"] == modality and r["split"] == split]
            cells.append(f"{len({r['identity'] for r in sel})} / {len(sel)}")
        lines.append(f"| {modality} | " + " | ".join(cells) + " |")
    cells = []
    for split in ("train", "val", "test", "heldout"):
        sel = [r for r in used if r["split"] == split]
        cells.append(f"**{len({r['identity'] for r in sel})} / {len(sel)}**")
    lines.append("| **all** | " + " | ".join(cells) + " |")
    lines += ["", "## Quality checks", ""]
    if dup_stats:
        stats = ", ".join(f"{k}={v}" for k, v in dup_stats.items())
        lines.append(f"- Duplicate resolution: {stats}")
    lines.append(f"- Undecodable / unusable files dropped: {len(corrupt)}")
    for path_, issues in list(corrupt)[:20]:
        lines.append(f"  - `{Path(path_).name}`: {'; '.join(issues)}")
    issue_counts = Counter(i for r in rows for i in r.get("attributes", {}).get("issues", []))
    for issue, count in issue_counts.most_common(10):
        lines.append(f"- Flag `{issue}`: {count} images")
    if dpi_report:
        lines += ["", "## Estimated dpi (ridge period vs 500-dpi FVC sensors)", ""]
        for key, value in dpi_report.items():
            lines.append(f"- {key}: {value}")
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
