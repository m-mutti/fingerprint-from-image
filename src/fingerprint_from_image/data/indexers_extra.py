"""Indexers for UNSW 2D/3D, IMPOSE-SYN, L3-SF and the in-repo procedural synthetic set."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from .indexers import Draft, draft
from .registry import DATASETS
from .schema import canonical_finger

__all__ = ["index_impose_syn", "index_l3sf", "index_synthetic", "index_unsw_2d3d"]

# --- UNSW / UWA 2D & 3D ---------------------------------------------------------
# README (zenodo.org/records/10432121): names SIRE-<subject>_<finger>_<capture>.bmp;
# finger ids 1..10 ("SIRE-102_3_2.bmp ... right middle finger") -> ISO positions;
# 3D (2D-equivalent, contactless) captures 1-2, post-processed variants with a
# suffix (e.g. _HT1, _R414), intermediates in "raw"; 2D optical captures 1-4.
_UNSW = re.compile(r"^(?:SIRE-)?(\d+)_(\d+)_(\d+)(?:_([A-Za-z0-9]+))?\.bmp$", re.I)


def index_unsw_2d3d(data_root: Path, *, include_postprocessed: bool = False) -> Iterator[Draft]:
    """Contact (2D optical) and contactless (3D unwrapped) prints of the same fingers.

    The identity ``unsw_2d3d:<subject>:<finger>`` is shared by both
    modalities, giving real contact<->contactless positives.  ``raw``
    intermediate images are skipped; post-processed 3D variants are skipped
    unless ``include_postprocessed``.  The 3D resolution is not documented,
    so its dpi is estimated from the ridge period at build time.
    """

    root = data_root / "unsw_2d3d"
    license = DATASETS["unsw_2d3d"].license
    for path in sorted(root.rglob("*.bmp")):
        relative = path.relative_to(root).parts
        if any(part.lower() == "raw" for part in relative[:-1]):
            continue
        match = _UNSW.match(path.name)
        if not match:
            continue
        subject, finger, capture, suffix = match.groups()
        contact = relative[0].lower().startswith("2d")
        if suffix and not include_postprocessed:
            continue
        yield draft(
            path, dataset="unsw_2d3d", subject=str(int(subject)),
            finger=canonical_finger(position=int(finger)),
            impression=int(capture) + (0 if contact else 10),
            sensor="optical 2D sensor" if contact else "3D structured-light (unwrapped 2D)",
            modality="contact_optical" if contact else "contactless",
            dpi=500, dpi_source="estimate", license=license, split_unit=str(int(subject)),
            variant=f"postproc:{suffix}" if suffix else "",
            attributes={"capture": int(capture), "part": relative[0]},
        )


# --- IMPOSE-SYN -----------------------------------------------------------------
_IMPOSE_QUERY = re.compile(r"^(\d+)_(-?\d+)\.png$")


def index_impose_syn(data_root: Path) -> Iterator[Draft]:
    """1,712 synthetic identities: one rolled gallery print + ~8.8 contactless poses.

    README: gallery ``{id}.png``, query ``{id}_{angle}.png`` (angle -59..59),
    512x512, 500 PPI, Apache-2.0.  Impression 0 = rolled gallery print;
    queries are numbered 1.. by ascending angle.
    """

    root = data_root / "impose_syn"
    license = DATASETS["impose_syn"].license
    for path in sorted((root / "gallery").glob("*.png")):
        if path.stem.isdigit():
            yield draft(
                path, dataset="impose_syn", subject=path.stem, finger="unknown", impression=0,
                sensor="IMPOSE rolled (synthetic)", modality="synthetic", dpi=500,
                dpi_source="dataset README", license=license, split_unit=path.stem,
                synthetic=True, attributes={"appearance": "rolled"},
            )
    queries: dict[str, list[tuple[int, Path]]] = {}
    for path in (root / "query").glob("*.png"):
        match = _IMPOSE_QUERY.match(path.name)
        if match:
            queries.setdefault(match[1], []).append((int(match[2]), path))
    for subject in sorted(queries):
        for impression, (angle, path) in enumerate(sorted(queries[subject]), 1):
            yield draft(
                path, dataset="impose_syn", subject=subject, finger="unknown",
                impression=impression, sensor="IMPOSE contactless (synthetic)",
                modality="synthetic", dpi=500, dpi_source="dataset README", license=license,
                split_unit=subject, synthetic=True,
                attributes={"appearance": "contactless", "angle": angle},
            )


# --- L3-SF ----------------------------------------------------------------------
_L3SF = re.compile(r"^(\d+)_(\d)_(\d+)\.png$")


def index_l3sf(data_root: Path) -> Iterator[Draft]:
    """Five subsets R1..R5 x 148 identities x 10 samples (``id_{1,2}_{1..5}``).

    Visual check (docs/DATASETS.md): within a subset ``_1_``/``_2_`` are the
    same print; across subsets the same id shows a different ridge pattern, so
    identity = (subset, id), encoded as ``subject=<id>``, ``finger=finger_<k>``.
    All subsets of an id share a split unit, so a possible shared master can
    never leak across splits.  The pore ground-truth folder is skipped (copies).
    """

    root = data_root / "l3sf" / "L3SF_V2" / "L3-SF"
    license = DATASETS["l3sf"].license
    for subset in sorted(root.glob("R*")):
        k = int(subset.name[1:])
        for path in sorted(subset.glob("*.png")):
            match = _L3SF.match(path.name)
            if not match:
                continue
            ident, half, sample = match.groups()
            yield draft(
                path, dataset="l3sf", subject=ident, finger=f"finger_{k}",
                impression=(int(half) - 1) * 5 + int(sample), sensor="L3-SF generator",
                modality="synthetic", dpi=500, dpi_source="estimate", license=license,
                split_unit=ident, synthetic=True, attributes={"subset": subset.name},
            )


# --- In-repo procedural synthetic set ---------------------------------------------
def index_synthetic(data_root: Path, slug: str = "synth_procedural") -> Iterator[Draft]:
    """Identities generated by :mod:`fingerprint_from_image.data.synth` (``meta.json``)."""

    root = data_root / slug
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for item in meta["images"]:
        path = root / item["file"]
        yield draft(
            path, dataset=slug, subject=str(item["identity"]), finger="unknown",
            impression=int(item["impression"]), sensor=f"procedural v{meta['version']}",
            modality="synthetic", dpi=float(meta["dpi"]), dpi_source="generator setting",
            license=meta["license"], split_unit=str(item["identity"]), synthetic=True,
            attributes={"seed": item["seed"]},
        )
