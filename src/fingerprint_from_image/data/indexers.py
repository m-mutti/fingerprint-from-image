"""Per-dataset indexers that turn files on disk into draft manifest rows.

A *draft* is a manifest row without the image-derived fields (``width``,
``height``, ``sha256``, ``mode``) and usually without ``split``; it carries a
``split_unit`` (the unit that must not straddle splits, normally the person)
and optionally a fixed ``split``.  :mod:`.build` fills in the rest.

Facts used here (sensors, dpi, finger numbering) are cited next to each
indexer; anything not documented is marked ``dpi_source="estimate"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..evaluation.datasets import load_socofing
from .registry import DATASETS
from .schema import canonical_finger, make_identity

__all__ = [
    "INDEXERS",
    "draft",
    "index_fvc",
    "index_mendeley_family",
    "index_neurotech",
    "index_socofing",
    "index_tdfd",
]

Draft = dict[str, Any]


def draft(path: Path, *, dataset: str, subject: str, finger: str, impression: int,
          sensor: str, modality: str, dpi: float, dpi_source: str, license: str,
          split_unit: str, split: str | None = None, synthetic: bool = False,
          variant: str = "", attributes: dict[str, Any] | None = None) -> Draft:
    """Assemble one draft row (absolute path, identity derived from the labels)."""

    row: Draft = {
        "path": str(path.resolve()),
        "dataset": dataset,
        "subject": subject,
        "finger": finger,
        "identity": make_identity(dataset, subject, finger),
        "impression": int(impression),
        "sensor": sensor,
        "modality": modality,
        "dpi": dpi,
        "dpi_source": dpi_source,
        "license": license,
        "synthetic": synthetic,
        "variant": variant,
        "attributes": attributes or {},
        "split_unit": split_unit,
    }
    if split is not None:
        row["split"] = split
    return row


# --- FVC -------------------------------------------------------------------------
# bias.csr.unibo.it/fvc{2000,2002,2004}/databases.asp (read 2026-10-22).
_FVC_SENSORS: dict[str, tuple[str, str, int, str]] = {
    "FVC2000_DB1": ("KeyTronic Secure Desktop Scanner", "contact_optical", 500, "spec"),
    "FVC2000_DB2": ("ST Microelectronics TouchChip", "contact_capacitive", 500, "spec"),
    "FVC2000_DB3": ("Identicator DF-90", "contact_optical", 500, "spec"),
    "FVC2000_DB4": ("SFinGe (synthetic)", "synthetic", 500, "spec (about 500 dpi)"),
    "FVC2002_DB1": ("Identix TouchView II", "contact_optical", 500, "spec"),
    "FVC2002_DB2": ("Biometrika FX2000", "contact_optical", 569, "spec"),
    "FVC2002_DB3": ("Precise Biometrics 100 SC", "contact_capacitive", 500, "spec"),
    "FVC2002_DB4": ("SFinGe v2.51 (synthetic)", "synthetic", 500, "spec (about 500 dpi)"),
    "FVC2004_DB1": ("CrossMatch V300", "contact_optical", 500, "spec"),
    "FVC2004_DB2": ("Digital Persona U.are.U 4000", "contact_optical", 500, "spec"),
    "FVC2004_DB3": ("Atmel FingerChip FCD4B14CB (sweep)", "contact_thermal", 512, "spec"),
    "FVC2004_DB4": ("SFinGe v3.0 (synthetic)", "synthetic", 500, "spec (about 500 dpi)"),
}
_FVC_FILE = re.compile(r"^(\d+)_(\d+)\.tif$", re.I)


def index_fvc(data_root: Path) -> Iterator[Draft]:
    """FVC2000/2002/2004 DB1-4, set A (fingers 1-100) and set B (101-110).

    Set A comes from the Handbook ESM (``data/fvc_full``); set B from the
    benchmark copies in ``data/fvc``.  Fingers are not linked to people, so
    each finger is its own subject.  Splits: FVC2004 -> ``heldout``;
    FVC2000/2002 set A -> ``train``, set B -> ``val`` (the benchmark sets).
    """

    license = DATASETS["fvc_full"].license
    dvd = data_root / "fvc_full" / "74034_3_En_4_MOESM1_ESM"
    for name, (sensor, modality, dpi, source) in _FVC_SENSORS.items():
        year, db = name[3:7], name[-1]
        folders = {"A": dvd / f"FVC{year}" / "Dbs" / f"Db{db}_a",
                   "B": data_root / "fvc" / f"{name}_B"}
        for set_name, folder in folders.items():
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                match = _FVC_FILE.match(path.name)
                if not match:
                    continue
                finger_no, impression = match.groups()
                if year == "2004":
                    split = "heldout"
                else:
                    split = "train" if set_name == "A" else "val"
                yield draft(
                    path, dataset=name, subject=finger_no, finger="unknown",
                    impression=int(impression), sensor=sensor, modality=modality, dpi=dpi,
                    dpi_source=source, license=license, split_unit=finger_no, split=split,
                    synthetic=modality == "synthetic", attributes={"set": set_name},
                )


# --- SOCOFing --------------------------------------------------------------------
# Shehu et al., arXiv 1807.10609: SecuGen Hamster plus (optical), 500 dpi capture.
# The released 96x103 images are downscaled (ridge period ~3 px vs ~9 px at 500 dpi),
# so the manifest dpi is the ridge-period estimate; the paper value is an attribute.
def index_socofing(data_root: Path, split_map: dict[str, str]) -> Iterator[Draft]:
    """SOCOFing real + altered prints; splits from the seed-0 subject split file."""

    root = data_root / "socofing" / "socofing"
    for record in load_socofing(root, include_altered=True):
        attrs = record.attributes
        variant = f"altered:{attrs['alteration']}:{attrs['level']}" if attrs["alteration"] else ""
        yield draft(
            record.path, dataset="SOCOFing", subject=record.subject,
            finger=canonical_finger(attrs["hand"], attrs["finger"]),
            impression=record.impression, sensor="SecuGen Hamster plus",
            modality="contact_optical", dpi=500, dpi_source="estimate",
            license="CC-BY-NC-SA-4.0", split_unit=record.subject,
            split=split_map.get(record.subject), variant=variant,
            attributes={"gender": attrs["gender"], "shipped_split": attrs["shipped_split"],
                        "capture_dpi_paper": 500},
        )


# --- Tsinghua TDFD ---------------------------------------------------------------
def index_tdfd(data_root: Path) -> Iterator[Draft]:
    """TDFD: folder = one normal/distorted pair; 185 fingers over 320 pairs.

    The pair -> finger mapping is not published, so different folders may hold
    the same finger.  To rule out leakage the whole set goes to ``train``.
    Frame ``000001`` is the normal print (impression 1); the other frame is
    the distorted one (impression 2).  FTIR scanner, 500 dpi (TDFD page).
    """

    root = data_root / "tdfd" / "Tsinghua Distorted Fingerprint Database"
    for folder in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        frames = sorted(folder.glob("*.png"))
        for impression, path in enumerate(frames, 1):
            yield draft(
                path, dataset="TDFD", subject=folder.name, finger="unknown",
                impression=impression, sensor="FTIR video scanner", modality="contact_optical",
                dpi=500, dpi_source="dataset page", license=DATASETS["tdfd"].license,
                split_unit="all", split="train",
                variant="" if impression == 1 else "distorted",
                attributes={"frame": path.stem, "note": "folders may share a finger"},
            )


# --- Neurotechnology samples -----------------------------------------------------
_NEURO_FILE = re.compile(r"^(\d+)_(\d+)_(\d+)\.tif$", re.I)


def index_neurotech(data_root: Path) -> Iterator[Draft]:
    """Cross Match Verifier 300 + U.are.U 4000 samples (500 ppi, Neurotechnology page).

    Files are ``person_finger_scan.tif``.  The two sets share person ids; the
    contact sheet confirms whether the same person/finger id is the same
    finger on both sensors (see docs/DATASETS.md).  Too small to split by
    subject usefully, so it is a cross-sensor ``val`` set.
    """

    sensors = {"crossmatch": "Cross Match Verifier 300", "uareu": "DigitalPersona U.are.U 4000"}
    for folder, sensor in sensors.items():
        root = data_root / "neurotech" / folder
        for path in sorted(root.glob("*.tif")):
            match = _NEURO_FILE.match(path.name)
            if not match:
                continue
            person, finger, scan = match.groups()
            yield draft(
                path, dataset="neurotech", subject=str(int(person)), finger=f"finger_{int(finger)}",
                impression=int(scan),
                sensor=sensor, modality="contact_optical", dpi=500,
                dpi_source="vendor page", license=DATASETS["neurotech"].license,
                split_unit=person, split="val", attributes={"scan": int(scan)},
            )


# --- Mendeley FAMILY -------------------------------------------------------------
_FAMILY_FILE = re.compile(r"^FM(\d+)_([CFM])(\d)\.png$", re.I)


def index_mendeley_family(data_root: Path) -> Iterator[Draft]:
    """100 families x {father, mother, child} x 5 right-hand fingers, 1 impression each.

    The finger order of ``_1.._5`` is not documented, so fingers are
    ``finger_1..5``.  Split unit = family (relatives stay together).
    Sensor and resolution are not documented; dpi is estimated later.
    """

    root = data_root / "mendeley_family" / "FAMILY FINGERPRINT DATASET"
    for path in sorted(root.rglob("*.png")):
        match = _FAMILY_FILE.match(path.name)
        if not match:
            continue
        family, role, finger = match.groups()
        yield draft(
            path, dataset="mendeley_family", subject=f"{int(family)}{role.upper()}",
            finger=f"finger_{finger}", impression=1, sensor="unknown",
            modality="contact_unknown", dpi=500, dpi_source="estimate",
            license=DATASETS["mendeley_family"].license, split_unit=str(int(family)),
            attributes={"family": int(family), "role": role.upper()},
        )


INDEXERS = ("fvc", "socofing", "tdfd", "neurotech", "mendeley_family")
