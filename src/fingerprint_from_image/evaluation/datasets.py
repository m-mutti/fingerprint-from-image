"""Dataset indexers for FVC2000/2002/2004 and SOCOFing.

Loaders return lightweight :class:`FingerprintRecord` objects (no pixels);
:func:`load_images` decodes them with the OpenCV-free
:func:`fingerprint_from_image.io.load_image`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io import load_image

__all__ = [
    "FVC_DPI",
    "FVC_SETS",
    "FingerprintRecord",
    "fvc_set_dir",
    "load_fvc_set",
    "load_images",
    "load_socofing",
    "parse_socofing_name",
]

FVC_SETS: tuple[str, ...] = tuple(
    f"FVC{year}_DB{db}_B" for year in (2000, 2002, 2004) for db in (1, 2, 3, 4)
)

# Nominal resolution per database (sets A and B share the sensor), from the
# official BioLab pages bias.csr.unibo.it/fvc{2000,2002,2004}/databases.asp.
# The synthetic SFinGe databases (DB4) are documented as "about 500 dpi".
FVC_DPI: dict[str, int] = {
    "FVC2000_DB1": 500,
    "FVC2000_DB2": 500,
    "FVC2000_DB3": 500,
    "FVC2000_DB4": 500,
    "FVC2002_DB1": 500,
    "FVC2002_DB2": 569,
    "FVC2002_DB3": 500,
    "FVC2002_DB4": 500,
    "FVC2004_DB1": 500,
    "FVC2004_DB2": 500,
    "FVC2004_DB3": 512,
    "FVC2004_DB4": 500,
}

_FVC_NAME = re.compile(r"^(?P<finger>\d+)_(?P<impression>\d+)\.(?:tif|tiff|bmp|png)$", re.I)
_SOCOFING_NAME = re.compile(
    r"^(?P<subject>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_"
    r"(?P<finger>thumb|index|middle|ring|little)_finger"
    r"(?:_(?P<alteration>CR|Obl|Zcut))?\.bmp$",
    re.I,
)
_LEVELS = ("Easy", "Medium", "Hard")
_ALTERATIONS = ("CR", "Obl", "Zcut")


@dataclass(frozen=True)
class FingerprintRecord:
    """One image on disk and its identity labels.

    ``finger_id`` identifies the physical finger (the unit of genuine pairs);
    ``subject`` the person (the unit of disjoint splits; for FVC, where
    fingers are not linked to people, it equals ``finger_id``).
    ``impression`` is 1-based for FVC; for SOCOFing 0 is the real print and
    1..9 enumerate ``(level, alteration)`` in Easy/Medium/Hard x CR/Obl/Zcut.
    """

    path: Path
    finger_id: str
    subject: str
    impression: int
    dataset: str
    attributes: dict[str, str] = field(default_factory=dict, compare=False)


def fvc_set_dir(data_root: str | Path, name: str) -> Path:
    """Directory of an FVC set, e.g. ``fvc_set_dir("data", "FVC2004_DB1_B")``."""

    root = Path(data_root)
    for candidate in (root / "fvc" / name, root / name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"FVC set {name!r} not found under {root} (or {root / 'fvc'})")


def load_fvc_set(directory: str | Path, name: str | None = None) -> list[FingerprintRecord]:
    """Index one FVC set directory of ``{finger}_{impression}.tif`` files.

    Records are sorted by finger then impression, so record ``k`` is flat
    sample index ``k`` of :func:`~fingerprint_from_image.evaluation.protocols.fvc_pairs`.
    Raises ``ValueError`` if fingers have different impression counts or the
    impressions are not ``1..n``.
    """

    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"FVC directory does not exist: {folder}")
    set_name = name or folder.name
    found: dict[int, dict[int, Path]] = {}
    for path in folder.iterdir():
        match = _FVC_NAME.match(path.name)
        if match and path.is_file():
            found.setdefault(int(match["finger"]), {})[int(match["impression"])] = path
    if not found:
        raise ValueError(f"no FVC images ({{finger}}_{{impression}}.tif) in {folder}")
    counts = {len(impressions) for impressions in found.values()}
    if len(counts) != 1:
        raise ValueError(f"fingers in {folder} have unequal impression counts {sorted(counts)}")
    n_impressions = counts.pop()
    records: list[FingerprintRecord] = []
    for finger in sorted(found):
        impressions = found[finger]
        if sorted(impressions) != list(range(1, n_impressions + 1)):
            raise ValueError(f"finger {finger} in {folder} lacks impressions 1..{n_impressions}")
        finger_id = f"{set_name}/{finger}"
        for impression in range(1, n_impressions + 1):
            records.append(
                FingerprintRecord(
                    path=impressions[impression],
                    finger_id=finger_id,
                    subject=finger_id,
                    impression=impression,
                    dataset=set_name,
                )
            )
    return records


def parse_socofing_name(name: str) -> dict[str, str] | None:
    """Parse ``100__M_Left_index_finger_CR.BMP``; return ``None`` if it does not match."""

    match = _SOCOFING_NAME.match(name)
    if match is None:
        return None
    hand = match["hand"].lower()
    finger = match["finger"].lower()
    alteration = match["alteration"] or ""
    canonical = {item.lower(): item for item in _ALTERATIONS}
    return {
        "subject": str(int(match["subject"])),
        "gender": match["gender"].upper(),
        "hand": hand,
        "finger": finger,
        "label": f"{hand}_{finger}",
        "alteration": canonical.get(alteration.lower(), ""),
    }


def _socofing_level(path: Path) -> str:
    for part in reversed(path.parts):
        for level in _LEVELS:
            if part.lower() == f"altered-{level.lower()}":
                return level
    return ""


def _shipped_split(path: Path, root: Path) -> str:
    relative = path.relative_to(root).parts
    return relative[0] if len(relative) > 1 and relative[0] in ("train", "val", "test") else ""


def load_socofing(
    root: str | Path, *, include_altered: bool = True
) -> list[FingerprintRecord]:
    """Index SOCOFing images below ``root`` (any layout; ``.git`` is skipped).

    ``finger_id`` is ``"{subject}_{hand}_{finger}"`` (e.g. ``"100_left_index"``)
    and ``attributes`` holds ``gender``, ``hand``, ``finger``, ``label``
    (matching :data:`fingerprint_from_image.types.FINGER_LABELS`),
    ``alteration`` (``""`` for real prints), ``level`` and ``shipped_split``.
    Records are sorted by subject, finger id, impression, then path.
    """

    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"SOCOFing root does not exist: {base}")
    records: list[FingerprintRecord] = []
    for path in base.rglob("*"):
        if ".git" in path.relative_to(base).parts or not path.is_file():
            continue
        parsed = parse_socofing_name(path.name)
        if parsed is None:
            continue
        level = _socofing_level(path)
        alteration = parsed["alteration"]
        if alteration and not include_altered:
            continue
        impression = 0
        if alteration:
            level_index = _LEVELS.index(level) if level else 0
            impression = 1 + 3 * level_index + _ALTERATIONS.index(alteration)
        subject = parsed.pop("subject")
        records.append(
            FingerprintRecord(
                path=path,
                finger_id=f"{subject}_{parsed['label']}",
                subject=subject,
                impression=impression,
                dataset="SOCOFing",
                attributes={**parsed, "level": level, "shipped_split": _shipped_split(path, base)},
            )
        )
    records.sort(key=lambda r: (int(r.subject), r.finger_id, r.impression, str(r.path)))
    return records


def load_images(records: Sequence[FingerprintRecord]) -> list[np.ndarray]:
    """Decode records to ``(H, W) uint8`` grayscale arrays via Pillow."""

    return [load_image(record.path) for record in records]
