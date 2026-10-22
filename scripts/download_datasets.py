"""Download and extract the public benchmark datasets into the git-ignored ``data/``.

Idempotent: archives already present with the expected SHA-256 are not
re-downloaded, and sets already extracted are skipped.  Standard library only.

Layout produced (what ``fingerprint-bench --data-root data`` expects)::

    data/raw/fvc/FVC{year}_DB{n}_B.zip      # original archives
    data/fvc/FVC{year}_DB{n}_B/{101..110}_{1..8}.tif
    data/raw/socofing/socofing.zip           # only with --socofing
    data/socofing/socofing/{train,val,test}/{Real,Altered/Altered-*}/*.BMP

FVC2000/2002/2004 DB_B (10 fingers x 8 impressions each) are the sets the
University of Bologna BioLab distributes freely from
``http://bias.csr.unibo.it/fvc{year}/Downloads/DB{n}_B.zip`` (research use;
the full DB_A sets ship only with the *Handbook of Fingerprint Recognition*).

SOCOFing (Shehu et al., 2018; 600 subjects x 10 fingers, real + altered) is
licensed **CC BY-NC-SA 4.0**: models trained on it are non-commercial.  The
original distribution is on Kaggle (``ruizgara/socofing``, login required); this
script fetches the community mirror ``ThanhQuy78/socofing`` on Hugging Face,
pinned to a revision and verified by SHA-256, only when ``--socofing`` and
``--accept-socofing-license`` are both given.  The mirror's train/val/test
folders are **not** subject-disjoint; build a subject-disjoint split with
``fingerprint_from_image.evaluation.socofing_subject_split``.

Usage::

    python scripts/download_datasets.py                       # all 12 FVC DB_B sets
    python scripts/download_datasets.py --sets FVC2004_DB1_B
    python scripts/download_datasets.py --socofing --accept-socofing-license
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger("download_datasets")


@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    sha256: str
    size: int | None = None


# SHA-256 computed from archives downloaded from bias.csr.unibo.it on 2026-10-22.
_FVC_SHA256 = {
    "FVC2000_DB1_B": "089e26e80e4276d7b52558722fcd3826fe4aaf4cacbcb82456977d556f0ed61a",
    "FVC2000_DB2_B": "e0331950a9860296afc01b19e8e5f70cb365e050aa3ec3e05832cd56f8cad8d2",
    "FVC2000_DB3_B": "d6dfa420d004b71e64c9847204b665235b9e8eba84c8a837af4f577a4a60051f",
    "FVC2000_DB4_B": "b757ca301a5c342301313f27bdbc661db8acfba20a76b308e48018090213f350",
    "FVC2002_DB1_B": "c039e8841bf847b7e8f036f358b76516b4c4548a9b24099a7ad3226dec441f88",
    "FVC2002_DB2_B": "47358bcace8819822d898f4784b829a219b35c3a7ecbfb4c9541ca3e6ed39937",
    "FVC2002_DB3_B": "a53c9658ac269c2aefeb5579a8d8a0815e09eabec8acc76641879bdd331ac50a",
    "FVC2002_DB4_B": "ac8cb3a13649352719b853428e4e7b51187300f69d588753fc751f2ffc295c33",
    "FVC2004_DB1_B": "dc41716d6c858600d856cda46f8eb4d990858b8a5b1f6ae107db0245297259f3",
    "FVC2004_DB2_B": "110c7baa1c429067fc363188c20c16366635edf965c7a67aac00419f893e8a58",
    "FVC2004_DB3_B": "865ec31abfa8a4d9ae86256d841d44e78d03723cdf4e50b4c45ad00660336924",
    "FVC2004_DB4_B": "705c209c00c02c98fd0b9eae39930ea77def552faada1b2432fa52e5c23d3a99",
}

FVC_ARCHIVES: dict[str, Archive] = {
    name: Archive(
        name=name,
        url=f"http://bias.csr.unibo.it/fvc{name[3:7]}/Downloads/{name[-5:]}.zip",
        sha256=digest,
    )
    for name, digest in _FVC_SHA256.items()
}

SOCOFING_REVISION = "564d269d55ec4cd07e4504b2549dae01bfa8bb94"
SOCOFING_ARCHIVE = Archive(
    name="socofing",
    url=(
        "https://huggingface.co/datasets/ThanhQuy78/socofing/resolve/"
        f"{SOCOFING_REVISION}/socofing.zip"
    ),
    sha256="824fb207169e0f2bcc899a327dfada2db78e13493643857f45a7093bdce66237",
    size=897_932_447,
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(archive: Archive, destination: Path) -> Path:
    """Download ``archive`` to ``destination`` unless a verified copy already exists."""

    if destination.is_file():
        if sha256_of(destination) == archive.sha256:
            logger.info("ok      %s (checksum verified)", destination)
            return destination
        logger.warning("corrupt %s (checksum mismatch); downloading again", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetch   %s", archive.url)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(archive.url, timeout=120) as response:  # noqa: S310
                shutil.copyfileobj(response, handle, length=1 << 20)
        except OSError:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    actual = sha256_of(temporary)
    if actual != archive.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{archive.url}: SHA-256 {actual} != expected {archive.sha256}")
    temporary.replace(destination)
    return destination


def safe_extract(
    zip_path: Path, target: Path, *, skip_parts: frozenset[str] = frozenset()
) -> int:
    """Extract members (rejecting absolute/``..`` paths); return the number written."""

    written = 0
    root = target.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if member.is_dir() or not parts or skip_parts.intersection(parts):
                continue
            output = (target / Path(*parts)).resolve()
            if member.filename.startswith(("/", "\\")) or root not in output.parents:
                raise RuntimeError(f"unsafe path in {zip_path}: {member.filename!r}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            written += 1
    return written


def install_fvc(data_root: Path, names: list[str], *, extract: bool = True) -> None:
    for name in names:
        archive = FVC_ARCHIVES[name]
        zip_path = fetch(archive, data_root / "raw" / "fvc" / f"{name}.zip")
        if not extract:
            continue
        target = data_root / "fvc" / name
        if target.is_dir() and len(list(target.glob("*.tif"))) == 80:
            logger.info("ok      %s (already extracted)", target)
            continue
        count = safe_extract(zip_path, target)
        logger.info("extract %s (%d files)", target, count)


def install_socofing(data_root: Path, *, extract: bool = True) -> None:
    zip_path = fetch(SOCOFING_ARCHIVE, data_root / "raw" / "socofing" / "socofing.zip")
    if not extract:
        return
    marker = data_root / "socofing" / "socofing" / "train" / "Real"
    if marker.is_dir() and any(marker.iterdir()):
        logger.info("ok      %s (already extracted)", marker.parent.parent)
        return
    # The mirror's archive embeds its own .git directory; it is not needed.
    count = safe_extract(zip_path, data_root / "socofing", skip_parts=frozenset({".git"}))
    logger.info("extract %s (%d files)", data_root / "socofing", count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch FVC DB_B and (optionally) SOCOFing.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--sets", default="all", help="comma-separated FVC sets, or 'all'")
    parser.add_argument("--no-fvc", action="store_true", help="skip the FVC sets")
    parser.add_argument("--socofing", action="store_true", help="also fetch SOCOFing (~900 MB)")
    parser.add_argument(
        "--accept-socofing-license",
        action="store_true",
        help="confirm SOCOFing is used under CC BY-NC-SA 4.0 (non-commercial)",
    )
    parser.add_argument("--no-extract", action="store_true", help="download and verify only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    names = list(FVC_ARCHIVES) if args.sets == "all" else args.sets.split(",")
    unknown = sorted(set(names) - set(FVC_ARCHIVES))
    if unknown:
        parser.error(f"unknown FVC set(s): {unknown}")
    if args.socofing and not args.accept_socofing_license:
        parser.error("SOCOFing is CC BY-NC-SA 4.0; pass --accept-socofing-license to proceed")
    try:
        if not args.no_fvc:
            install_fvc(args.data_root, names, extract=not args.no_extract)
        if args.socofing:
            install_socofing(args.data_root, extract=not args.no_extract)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        logger.error("error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
