"""Filename parsing of the dataset indexers on fake directory trees (no real data)."""

from __future__ import annotations

from pathlib import Path

from fingerprint_from_image.data.indexers import (
    index_fvc,
    index_naijafingset,
    index_neurotech,
    index_tdfd,
)
from fingerprint_from_image.data.indexers_extra import (
    index_impose_syn,
    index_l3sf,
    index_unsw_2d3d,
)


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_fvc_sets_and_heldout(tmp_path: Path) -> None:
    dvd = "fvc_full/74034_3_En_4_MOESM1_ESM"
    _touch(tmp_path, f"{dvd}/FVC2002/Dbs/Db2_a/1_1.tif", f"{dvd}/FVC2004/Dbs/Db3_a/7_2.tif",
           "fvc/FVC2002_DB2_B/101_1.tif", f"{dvd}/FVC2002/Dbs/Db2_a/index_a.MFA")
    rows = {Path(r["path"]).name + r["dataset"]: r for r in index_fvc(tmp_path)}
    assert len(rows) == 3
    a = rows["1_1.tifFVC2002_DB2"]
    assert (a["split"], a["dpi"], a["modality"], a["identity"]) == (
        "train", 569, "contact_optical", "FVC2002_DB2:1:unknown")
    assert rows["101_1.tifFVC2002_DB2"]["split"] == "val"
    held = rows["7_2.tifFVC2004_DB3"]
    assert held["split"] == "heldout" and held["modality"] == "contact_thermal"


def test_naija_neurotech_tdfd(tmp_path: Path) -> None:
    _touch(tmp_path, "naijafingset/x/FingerPrints/1000_LI.jpg",
           "naijafingset/x/FingerPrints/977_RT(1).jpg", "naijafingset/x/notes.jpg",
           "neurotech/crossmatch/012_3_2.tif", "neurotech/uareu/012_3_1.tif",
           "tdfd/Tsinghua Distorted Fingerprint Database/5/000001.png",
           "tdfd/Tsinghua Distorted Fingerprint Database/5/000088.png")
    naija = sorted(index_naijafingset(tmp_path), key=lambda r: r["subject"])
    assert [(r["subject"], r["finger"], r["impression"]) for r in naija] == [
        ("1000", "left_index", 1), ("977", "right_thumb", 2)]
    neuro = list(index_neurotech(tmp_path))
    assert {r["identity"] for r in neuro} == {"neurotech:12:finger_3"}
    assert {r["sensor"] for r in neuro} == {"Cross Match Verifier 300",
                                            "DigitalPersona U.are.U 4000"}
    tdfd = list(index_tdfd(tmp_path))
    assert [r["attributes"]["distorted"] for r in tdfd] == [False, True]
    assert {r["split"] for r in tdfd} == {"train"}


def test_unsw_impose_l3sf(tmp_path: Path) -> None:
    _touch(tmp_path, "unsw_2d3d/2D Database/102_3_4.bmp", "unsw_2d3d/DS1/102/SIRE-102_3_2.bmp",
           "unsw_2d3d/DS1/102/SIRE-102_3_2_HT1.bmp", "unsw_2d3d/DS1/102/raw/SIRE-102_3_2.bmp",
           "impose_syn/gallery/00007.png", "impose_syn/query/00007_-12.png",
           "impose_syn/query/00007_30.png", "l3sf/L3SF_V2/L3-SF/R2/14_2_3.png")
    unsw = list(index_unsw_2d3d(tmp_path))
    assert {(r["modality"], r["impression"]) for r in unsw} == {("contact_optical", 4),
                                                               ("contactless", 12)}
    assert {r["identity"] for r in unsw} == {"unsw_2d3d:102:right_middle"}
    impose = sorted(index_impose_syn(tmp_path), key=lambda r: r["impression"])
    assert [(r["impression"], r["attributes"].get("angle")) for r in impose] == [
        (0, None), (1, -12), (2, 30)]
    (l3,) = list(index_l3sf(tmp_path))
    assert (l3["identity"], l3["impression"]) == ("l3sf:14:finger_2", 8)
