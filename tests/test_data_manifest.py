"""Manifest schema, split leakage checks, dedup and loader on tiny synthetic fixtures."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_from_image.data import (
    ManifestDataset,
    ManifestError,
    assign_subject_splits,
    canonical_finger,
    check_splits,
    make_identity,
    read_manifest,
    validate_record,
    write_manifest,
)
from fingerprint_from_image.data.archives import (
    UnsafeArchiveError,
    is_safe_member,
    safe_extract_zip,
)
from fingerprint_from_image.data.contact_sheet import render_contact_sheet
from fingerprint_from_image.data.dedup import find_near_duplicate_pairs, resolve_duplicates
from fingerprint_from_image.data.quality import (
    estimate_ridge_period,
    fine_similarity,
    near_dup_descriptor,
    probe_image,
)


def _ridges(seed: int, size: int = 160, period: float = 9.0) -> np.ndarray:
    """A synthetic ridge pattern (curved sinusoid) on a white canvas."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    bend = rng.uniform(-0.004, 0.004)
    phase = (np.cos(angle) * xx + np.sin(angle) * yy + bend * (xx - size / 2) ** 2)
    image = 128 + 100 * np.sin(2 * np.pi * phase / period + rng.uniform(0, 6.28))
    image += rng.normal(0, 6, image.shape)
    mask = (xx - size / 2) ** 2 / (0.42 * size) ** 2 + (yy - size / 2) ** 2 / (0.47 * size) ** 2
    image[mask > 1] = 255
    return np.clip(image, 0, 255).astype(np.uint8)


def _row(path: Path, dataset: str, subject: str, impression: int, split: str, **extra) -> dict:
    image = np.asarray(Image.open(path))
    row = {
        "path": str(path), "dataset": dataset, "subject": subject, "finger": "unknown",
        "identity": make_identity(dataset, subject, "unknown"), "impression": impression,
        "sensor": "test", "modality": "synthetic", "dpi": 500, "dpi_source": "test",
        "width": int(image.shape[1]), "height": int(image.shape[0]), "split": split,
        "license": "CC0-1.0", "sha256": probe_image(path).sha256,
    }
    row.update(extra)
    return row


@pytest.fixture()
def tiny_set(tmp_path: Path) -> list[dict]:
    rows = []
    for subject in range(6):
        for impression in range(3):
            path = tmp_path / f"s{subject}_{impression}.png"
            Image.fromarray(_ridges(subject * 10 + impression)).save(path)
            split = ("train", "train", "train", "train", "val", "test")[subject]
            rows.append(_row(path, "toy", str(subject), impression, split))
    return rows


def test_valid_row_passes_and_roundtrips(tiny_set: list[dict], tmp_path: Path) -> None:
    for row in tiny_set:
        validate_record(row)
    target = tmp_path / "m" / "toy.jsonl"
    assert write_manifest(tiny_set, target) == len(tiny_set)
    assert list(read_manifest(target, validate=True)) == tiny_set


@pytest.mark.parametrize(
    ("field", "value"),
    [("modality", "photo"), ("split", "dev"), ("finger", "left_toe"), ("identity", "x:y:z"),
     ("dpi", 0), ("impression", "1"), ("sha256", "abc"), ("bogus", 1)],
)
def test_invalid_rows_are_rejected(tiny_set: list[dict], field: str, value: object) -> None:
    row = dict(tiny_set[0], **{field: value})
    with pytest.raises(ManifestError):
        validate_record(row)


def test_missing_field_and_local_finger_labels(tiny_set: list[dict]) -> None:
    row = dict(tiny_set[0])
    del row["sha256"]
    with pytest.raises(ManifestError):
        validate_record(row)
    local = dict(tiny_set[0], finger="finger_3", identity=make_identity("toy", "0", "finger_3"))
    validate_record(local)
    assert canonical_finger("Right", "Index") == "right_index"
    assert canonical_finger(position=3) == "right_middle"
    assert canonical_finger(position=12) == "unknown"


def test_subject_split_is_deterministic_and_complete() -> None:
    subjects = [str(i) for i in range(100)]
    first = assign_subject_splits(subjects, seed=0)
    assert first == assign_subject_splits(reversed(subjects), seed=0)
    assert first != assign_subject_splits(subjects, seed=1)
    counts = {s: list(first.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 80, "val": 10, "test": 10}
    assert set(assign_subject_splits(["a", "b", "c"]).values()) == {"train", "val", "test"}


def test_check_splits_detects_leaks(tiny_set: list[dict]) -> None:
    assert check_splits(tiny_set) == []
    leaked = [dict(r) for r in tiny_set]
    leaked[0]["split"] = "test"  # one impression of subject 0 moved
    problems = check_splits(leaked)
    assert any("identity toy:0:unknown" in p for p in problems)
    assert any("subject toy:0" in p for p in problems)
    fvc = dict(tiny_set[0], dataset="FVC2004_DB1", identity="FVC2004_DB1:0:unknown")
    assert any("FVC2004" in p for p in check_splits([fvc]))
    leaked[0]["exclude"] = True  # excluded rows are ignored
    assert check_splits(leaked) == []


def test_duplicates_across_splits_keep_the_protected_copy(tiny_set: list[dict],
                                                          tmp_path: Path) -> None:
    copy = tmp_path / "copy_of_test.png"
    copy.write_bytes(Path(tiny_set[15]["path"]).read_bytes())  # subject 5 (test)
    rows = [dict(r) for r in tiny_set] + [_row(copy, "other", "x", 0, "train")]
    heldout = tmp_path / "heldout_copy.png"
    Image.fromarray(_ridges(0)).save(heldout)  # same pixels as s0_0 (train)
    rows.append(_row(heldout, "FVC2004_DB1", "1", 1, "heldout"))
    stats = resolve_duplicates(rows)
    assert stats["excluded_cross_identity"] == 2
    assert rows[-2]["exclude"] and "(test)" in rows[-2]["exclude_reason"]
    assert rows[0]["exclude"] and "(heldout)" in rows[0]["exclude_reason"]
    assert not rows[-1].get("exclude") and not rows[15].get("exclude")
    assert rows[-1]["duplicate_group"] == rows[0]["duplicate_group"]
    assert check_splits(rows) == []


def test_near_duplicate_search_and_descriptors(tiny_set: list[dict]) -> None:
    images = [np.asarray(Image.open(r["path"])) for r in tiny_set]
    shifted = np.roll(images[0], 1, axis=1)
    descriptors = np.stack([near_dup_descriptor(i) for i in images + [shifted]])
    identities = [r["identity"] for r in tiny_set] + ["toy:other:unknown"]
    pairs = find_near_duplicate_pairs(descriptors[:, :4096], identities, threshold=0.6)
    assert [(i, j) for i, j, _ in pairs] == [(0, len(images))]
    assert fine_similarity(images[0], shifted) > 0.8
    assert abs(fine_similarity(images[0], images[3])) < 0.4


def test_probe_flags_and_ridge_period(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert not probe_image(bad).ok
    flat = tmp_path / "flat.png"
    Image.new("L", (64, 64), 200).save(flat)
    assert not probe_image(flat).ok
    deep = tmp_path / "deep.png"
    Image.fromarray(_ridges(1).astype(np.uint16) * 257).save(deep)
    probe = probe_image(deep, ridge_period=True)
    assert probe.ok and probe.mode.startswith("I")
    assert probe.ridge_period is not None and 7.0 < probe.ridge_period < 11.5
    assert estimate_ridge_period(np.full((200, 200), 128, np.uint8)) is None


def test_loader_filters_labels_and_pk_batches(tiny_set: list[dict], tmp_path: Path) -> None:
    manifest = tmp_path / "toy.jsonl"
    rows = [dict(r) for r in tiny_set]
    rows[1]["exclude"] = True
    write_manifest(rows, manifest)
    train = ManifestDataset.from_files(manifest, splits={"train"})
    assert len(train) == 11 and train.num_identities == 4
    image, label = train[0]
    assert image.dtype == np.uint8 and image.ndim == 2 and label == 0
    batches = list(train.pk_batches(p=2, k=3, seed=0))
    assert len(batches) == 2
    for batch in batches:
        labels = train.labels[batch]
        assert len(set(labels.tolist())) == 2 and len(batch) == 6
    assert len(ManifestDataset.from_files(manifest, splits={"val", "test"})) == 6
    with pytest.raises(FileNotFoundError):
        ManifestDataset.from_files(tmp_path / "missing*.jsonl")


def test_safe_zip_extraction(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("set/a.png", b"png")
        archive.writestr("set/run.exe", b"MZ")
        archive.writestr("__MACOSX/set/._a.png", b"junk")
    target = tmp_path / "out"
    assert safe_extract_zip(good, target) == 1
    assert (target / "set" / "a.png").is_file() and not (target / "set" / "run.exe").exists()
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../escape.png", b"x")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.png").exists()
    assert not is_safe_member("C:/x.png") and not is_safe_member("/etc/x.png")
    assert is_safe_member("a/b.png")


def test_contact_sheet_renders(tiny_set: list[dict], tmp_path: Path) -> None:
    out = render_contact_sheet(tiny_set, tmp_path / "sheet.png", identities=3, impressions=3,
                               cell=48)
    with Image.open(out) as sheet:
        assert sheet.width == 3 * 48 and sheet.height > 3 * 48


def test_synthetic_generator_is_deterministic() -> None:
    from fingerprint_from_image.data.synth import make_master, render_impression

    first = make_master(3, shape=(96, 80), iterations=3)
    again = make_master(3, shape=(96, 80), iterations=3)
    other = make_master(4, shape=(96, 80), iterations=3)
    assert np.array_equal(first.ridges, again.ridges)
    assert not np.array_equal(first.ridges, other.ridges)
    impression = render_impression(first, seed=7)
    assert impression.shape == (96, 80) and impression.dtype == np.uint8
    assert np.array_equal(impression, render_impression(again, seed=7))
    assert not np.array_equal(impression, render_impression(first, seed=8))
