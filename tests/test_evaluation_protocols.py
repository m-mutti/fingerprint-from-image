from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_from_image.evaluation.datasets import (
    load_fvc_set,
    load_socofing,
    parse_socofing_name,
)
from fingerprint_from_image.evaluation.protocols import (
    check_subject_overlap,
    fvc_pairs,
    socofing_subject_split,
    split_subjects,
)


def test_fvc_pairs_counts_match_the_official_protocol() -> None:
    db_b = fvc_pairs(10, 8, "fvc")
    assert db_b.genuine.shape == (10 * comb(8, 2), 2) == (280, 2)
    assert db_b.impostor.shape == (comb(10, 2), 2) == (45, 2)
    db_a = fvc_pairs(100, 8, "fvc")
    assert len(db_a.genuine) == 2800 and len(db_a.impostor) == 4950
    every = fvc_pairs(10, 8, "all")
    assert len(every.impostor) == comb(80, 2) - 280 == 2880
    assert db_b.n_samples == 80


def test_fvc_pairs_structure() -> None:
    pairs = fvc_pairs(4, 3, "all")
    m = pairs.n_impressions
    assert (pairs.genuine[:, 0] < pairs.genuine[:, 1]).all()
    assert (pairs.genuine[:, 0] // m == pairs.genuine[:, 1] // m).all()
    assert (pairs.impostor[:, 0] // m != pairs.impostor[:, 1] // m).all()
    assert (pairs.impostor[:, 0] < pairs.impostor[:, 1]).all()
    # Unordered and unique: genuine + impostor cover every pair exactly once.
    rows = {tuple(r) for r in pairs.all_pairs().tolist()}
    assert len(rows) == len(pairs.all_pairs()) == comb(12, 2)
    fvc = fvc_pairs(4, 3, "fvc")
    assert (fvc.impostor % m == 0).all()
    assert {tuple(r) for r in fvc.impostor.tolist()} <= rows
    with pytest.raises(ValueError):
        fvc_pairs(0, 8)
    with pytest.raises(ValueError):
        fvc_pairs(2, 2, "some")  # type: ignore[arg-type]


def test_split_subjects_is_disjoint_complete_and_deterministic() -> None:
    subjects = [str(i) for i in range(1, 601)]
    split = split_subjects(subjects, seed=7)
    assert {k: len(v) for k, v in split.splits.items()} == {"train": 480, "val": 60, "test": 60}
    assert not any(check_subject_overlap(split.splits).values())
    assert sorted(sum(map(list, split.splits.values()), []), key=int) == subjects
    assert split == split_subjects(reversed(subjects), seed=7)
    assert split.splits != split_subjects(subjects, seed=8).splits
    assert split.split_of(split.splits["val"][0]) == "val"
    with pytest.raises(ValueError):
        split_subjects([])


def _touch_bmp(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(path)


def test_parse_socofing_names() -> None:
    real = parse_socofing_name("100__M_Left_index_finger.BMP")
    assert real == {
        "subject": "100",
        "gender": "M",
        "hand": "left",
        "finger": "index",
        "label": "left_index",
        "alteration": "",
    }
    altered = parse_socofing_name("7__F_Right_little_finger_Zcut.BMP")
    assert altered is not None and altered["alteration"] == "Zcut"
    assert parse_socofing_name("notes.txt") is None


def test_socofing_resplit_when_shipped_folders_share_subjects(tmp_path: Path) -> None:
    root = tmp_path / "socofing"
    for split, subject in (("train", 1), ("train", 2), ("val", 1), ("test", 3), ("train", 4)):
        _touch_bmp(root / split / "Real" / f"{subject}__M_Left_index_finger.BMP")
    _touch_bmp(root / "train" / "Altered" / "Altered-Hard" / "2__M_Left_index_finger_CR.BMP")
    _touch_bmp(root / ".git" / "objects" / "1__M_Left_index_finger.BMP")  # must be ignored
    records = load_socofing(root)
    assert len(records) == 6
    altered = [r for r in records if r.attributes["alteration"]]
    assert altered[0].impression == 1 + 3 * 2 + 0 and altered[0].attributes["level"] == "Hard"
    assert load_socofing(root, include_altered=False)[0].finger_id == "1_left_index"

    manifest_path = tmp_path / "split.json"
    halves = (("a", 0.5), ("b", 0.5))
    manifest = socofing_subject_split(root, manifest_path, seed=0, fractions=halves)
    assert manifest["resplit"] is True
    assert manifest["shipped_overlap"]["train&val"] == 1
    assert not any(check_subject_overlap(manifest["subjects"]).values())
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {f["path"] for f in on_disk["files"]} >= {"train/Real/1__M_Left_index_finger.BMP"}
    by_subject = {}
    for item in on_disk["files"]:
        by_subject.setdefault(item["subject"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in by_subject.values())


def test_socofing_keeps_disjoint_shipped_split(tmp_path: Path) -> None:
    root = tmp_path / "s"
    _touch_bmp(root / "train" / "Real" / "1__M_Left_index_finger.BMP")
    _touch_bmp(root / "test" / "Real" / "2__F_Right_thumb_finger.BMP")
    manifest = socofing_subject_split(root, include_files=False)
    assert manifest["resplit"] is False
    assert manifest["subjects"] == {"train": ["1"], "test": ["2"]}
    assert "files" not in manifest


def test_load_fvc_set_orders_and_validates(tmp_path: Path) -> None:
    folder = tmp_path / "FVC2002_DB1_B"
    for finger in (102, 101):
        for impression in (2, 1):
            _touch_bmp(folder / f"{finger}_{impression}.tif")
    records = load_fvc_set(folder)
    assert [(r.finger_id, r.impression) for r in records] == [
        ("FVC2002_DB1_B/101", 1),
        ("FVC2002_DB1_B/101", 2),
        ("FVC2002_DB1_B/102", 1),
        ("FVC2002_DB1_B/102", 2),
    ]
    (folder / "102_2.tif").unlink()
    with pytest.raises(ValueError):
        load_fvc_set(folder)
    with pytest.raises(FileNotFoundError):
        load_fvc_set(tmp_path / "missing")


def test_evaluation_modules_do_not_import_heavy_runtimes() -> None:
    import ast

    import fingerprint_from_image.evaluation as evaluation

    package = Path(evaluation.__file__).parent
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in {"torch", "sklearn", "scipy"}, (source.name, name)
                if root == "cv2":
                    # Only the lazily imported classic baseline may touch OpenCV.
                    assert source.name == "scorers.py" and node.col_offset > 0, source.name
