from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from conftest import affine_variant, synthetic_fingerprint

from fingerprint_extraction import ExtractionConfig, FingerprintCrop
from fingerprint_matching import (
    APPLICATION_ID,
    FingerprintDatabaseError,
    FingerprintMatcher,
    FingerprintQualityError,
    FingerprintStore,
    FingerprintTemplate,
    MatcherConfig,
    create_template,
    match_templates,
)


def test_template_creation_is_deterministic_and_rejects_blank_images() -> None:
    image = synthetic_fingerprint(3)
    first = create_template(image)
    second = create_template(image)

    assert first.keypoint_count >= 8
    assert first.descriptors.shape == (first.keypoint_count, 128)
    assert first.descriptors.dtype == np.float32
    np.testing.assert_allclose(first.keypoints, second.keypoints)
    np.testing.assert_allclose(first.descriptors, second.descriptors)

    with pytest.raises(FingerprintQualityError):
        create_template(np.full((400, 400), 255, dtype=np.uint8))

    rng = np.random.default_rng(123)
    noise = rng.integers(0, 256, (520, 520), dtype=np.uint8)
    blurred_noise = cv.GaussianBlur(noise, (5, 5), 1.0)
    with pytest.raises(FingerprintQualityError, match="ridge flow"):
        create_template(blurred_noise)

    text = np.full((520, 520), 255, dtype=np.uint8)
    cv.putText(text, "FINGERPRINT", (10, 260), cv.FONT_HERSHEY_SIMPLEX, 2, 0, 5)
    with pytest.raises(FingerprintQualityError, match="region is too small"):
        create_template(text)

    distant_patch = np.full((600, 600), 255, dtype=np.uint8)
    small_print = cv.resize(
        synthetic_fingerprint(29), (130, 130), interpolation=cv.INTER_AREA
    )
    distant_patch[235:365, 235:365] = small_print
    with pytest.raises(FingerprintQualityError, match="region is too small"):
        create_template(distant_patch)


def test_repository_sample_panels_pass_ridge_validation() -> None:
    montage = cv.imread(
        str(Path(__file__).resolve().parents[1] / "sample_output.jpg"),
        cv.IMREAD_GRAYSCALE,
    )
    assert montage is not None
    panel_boxes = (
        (195, 135, 730, 315),
        (845, 128, 1380, 322),
        (195, 430, 730, 656),
        (845, 426, 1378, 573),
    )

    templates = [
        create_template(montage[y1:y2, x1:x2])
        for x1, y1, x2, y2 in panel_boxes
    ]

    assert all(template.keypoint_count >= 8 for template in templates)
    for index, template in enumerate(templates):
        variant = affine_variant(
            montage[
                panel_boxes[index][1] : panel_boxes[index][3],
                panel_boxes[index][0] : panel_boxes[index][2],
            ],
            angle=3.0,
            scale=0.98,
            tx=4.0,
            ty=-2.0,
            noise_seed=900 + index,
        )
        assert match_templates(create_template(variant), template).accepted
        assert all(
            not match_templates(template, other).accepted
            for other_index, other in enumerate(templates)
            if other_index != index
        )


def test_geometric_match_separates_variant_from_impostor() -> None:
    enrolled_image = synthetic_fingerprint(11)
    genuine_query = affine_variant(enrolled_image, noise_seed=501)
    impostor_query = synthetic_fingerprint(27)

    enrolled = create_template(enrolled_image)
    genuine = match_templates(create_template(genuine_query), enrolled)
    impostor = match_templates(create_template(impostor_query), enrolled)

    assert 0.0 <= impostor.score < genuine.score <= 1.0
    assert genuine.accepted
    assert genuine.inliers >= 4
    assert not impostor.accepted


def test_matching_is_resolution_independent() -> None:
    image = synthetic_fingerprint(19)
    smaller = cv.resize(image, (208, 208), interpolation=cv.INTER_AREA)

    result = match_templates(create_template(smaller), create_template(image))

    assert result.accepted
    assert result.inliers >= 6


def test_tiny_geometric_cluster_cannot_establish_a_match() -> None:
    points = np.asarray(
        [
            [500 + x, 500 + y]
            for x, y in (
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (2, 2),
            )
        ],
        dtype=np.float32,
    )
    descriptors = np.zeros((len(points), 128), dtype=np.float32)
    descriptors[np.arange(len(points)), np.arange(len(points))] = 1.0
    template = FingerprintTemplate(
        keypoints=points,
        descriptors=descriptors,
        image_shape=(1000, 1000),
        quality=1.0,
        roi_fraction=1.0,
    )

    result = match_templates(template, template)

    assert not result.accepted
    assert result.score == 0.0
    assert result.reason and "spatial coverage" in result.reason


def test_database_round_trip_incremental_enrollment_and_identification(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fingerprints.db"
    alice = synthetic_fingerprint(41)
    bob = synthetic_fingerprint(88)

    with FingerprintMatcher(database) as matcher:
        first = matcher.enroll("alice", alice, finger="right_index", kind="cropped")
        duplicate = matcher.enroll(
            "alice", alice, finger="right_index", kind="cropped"
        )
        second = matcher.enroll("bob", bob, finger="right_index", kind="cropped")
        assert (first.added, first.failed) == (1, 0)
        assert (duplicate.skipped, duplicate.failed) == (1, 0)
        assert (second.added, second.failed) == (1, 0)

    with FingerprintMatcher(database) as reopened:
        assert reopened.stats()["identities"] == 2
        assert reopened.stats()["templates"] == 2
        results = reopened.identify(
            affine_variant(alice, angle=-5.0, noise_seed=777),
            kind="cropped",
            top_k=2,
        )
        assert [result.identity for result in results] == ["alice", "bob"]
        assert results[0].accepted
        assert results[0].score > results[1].score
        verified = reopened.verify(
            "alice",
            affine_variant(alice, angle=4.0, noise_seed=778),
            kind="cropped",
        )
        assert verified.accepted


def test_directory_enrollment_uses_identity_and_finger_layout(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    alice_dir = dataset / "alice" / "left_index"
    bob_dir = dataset / "bob" / "right_thumb"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)
    assert cv.imwrite(str(alice_dir / "capture_1.png"), synthetic_fingerprint(101))
    assert cv.imwrite(str(bob_dir / "capture_1.png"), synthetic_fingerprint(202))

    with FingerprintMatcher(tmp_path / "index.db") as matcher:
        report = matcher.enroll_directory(dataset)
        assert (report.added, report.failed) == (2, 0)
        assert matcher.stats()["templates_per_finger"] == {
            "alice/left_index": 1,
            "bob/right_thumb": 1,
        }


def test_multi_crop_query_is_order_independent(tmp_path: Path) -> None:
    first = synthetic_fingerprint(301)
    second = synthetic_fingerprint(302)
    with FingerprintMatcher(tmp_path / "hands.db") as matcher:
        matcher.enroll("alice", first, finger="finger_a", kind="cropped")
        matcher.enroll("alice", second, finger="finger_b", kind="cropped")
        matcher.enroll(
            "bob", synthetic_fingerprint(401), finger="finger_a", kind="cropped"
        )
        matcher.enroll(
            "bob", synthetic_fingerprint(402), finger="finger_b", kind="cropped"
        )

        query_a = matcher.template(affine_variant(first, angle=5, noise_seed=1))
        query_b = matcher.template(affine_variant(second, angle=-6, noise_seed=2))
        forward = matcher.identify([query_a, query_b], top_k=2)
        reversed_query = matcher.identify([query_b, query_a], top_k=2)

        assert forward[0].identity == reversed_query[0].identity == "alice"
        assert forward[0].accepted and reversed_query[0].accepted
        assert forward[0].score == pytest.approx(reversed_query[0].score)


def test_one_matching_crop_cannot_accept_a_four_crop_hand(tmp_path: Path) -> None:
    enrolled = synthetic_fingerprint(450)
    with FingerprintMatcher(tmp_path / "sparse.db") as matcher:
        matcher.enroll("alice", enrolled, finger="only_finger", kind="cropped")
        queries = [
            matcher.template(affine_variant(enrolled, noise_seed=451)),
            matcher.template(synthetic_fingerprint(452)),
            matcher.template(synthetic_fingerprint(453)),
            matcher.template(synthetic_fingerprint(454)),
        ]

        result = matcher.verify("alice", queries)

        assert result.matched_fingers == 1
        assert not result.accepted
        assert result.score == pytest.approx(result.assignments[0].result.score / 4)


def test_partial_hand_query_cannot_reduce_the_evidence_quorum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enrolled = synthetic_fingerprint(460)
    blank = np.full_like(enrolled, 255)
    crops = [
        FingerprintCrop(
            image=affine_variant(enrolled, noise_seed=461),
            bbox=(index * 10, 0, 100, 100),
            contour_area=10_000,
            quality=1.0,
        )
        for index in range(1)
    ]
    crops.extend(
        FingerprintCrop(
            image=blank,
            bbox=(index * 10, 0, 100, 100),
            contour_area=10_000,
            quality=1.0,
        )
        for index in range(1, 4)
    )
    monkeypatch.setattr("fingerprint_matching.extract_fingerprints", lambda *_args: crops)

    with FingerprintMatcher(tmp_path / "partial-query.db") as matcher:
        matcher.enroll("alice", enrolled, finger="index", kind="cropped")
        with pytest.raises(FingerprintQualityError, match="only 1/4 usable"):
            matcher.verify(
                "alice", np.zeros((200, 200, 3), dtype=np.uint8), kind="hand"
            )


def test_hand_enrollment_is_atomic_when_one_crop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blank = np.full((520, 520), 255, dtype=np.uint8)
    crops = [
        FingerprintCrop(
            image=synthetic_fingerprint(470 + index) if index < 3 else blank,
            bbox=(index * 10, 0, 100, 100),
            contour_area=10_000,
            quality=1.0,
        )
        for index in range(4)
    ]
    monkeypatch.setattr("fingerprint_matching.extract_fingerprints", lambda *_args: crops)

    with FingerprintMatcher(tmp_path / "atomic-hand.db") as matcher:
        report = matcher.enroll_hand(
            "alice", np.zeros((200, 200, 3), dtype=np.uint8)
        )

        assert report.failed == 1
        assert report.added == 0
        assert matcher.stats()["templates"] == 0


def test_near_tie_is_reported_as_ambiguous(tmp_path: Path) -> None:
    image = synthetic_fingerprint(505)
    alias_image = image.copy()
    alias_patch = alias_image[200:220, 200:220].astype(np.int16) + 1
    alias_image[200:220, 200:220] = np.clip(alias_patch, 0, 255).astype(np.uint8)
    with FingerprintMatcher(tmp_path / "ambiguous.db") as matcher:
        matcher.enroll("alice", image, kind="cropped")
        matcher.enroll("alias", alias_image, kind="cropped")

        result = matcher.identify(image, kind="cropped", top_k=2)

        assert result[0].ambiguous
        assert not result[0].accepted
        assert result[1].ambiguous
        assert not result[1].accepted
        assert result[0].score - result[1].score < matcher.config.ambiguity_margin


def test_conflicting_duplicate_enrollment_is_rejected(tmp_path: Path) -> None:
    image = synthetic_fingerprint(515)
    with FingerprintMatcher(tmp_path / "duplicates.db") as matcher:
        assert matcher.enroll("alice", image, kind="cropped").added == 1
        conflicting = matcher.enroll("bob", image, kind="cropped")

        assert conflicting.failed == 1
        assert "refusing conflicting enrollment" in conflicting.errors[0]["error"]


def test_concurrent_conflicting_enrollment_cannot_bypass_unique_guard(
    tmp_path: Path,
) -> None:
    database = tmp_path / "concurrent.db"
    template = create_template(synthetic_fingerprint(516))
    with FingerprintMatcher(database):
        pass
    barrier = threading.Barrier(4)

    def add(identity_number: int) -> int | str:
        with FingerprintStore(database) as store:
            barrier.wait()
            try:
                result = store.add_template(
                    f"identity_{identity_number}",
                    "index",
                    template,
                    source_digest="same-source-digest",
                )
                return -1 if result is None else result
            except FingerprintDatabaseError as exc:
                return str(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(add, range(4)))

    assert sum(isinstance(outcome, int) and outcome > 0 for outcome in outcomes) == 1
    assert sum(isinstance(outcome, str) for outcome in outcomes) == 3
    with FingerprintStore(database) as store:
        assert store.stats()["templates"] == 1
        assert store.stats()["identities"] == 1


def test_failed_array_enrollment_does_not_dump_pixels_into_report(tmp_path: Path) -> None:
    blank = np.full((300, 400), 237, dtype=np.uint8)
    with FingerprintMatcher(tmp_path / "safe-errors.db") as matcher:
        report = matcher.enroll("alice", blank, kind="cropped")

    assert report.failed == 1
    assert report.errors[0]["source"] == "<ndarray shape=(300, 400) dtype=uint8>"
    assert len(json_text := str(report.to_dict())) < 500
    assert "[[237" not in json_text


def test_database_rejects_incompatible_feature_configuration(tmp_path: Path) -> None:
    database = tmp_path / "configured.db"
    with FingerprintMatcher(database) as matcher:
        matcher.enroll("alice", synthetic_fingerprint(606), kind="cropped")

    with pytest.raises(FingerprintDatabaseError, match="feature settings differ"):
        FingerprintMatcher(database, config=MatcherConfig(max_features=800))


def test_database_rejects_incompatible_extraction_configuration(tmp_path: Path) -> None:
    database = tmp_path / "extraction-configured.db"
    with FingerprintMatcher(database):
        pass

    with pytest.raises(FingerprintDatabaseError, match="extraction settings differ"):
        FingerprintMatcher(
            database, extraction_config=ExtractionConfig(max_fingers=3)
        )


def test_rejected_sqlite_database_is_not_modified(tmp_path: Path) -> None:
    database = tmp_path / "foreign.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE fingerprint_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        connection.execute(
            "INSERT INTO fingerprint_metadata VALUES ('schema_version', '999')"
        )
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    with sqlite3.connect(database) as connection:
        before = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

    with pytest.raises(FingerprintDatabaseError):
        FingerprintMatcher(database)

    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    assert after == before
