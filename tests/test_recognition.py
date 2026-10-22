from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fingerprint_from_image.errors import FingerprintDatabaseError, FingerprintMatchingError
from fingerprint_from_image.recognition import (
    EmbeddingStore,
    FingerprintRecognizer,
    RecognizerConfig,
    best_assignment,
)
from fingerprint_from_image.types import FingerCrop


class BlockEmbedder:
    """Deterministic stand-in for a learned model: 16x16 block means, unit norm."""

    model_version = "unit-block-v1"
    dim = 256

    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for image in images:
            data = np.asarray(image, dtype=np.float32)[:128, :128]
            blocks = data.reshape(16, 8, 16, 8).mean(axis=(1, 3)).reshape(-1)
            blocks -= blocks.mean()
            rows.append(blocks / max(float(np.linalg.norm(blocks)), 1e-6))
        return np.stack(rows).astype(np.float32)


def _print(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (128, 128), dtype=np.uint8)


def _variant(image: np.ndarray, seed: int, sigma: float = 12.0) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0.0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


class FixedDetector:
    def __init__(self, images: Sequence[np.ndarray]) -> None:
        self.images = list(images)

    def __call__(self, _rgb: np.ndarray) -> list[FingerCrop]:
        labels = ("right_index", "right_middle", "right_ring", "right_little")
        pairs = zip(self.images, labels, strict=False)
        return [FingerCrop(image, label=label) for image, label in pairs]


def test_enroll_identify_and_verify(tmp_path: Path) -> None:
    alice, bob = _print(1), _print(2)
    with FingerprintRecognizer(tmp_path / "v4.db", BlockEmbedder()) as recognizer:
        assert recognizer.enroll("alice", alice, finger="right_index") == 1
        assert recognizer.enroll("bob", bob, finger="right_index") == 1
        assert recognizer.enroll("alice", alice, finger="right_index") == 0
        ranked = recognizer.identify(_variant(alice, 10))
        assert [item.identity for item in ranked] == ["alice", "bob"]
        assert ranked[0].accepted and not ranked[1].accepted
        assert recognizer.verify("alice", _variant(alice, 11)).accepted
        assert not recognizer.verify("bob", _variant(alice, 12)).accepted
        with pytest.raises(FingerprintMatchingError):
            recognizer.verify("carol", alice)
        assert recognizer.store.stats()["embeddings"] == 2


def test_conflicting_enrollment_is_refused(tmp_path: Path) -> None:
    with FingerprintRecognizer(tmp_path / "v4.db", BlockEmbedder()) as recognizer:
        recognizer.enroll("alice", _print(3))
        with pytest.raises(FingerprintDatabaseError):
            recognizer.enroll("mallory", _print(3))


def test_one_matching_fingertip_cannot_accept_a_four_finger_hand(tmp_path: Path) -> None:
    enrolled = [_print(20 + index) for index in range(4)]
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    database = tmp_path / "v4.db"
    with FingerprintRecognizer(
        database, BlockEmbedder(), detector=FixedDetector(enrolled)
    ) as recognizer:
        assert recognizer.enroll("alice", rgb, kind="hand") == 4
    genuine = [_variant(image, 30 + index) for index, image in enumerate(enrolled)]
    with FingerprintRecognizer(
        database, BlockEmbedder(), detector=FixedDetector(genuine[::-1])
    ) as recognizer:
        result = recognizer.verify("alice", rgb, kind="hand")
        assert result.accepted and result.matched_fingers == 4
    partial = [genuine[0], _print(90), _print(91), _print(92)]
    with FingerprintRecognizer(
        database, BlockEmbedder(), detector=FixedDetector(partial)
    ) as recognizer:
        result = recognizer.verify("alice", rgb, kind="hand")
        assert result.matched_fingers == 1 and not result.accepted


def test_near_tie_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    base = _print(40)
    config = RecognizerConfig(ambiguity_margin=0.05)
    with FingerprintRecognizer(tmp_path / "v4.db", BlockEmbedder(), config=config) as recognizer:
        recognizer.enroll("alice", base)
        recognizer.enroll("twin", _variant(base, 41, sigma=2.0))
        ranked = recognizer.identify(_variant(base, 42))
        assert all(item.ambiguous and not item.accepted for item in ranked[:2])


def test_database_binding_and_legacy_refusal(tmp_path: Path) -> None:
    database = tmp_path / "v4.db"
    with FingerprintRecognizer(database, BlockEmbedder()) as recognizer:
        recognizer.enroll("alice", _print(5))
    with pytest.raises(FingerprintDatabaseError, match="model"):
        EmbeddingStore(database, model_version="other-model", dim=256)

    from fingerprint_from_image.store import FingerprintStore

    legacy = tmp_path / "legacy.db"
    FingerprintStore(legacy).close()
    with pytest.raises(FingerprintDatabaseError, match="v3 RootSIFT"):
        EmbeddingStore(legacy, model_version="unit-block-v1", dim=256)


def test_tampered_embedding_fails_checksum(tmp_path: Path) -> None:
    database = tmp_path / "v4.db"
    with FingerprintRecognizer(database, BlockEmbedder()) as recognizer:
        recognizer.enroll("alice", _print(6))
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE embeddings SET vector = zeroblob(1024)")
    with EmbeddingStore(database, model_version="unit-block-v1", dim=256) as store:
        with pytest.raises(FingerprintDatabaseError, match="checksum"):
            store.gallery()


def test_remove_identity(tmp_path: Path) -> None:
    with FingerprintRecognizer(tmp_path / "v4.db", BlockEmbedder()) as recognizer:
        recognizer.enroll("alice", _print(7))
        recognizer.enroll("bob", _print(8))
        assert recognizer.store.remove_identity("alice") == 1
        assert [item.identity for item in recognizer.identify(_print(8))] == ["bob"]


def test_best_assignment_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    for rows, columns in ((3, 3), (4, 2), (2, 5)):
        scores = rng.random((rows, columns))
        pairs = best_assignment(scores)
        best = max(
            sum(scores[r, c] for r, c in zip(perm_rows, perm_cols, strict=True))
            for k in range(min(rows, columns) + 1)
            for perm_rows in itertools.combinations(range(rows), k)
            for perm_cols in itertools.permutations(range(columns), k)
        )
        assert sum(scores[r, c] for r, c in pairs) == pytest.approx(best)
