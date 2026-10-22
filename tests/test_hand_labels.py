from __future__ import annotations

import numpy as np
import pytest

from fingerprint_from_image.hand.topology import (
    FINGERS,
    HAND_CONNECTIONS,
    finger_label,
    palm_chirality,
    palm_facing,
    side_from_score,
)
from fingerprint_from_image.types import FINGER_LABELS
from test_hand_crops import make_hand


def test_side_from_score_and_mirroring() -> None:
    assert side_from_score(0.98) == ("right", pytest.approx(0.98))
    assert side_from_score(0.10) == ("left", pytest.approx(0.90))
    assert side_from_score(0.98, mirrored=True)[0] == "left"
    assert side_from_score(0.10, mirrored=True)[0] == "right"


def test_finger_labels_are_valid_finger_crop_labels() -> None:
    for side in ("left", "right"):
        for index in range(5):
            assert finger_label(side, index) in FINGER_LABELS
    assert finger_label("right", 1) == "right_index"
    assert finger_label(None, 3) == "unknown"
    assert FINGERS == ("thumb", "index", "middle", "ring", "little")


def test_connections_cover_all_landmarks() -> None:
    assert {i for edge in HAND_CONNECTIONS for i in edge} == set(range(21))


def test_palm_chirality_and_facing() -> None:
    right_palm = make_hand()  # thumb on image right, palm toward camera
    assert palm_chirality(right_palm) < 0
    assert palm_chirality(make_hand(2.0, scale=3.0)) == pytest.approx(palm_chirality(right_palm))
    mirrored = right_palm * np.array([-1.0, 1.0])  # looks like a left palm / right back
    assert palm_chirality(mirrored) > 0
    assert palm_facing(right_palm, apparent_right=True) is True
    assert palm_facing(right_palm, apparent_right=False) is False  # left hand, back shown
    assert palm_facing(mirrored, apparent_right=False) is True
    edge_on = right_palm.copy()
    edge_on[:, 0] = right_palm[0, 0]  # all x collapse: hand seen edge-on
    assert palm_facing(edge_on, apparent_right=True) is None
