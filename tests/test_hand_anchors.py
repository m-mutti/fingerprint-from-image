from __future__ import annotations

import numpy as np
import pytest

from fingerprint_from_image.hand.anchors import (
    AnchorOptions,
    decode_palm_detections,
    generate_anchors,
    iou_matrix,
    weighted_nms,
)


def test_palm_anchor_layout_matches_mediapipe() -> None:
    anchors = generate_anchors()
    assert anchors.shape == (2016, 4)
    # Stride 8: 24x24 cells x 2 anchors, then stride 16: 12x12 cells x 6 anchors.
    np.testing.assert_allclose(anchors[0], [0.5 / 24, 0.5 / 24, 1, 1], atol=1e-7)
    np.testing.assert_allclose(anchors[1], anchors[0])
    np.testing.assert_allclose(anchors[2, :2], [1.5 / 24, 0.5 / 24], atol=1e-7)
    np.testing.assert_allclose(anchors[1151, :2], [23.5 / 24, 23.5 / 24], atol=1e-7)
    np.testing.assert_allclose(anchors[1152, :2], [0.5 / 12, 0.5 / 12], atol=1e-7)
    assert np.all(anchors[1152:1158, :2] == anchors[1152, :2])
    np.testing.assert_allclose(anchors[-1, :2], [11.5 / 12, 11.5 / 12], atol=1e-7)
    assert np.all(anchors[:, 2:] == 1.0)  # fixed anchor size


def test_anchor_options_validation_and_variable_size() -> None:
    with pytest.raises(ValueError):
        AnchorOptions(num_layers=2, strides=(8,))
    options = AnchorOptions(input_size=64, num_layers=2, strides=(8, 16), fixed_anchor_size=False)
    anchors = generate_anchors(options)
    assert anchors.shape == (8 * 8 * 2 + 4 * 4 * 2, 4)
    assert anchors[0, 2] == pytest.approx(options.min_scale)


def test_decode_zero_regression_returns_anchor_boxes() -> None:
    anchors = generate_anchors()
    raw = np.zeros((2016, 18), dtype=np.float32)
    raw[:, 2:4] = 192 * 0.25  # w = h = 0.25 of the input
    raw[:, 4:6] = [192 * 0.1, -192 * 0.05]  # first keypoint offset
    logits = np.full(2016, -10.0)
    logits[[5, 1500]] = [3.0, 0.0]  # sigmoid 0.95 and 0.5
    boxes, keypoints, scores = decode_palm_detections(raw, logits, anchors, score_threshold=0.5)
    assert len(boxes) == 2 and keypoints.shape == (2, 7, 2)
    centre = anchors[5, :2]
    np.testing.assert_allclose(boxes[0], [*(centre - 0.125), *(centre + 0.125)], atol=1e-6)
    np.testing.assert_allclose(keypoints[0, 0], centre + [0.1, -0.05], atol=1e-6)
    np.testing.assert_allclose(keypoints[0, 1], centre, atol=1e-6)
    assert scores[0] == pytest.approx(1 / (1 + np.exp(-3.0)))


def test_decode_handles_no_detection_and_huge_logits() -> None:
    anchors = generate_anchors()
    raw = np.zeros((1, 2016, 18), dtype=np.float32)
    boxes, keypoints, scores = decode_palm_detections(raw, np.full(2016, -50.0), anchors)
    assert boxes.shape == (0, 4) and keypoints.shape == (0, 7, 2) and scores.shape == (0,)
    with np.errstate(over="raise"):
        _, _, scores = decode_palm_detections(raw, np.full((2016, 1), -1e6), anchors)
    assert len(scores) == 0


def test_iou_matrix() -> None:
    a = np.array([[0, 0, 2, 2], [0, 0, 1, 1]], dtype=float)
    b = np.array([[1, 1, 3, 3], [5, 5, 5, 5]], dtype=float)
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 2)
    assert iou[0, 0] == pytest.approx(1 / 7)
    assert iou[1, 0] == 0 and iou[0, 1] == 0  # disjoint and degenerate boxes


def test_weighted_nms_merges_clusters_by_score() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0.5, 0, 10.5, 10]],
                     dtype=float)
    keypoints = np.zeros((4, 7, 2))
    keypoints[:, 0, 0] = [0, 2, 50, 1]
    scores = np.array([0.9, 0.3, 0.8, 0.6])
    merged = weighted_nms(boxes, keypoints, scores, iou_threshold=0.3)
    assert [round(d.score, 2) for d in merged] == [0.9, 0.8]
    expected_x0 = (0.9 * 0 + 0.3 * 1 + 0.6 * 0.5) / 1.8
    assert merged[0].box[0] == pytest.approx(expected_x0, rel=1e-5)
    assert merged[0].keypoints[0, 0] == pytest.approx((0.3 * 2 + 0.6 * 1) / 1.8, rel=1e-5)
    np.testing.assert_allclose(merged[1].box, boxes[2])
    assert len(weighted_nms(boxes, keypoints, scores, max_detections=1)) == 1
    assert weighted_nms(boxes[:0], keypoints[:0], scores[:0]) == []
