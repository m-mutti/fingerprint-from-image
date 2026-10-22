"""SSD anchor generation, palm-box decoding, and weighted NMS (numpy only).

These functions re-implement MediaPipe's ``SsdAnchorsCalculator``,
``TensorsToDetectionsCalculator`` and ``NonMaxSuppressionCalculator`` (weighted
mode) for the BlazePalm detector shipped in ``hand_landmarker.task``.

Coordinates are normalised to the detector input (``[0, 1]`` across the
letterboxed 192 x 192 tensor) until :mod:`.geometry` maps them back to source
pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PALM_ANCHOR_OPTIONS",
    "AnchorOptions",
    "PalmDetection",
    "decode_palm_detections",
    "generate_anchors",
    "iou_matrix",
    "weighted_nms",
]


@dataclass(frozen=True)
class AnchorOptions:
    """Subset of MediaPipe ``SsdAnchorsCalculatorOptions`` used by BlazePalm."""

    input_size: int = 192
    num_layers: int = 4
    min_scale: float = 0.1484375
    max_scale: float = 0.75
    anchor_offset: float = 0.5
    strides: tuple[int, ...] = (8, 16, 16, 16)
    aspect_ratios: tuple[float, ...] = (1.0,)
    interpolated_scale_aspect_ratio: float = 1.0
    fixed_anchor_size: bool = True

    def __post_init__(self) -> None:
        if len(self.strides) != self.num_layers:
            raise ValueError("strides must have one entry per layer")


PALM_ANCHOR_OPTIONS = AnchorOptions()


def _scale(options: AnchorOptions, index: int) -> float:
    if options.num_layers == 1:
        return (options.min_scale + options.max_scale) * 0.5
    span = options.max_scale - options.min_scale
    return options.min_scale + span * index / (options.num_layers - 1.0)


def generate_anchors(options: AnchorOptions = PALM_ANCHOR_OPTIONS) -> np.ndarray:
    """Return ``(N, 4)`` anchors as ``(x_center, y_center, width, height)``.

    Consecutive layers with the same stride share one feature map, exactly as
    in MediaPipe, so the BlazePalm options yield ``24*24*2 + 12*12*6 = 2016``
    anchors in row-major, anchor-minor order.
    """

    anchors: list[np.ndarray] = []
    layer = 0
    while layer < options.num_layers:
        sizes: list[tuple[float, float]] = []
        last = layer
        while last < options.num_layers and options.strides[last] == options.strides[layer]:
            scale = _scale(options, last)
            for ratio in options.aspect_ratios:
                root = float(np.sqrt(ratio))
                sizes.append((scale * root, scale / root))
            if options.interpolated_scale_aspect_ratio > 0.0:
                next_scale = 1.0 if last == options.num_layers - 1 else _scale(options, last + 1)
                interpolated = float(np.sqrt(scale * next_scale))
                root = float(np.sqrt(options.interpolated_scale_aspect_ratio))
                sizes.append((interpolated * root, interpolated / root))
            last += 1
        stride = options.strides[layer]
        rows = int(np.ceil(options.input_size / stride))
        cols = int(np.ceil(options.input_size / stride))
        yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        centers = np.stack(
            [(xx + options.anchor_offset) / cols, (yy + options.anchor_offset) / rows], axis=-1
        ).reshape(-1, 1, 2)
        size_array = np.asarray(sizes, dtype=np.float64)
        if options.fixed_anchor_size:
            size_array = np.ones_like(size_array)
        block = np.concatenate(
            [
                np.broadcast_to(centers, (centers.shape[0], len(sizes), 2)),
                np.broadcast_to(size_array, (centers.shape[0], len(sizes), 2)),
            ],
            axis=-1,
        )
        anchors.append(block.reshape(-1, 4))
        layer = last
    return np.concatenate(anchors, axis=0).astype(np.float32)


@dataclass(frozen=True)
class PalmDetection:
    """One decoded palm in normalised detector-input coordinates."""

    box: np.ndarray  # (4,) xmin, ymin, xmax, ymax
    keypoints: np.ndarray  # (7, 2) x, y
    score: float


def decode_palm_detections(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    *,
    input_size: int = 192,
    score_threshold: float = 0.5,
    score_clip: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw BlazePalm regressors into boxes, keypoints and scores.

    Args:
        raw_boxes: ``(N, 18)`` regressors (``x, y, w, h`` + 7 keypoints, pixels
            of the detector input relative to the anchor centre).
        raw_scores: ``(N,)`` or ``(N, 1)`` logits.
        anchors: ``(N, 4)`` from :func:`generate_anchors`.

    Returns:
        ``boxes (M, 4)`` as ``xmin, ymin, xmax, ymax``, ``keypoints (M, 7, 2)``
        and ``scores (M,)`` for anchors whose sigmoid score passes the threshold.
    """

    raw_boxes = np.asarray(raw_boxes, dtype=np.float32).reshape(len(anchors), -1)
    logits = np.clip(np.asarray(raw_scores, dtype=np.float64).reshape(-1), -score_clip, score_clip)
    scores = 1.0 / (1.0 + np.exp(-logits))
    keep = scores >= score_threshold
    raw, anc, scores = raw_boxes[keep], anchors[keep], scores[keep]
    scale = float(input_size)
    cx = raw[:, 0] / scale * anc[:, 2] + anc[:, 0]
    cy = raw[:, 1] / scale * anc[:, 3] + anc[:, 1]
    w = raw[:, 2] / scale * anc[:, 2]
    h = raw[:, 3] / scale * anc[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kp = raw[:, 4:].reshape(len(raw), (raw.shape[1] - 4) // 2, 2) / scale
    kp = kp * anc[:, None, 2:4] + anc[:, None, 0:2]
    return boxes, kp, scores


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``(A, 4)`` and ``(B, 4)`` ``xmin, ymin, xmax, ymax`` boxes."""

    a = np.asarray(boxes_a, dtype=np.float64)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float64)[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def weighted_nms(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = 0.3,
    max_detections: int | None = None,
) -> list[PalmDetection]:
    """MediaPipe-style weighted non-maximum suppression.

    The highest-scoring remaining box absorbs every remaining box whose IoU
    with it exceeds ``iou_threshold``; the output box and keypoints are the
    score-weighted mean of that cluster, and the output score is the cluster
    maximum. Clusters are emitted in descending score order.
    """

    order = np.argsort(-np.asarray(scores), kind="stable")
    remaining = list(order)
    results: list[PalmDetection] = []
    while remaining and (max_detections is None or len(results) < max_detections):
        top = remaining[0]
        overlaps = iou_matrix(boxes[top][None], boxes[remaining])[0]
        members = [idx for idx, ov in zip(remaining, overlaps, strict=True)
                   if idx == top or ov > iou_threshold]
        weights = scores[members].astype(np.float64)
        weights = weights / weights.sum()
        box = (boxes[members] * weights[:, None]).sum(axis=0)
        kps = (keypoints[members] * weights[:, None, None]).sum(axis=0)
        results.append(PalmDetection(box.astype(np.float32), kps.astype(np.float32),
                                     float(scores[top])))
        member_set = set(members)
        remaining = [idx for idx in remaining if idx not in member_set]
    return results
