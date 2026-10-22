"""Learned hand -> fingertip pipeline: palm detection, landmarks, labelled crops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import to_rgb
from ..types import FingerCrop
from .anchors import PALM_ANCHOR_OPTIONS, decode_palm_detections, generate_anchors, iou_matrix
from .anchors import weighted_nms as _weighted_nms
from .crops import CropConfig, finger_boxes, sample_crop
from .enhance import normalize_fingertip, ridge_quality
from .geometry import RotatedRect, letterbox, rect_from_landmarks, rect_from_palm, warp_rect
from .runtime import (
    LANDMARK_MODEL,
    PALM_MODEL,
    LandmarkModel,
    LandmarkOutput,
    PalmModel,
    load_session,
)
from .topology import Side, finger_label, palm_facing, side_from_score

__all__ = ["DetectorConfig", "HandFingerDetector", "HandNotFoundError", "HandResult"]


def _color_image(rgb: np.ndarray) -> Image.Image:
    """Validate colour input; the palm model cannot find hands in grayscale."""

    if np.asarray(rgb).ndim == 2:
        raise ValueError(
            "HandFingerDetector needs a colour image; use load_image(path, mode='RGB')"
        )
    return Image.fromarray(to_rgb(rgb))


class HandNotFoundError(ValueError):
    """Raised when no hand passes the palm-detection and presence thresholds."""


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds and options of :class:`HandFingerDetector`.

    Attributes:
        max_hands: maximum number of hands returned (highest palm score first).
        palm_score_threshold: minimum BlazePalm sigmoid score.
        nms_iou_threshold: IoU above which palm boxes are merged (weighted NMS).
        presence_threshold: minimum landmark-model hand-presence probability.
        refine_iterations: extra landmark passes on the ROI re-fitted to the
            previous landmarks (MediaPipe's tracking ROI); 0 disables it.
        duplicate_iou: hands whose landmark boxes overlap more are merged.
        mirrored: the image is a horizontally flipped selfie preview; swaps
            left/right labels (see :mod:`.topology`).
        require_palm_side: drop hands whose back faces the camera (no pads).
        enhance: apply illumination flattening + CLAHE to the crops.
        clahe_clip_limit: CLAHE clip limit (histogram multiples).
        num_threads: ONNX Runtime intra-op threads (0 = runtime default).
        crop: fingertip box geometry.
    """

    max_hands: int = 2
    palm_score_threshold: float = 0.5
    nms_iou_threshold: float = 0.3
    presence_threshold: float = 0.5
    refine_iterations: int = 1
    duplicate_iou: float = 0.5
    mirrored: bool = False
    require_palm_side: bool = False
    enhance: bool = True
    clahe_clip_limit: float = 2.0
    num_threads: int = 4
    crop: CropConfig = field(default_factory=CropConfig)

    def __post_init__(self) -> None:
        if self.max_hands < 1:
            raise ValueError("max_hands must be at least 1")
        if self.refine_iterations < 0:
            raise ValueError("refine_iterations must be non-negative")
        for name in ("palm_score_threshold", "nms_iou_threshold", "presence_threshold",
                     "duplicate_iou"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class HandResult:
    """One detected hand in source-pixel coordinates.

    ``landmarks`` is ``(21, 3)``: ``x, y`` in source pixels and a relative depth
    ``z`` in the same pixel scale (smaller = closer to the camera).
    ``handedness_score`` is the raw model probability of a right hand for the
    image as given; ``side`` already accounts for ``DetectorConfig.mirrored``.
    """

    landmarks: np.ndarray = field(repr=False, compare=False)
    side: Side
    side_confidence: float
    handedness_score: float
    presence: float
    palm_score: float
    palm_facing: bool | None
    roi: RotatedRect
    world_landmarks: np.ndarray = field(repr=False, compare=False)


def _bbox(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points[:, :2].min(axis=0), points[:, :2].max(axis=0)])


class HandFingerDetector:
    """Detect hands in an RGB photograph and return labelled fingertip crops.

    Args:
        model_dir: directory holding the converted ONNX files; defaults to the
            model cache (``$FINGERPRINT_FROM_IMAGE_MODELS``).
        providers: ONNX Runtime execution providers (default CPU).
        config: thresholds and crop geometry.
    """

    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None,
                 config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        threads = self.config.num_threads
        self._palm = PalmModel(load_session(PALM_MODEL, model_dir, providers, threads))
        self._landmarks = LandmarkModel(load_session(LANDMARK_MODEL, model_dir, providers, threads))
        self._anchors = generate_anchors(PALM_ANCHOR_OPTIONS)

    # -- stages ---------------------------------------------------------------
    def detect_palms(self, image: Image.Image) -> list[tuple[RotatedRect, float]]:
        """Palm detections as hand ROIs in source pixels, best first."""

        cfg = self.config
        tensor, mapping = letterbox(image, PalmModel.input_size)
        raw_boxes, logits = self._palm(tensor)
        boxes, keypoints, scores = decode_palm_detections(
            raw_boxes, logits, self._anchors, input_size=PalmModel.input_size,
            score_threshold=cfg.palm_score_threshold)
        palms = _weighted_nms(boxes, keypoints, scores, iou_threshold=cfg.nms_iou_threshold,
                              max_detections=2 * cfg.max_hands)
        rois = []
        for palm in palms:
            corners = mapping.to_source(palm.box.reshape(2, 2)).reshape(-1)
            rois.append((rect_from_palm(corners, mapping.to_source(palm.keypoints)), palm.score))
        return rois

    def _run_landmarks(self, image: Image.Image, roi: RotatedRect
                       ) -> tuple[np.ndarray, LandmarkOutput]:
        size = LandmarkModel.input_size
        patch = warp_rect(image, roi, size, size)
        output = self._landmarks(np.asarray(patch, dtype=np.float32) / 255.0)
        normalized = output.landmarks[:, :2] / size
        points = np.empty((21, 3))
        points[:, :2] = roi.to_image(normalized)
        points[:, 2] = output.landmarks[:, 2] / size * roi.width
        return points, output

    def detect_hands(self, rgb: np.ndarray) -> list[HandResult]:
        """Run palm detection and landmark regression; returns up to ``max_hands``."""

        return self._detect(_color_image(rgb))

    def _detect(self, image: Image.Image) -> list[HandResult]:
        cfg = self.config
        hands: list[HandResult] = []
        for roi, palm_score in self.detect_palms(image):
            points, output = self._run_landmarks(image, roi)
            for _ in range(cfg.refine_iterations):
                if output.presence < cfg.presence_threshold:
                    break
                refined_roi = rect_from_landmarks(points)
                refined_points, refined = self._run_landmarks(image, refined_roi)
                if refined.presence < cfg.presence_threshold:
                    break
                roi, points, output = refined_roi, refined_points, refined
            if output.presence < cfg.presence_threshold:
                continue
            if hands and iou_matrix(_bbox(points)[None],
                                    np.stack([_bbox(h.landmarks) for h in hands])).max() \
                    > cfg.duplicate_iou:
                continue
            side, confidence = side_from_score(output.handedness, mirrored=cfg.mirrored)
            facing = palm_facing(points, output.handedness > 0.5)
            if cfg.require_palm_side and facing is False:
                continue
            hands.append(HandResult(points, side, confidence, output.handedness, output.presence,
                                    palm_score, facing, roi, output.world))
            if len(hands) >= cfg.max_hands:
                break
        return hands

    def finger_crops(self, rgb: np.ndarray, hand: HandResult, *, source: str | None = None
                     ) -> list[FingerCrop]:
        """Five upright fingertip crops (thumb..little) for one detected hand."""

        return self._crops(_color_image(rgb), hand, source)

    def _crops(self, image: Image.Image, hand: HandResult, source: str | None
               ) -> list[FingerCrop]:
        cfg = self.config
        crops: list[FingerCrop] = []
        for box in finger_boxes(hand.landmarks, cfg.crop):
            values, valid_fraction, scale = sample_crop(image, box, cfg.crop)
            period = box.ridge_period_px(cfg.crop) * scale
            quality = ridge_quality(values, period) * valid_fraction
            if cfg.enhance:
                pixels = normalize_fingertip(values, period, clip_limit=cfg.clahe_clip_limit)
            else:
                pixels = np.clip(values + 0.5, 0, 255).astype(np.uint8)
            crops.append(FingerCrop(pixels, finger_label(hand.side, box.finger),
                                    float(np.clip(quality, 0.0, 1.0)), box.polygon, source))
        return crops

    def __call__(self, rgb: np.ndarray, *, source: str | None = None) -> list[FingerCrop]:
        """Detect hands and return all their fingertip crops (5 per hand).

        Raises:
            HandNotFoundError: if no hand is detected.
            ValueError: if ``rgb`` is a grayscale image.
        """

        image = _color_image(rgb)
        hands = self._detect(image)
        if not hands:
            raise HandNotFoundError("No hand detected in the image")
        return [crop for hand in hands for crop in self._crops(image, hand, source)]
