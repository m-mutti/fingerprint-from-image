"""Registered hand models and thin ONNX Runtime wrappers.

The ONNX files are converted from Google's MediaPipe ``hand_landmarker.task``
(Apache-2.0) by ``tools/hand_models/convert.py``; they are not downloaded
automatically because the upstream artifact is a TFLite bundle. Place them in
the model cache (``$FINGERPRINT_FROM_IMAGE_MODELS``) or pass ``model_dir``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import ModelSpec, model_path, register

__all__ = [
    "LANDMARK_MODEL",
    "PALM_MODEL",
    "LandmarkModel",
    "LandmarkOutput",
    "PalmModel",
    "load_session",
]

_UPSTREAM = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

PALM_MODEL = register(ModelSpec(
    name="mediapipe_hand_detector.onnx",
    sha256="615009f34258d557312a4eea5cab235c4fbeb3ada2a329a9bac036ead4545a94",
    url=None,
    license="Apache-2.0",
    description=f"BlazePalm detector (192x192) from {_UPSTREAM}, tf2onnx 1.16.1 opset 17",
))

LANDMARK_MODEL = register(ModelSpec(
    name="mediapipe_hand_landmarks_detector.onnx",
    sha256="27953c218051f32d3b9e95eb55bc0e6f860947eaa25a637499bb8e181c632b25",
    url=None,
    license="Apache-2.0",
    description=f"21-point hand landmarks (224x224) from {_UPSTREAM}, tf2onnx 1.16.1 opset 17",
))


def load_session(spec: ModelSpec, model_dir: Path | None,
                 providers: Sequence[str] | None, num_threads: int = 4) -> Any:
    """Resolve, checksum-verify and open an ONNX Runtime session for ``spec``.

    ``num_threads`` sets ONNX Runtime's intra-op pool (0 = its default, all
    cores); a few threads are faster and steadier for these small models.
    """

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The hand detector needs onnxruntime: pip install onnxruntime"
        ) from exc
    explicit = None if model_dir is None else Path(model_dir) / spec.name
    path = model_path(spec.name, explicit=explicit, download=False)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = max(0, int(num_threads))
    chosen = list(providers) if providers else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=options, providers=chosen)


class PalmModel:
    """BlazePalm: ``(1, 192, 192, 3)`` RGB in ``[0, 1]`` -> 2016 boxes + logits."""

    input_size = 192

    def __init__(self, session: Any) -> None:
        self._session = session
        self._input = session.get_inputs()[0].name

    def __call__(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores = self._session.run(None, {self._input: tensor[None].astype(np.float32)})
        return boxes[0], scores[0, :, 0]


class LandmarkOutput:
    """Raw landmark-model output for one ROI."""

    __slots__ = ("handedness", "landmarks", "presence", "world")

    def __init__(self, landmarks: np.ndarray, presence: float, handedness: float,
                 world: np.ndarray) -> None:
        self.landmarks = landmarks  # (21, 3) in input pixels (x, y) and relative z
        self.presence = presence
        self.handedness = handedness  # probability of a right hand (non-mirrored image)
        self.world = world  # (21, 3) metric coordinates, metres


class LandmarkModel:
    """Hand landmarks: ``(N, 224, 224, 3)`` RGB in ``[0, 1]``."""

    input_size = 224

    def __init__(self, session: Any) -> None:
        self._session = session
        self._input = session.get_inputs()[0].name
        self._names = [o.name for o in session.get_outputs()]

    def __call__(self, tensor: np.ndarray) -> LandmarkOutput:
        values = self._session.run(None, {self._input: tensor[None].astype(np.float32)})
        outputs = dict(zip(self._names, values, strict=True))
        return LandmarkOutput(
            landmarks=outputs["Identity"].reshape(21, 3).astype(np.float64),
            presence=float(outputs["Identity_1"].reshape(-1)[0]),
            handedness=float(outputs["Identity_2"].reshape(-1)[0]),
            world=outputs["Identity_3"].reshape(21, 3).astype(np.float64),
        )
