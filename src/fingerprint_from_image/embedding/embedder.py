"""ONNX Runtime fingerprint embedder (numpy + Pillow + onnxruntime; no torch, no OpenCV)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..io import to_gray
from ..models import ModelUnavailableError, model_path
from .preprocess import INPUT_SIZE, TARGET_PERIOD, preprocess
from .registry import DEFAULT_MODEL

__all__ = ["FingerprintEmbedder"]


class FingerprintEmbedder:
    """Fixed-length, L2-normalised fingerprint embeddings from grayscale images.

    Every image goes through :func:`~fingerprint_from_image.embedding.preprocess.preprocess`
    (ridge-period normalisation to ``TARGET_PERIOD`` px, foreground-centred
    ``INPUT_SIZE`` crop, masked z-score), exactly as during training/evaluation.

    Args:
        model_path: explicit ``.onnx`` file. When omitted, the registered
            ``model_name`` is resolved from the model cache and checksum-verified.
        providers: ONNX Runtime execution providers (default: CPU).
        model_name: registry name, e.g. the INT8 variant.
        num_threads: ONNX Runtime intra-op threads (0 = ORT default).
        batch_size: images per ``session.run`` call.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        providers: Sequence[str] | None = None,
        model_name: str = DEFAULT_MODEL,
        num_threads: int = 0,
        batch_size: int = 16,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "FingerprintEmbedder needs onnxruntime: "
                "pip install 'fingerprint-from-image[embedding]'"
            ) from exc
        if model_path is not None:
            path = Path(model_path)
            if not path.is_file():
                raise ModelUnavailableError(f"Model file not found: {path}")
        else:
            path = _resolve(model_name)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = max(0, int(num_threads))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        chosen = list(providers) if providers else ["CPUExecutionProvider"]
        self._session: Any = ort.InferenceSession(str(path), sess_options=options, providers=chosen)
        self._input = self._session.get_inputs()[0].name
        meta = self._session.get_modelmeta().custom_metadata_map
        self.model_path = path
        self.model_version: str = meta.get("model_version", path.stem)
        self.input_size = int(meta.get("input_size", INPUT_SIZE))
        if float(meta.get("target_period", TARGET_PERIOD)) != TARGET_PERIOD:
            raise ModelUnavailableError(
                f"{path.name} expects ridge period {meta['target_period']}, "
                f"this runtime normalises to {TARGET_PERIOD}"
            )
        output_shape = self._session.get_outputs()[0].shape
        dim = meta.get("dim", output_shape[-1] if isinstance(output_shape[-1], int) else None)
        if dim is None:
            dim = self._run(np.zeros((1, 1, self.input_size, self.input_size), np.float32)).shape[1]
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))

    def _run(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._session.run(None, {self._input: batch})[0], dtype=np.float32)

    def prepare(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Preprocess images into the ``(N, 1, S, S)`` float32 network input."""

        if not images:
            return np.zeros((0, 1, self.input_size, self.input_size), dtype=np.float32)
        arrays = [preprocess(to_gray(image), self.input_size) for image in images]
        return np.stack(arrays)[:, None].astype(np.float32)

    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Embed grayscale/RGB images → ``(N, dim)`` float32 rows of unit L2 norm."""

        inputs = self.prepare(list(images))
        if len(inputs) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = [self._run(inputs[i : i + self.batch_size])
                for i in range(0, len(inputs), self.batch_size)]
        vectors = np.concatenate(rows).reshape(len(inputs), -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def __repr__(self) -> str:
        return f"FingerprintEmbedder(model_version={self.model_version!r}, dim={self.dim})"


def _resolve(name: str) -> Path:
    return model_path(name, download=True)
