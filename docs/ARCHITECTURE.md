# v3 architecture — learned, OpenCV-free fingerprint pipeline

**Status:** design contract for the v3 rewrite (2026-10-22). v2 (classical
HSV + RootSIFT + RANSAC) remains available as the `classic` baseline and on the
`backup` branch; v3 replaces it as the default once it beats it on public
benchmarks.

## Why move away from the classical pipeline

| Classical v2 component | Failure mode | v3 replacement |
|---|---|---|
| HSV skin thresholding | skin tone, white balance, background clutter; no fingertip localisation; unstable `finger_N` labels | learned hand-landmark detector → distal-phalanx crops with anatomical labels (`right_index`, …) |
| Adaptive thresholding | destroys ridge detail on phone photos | learned-invariance embedding on normalised grayscale (no binarisation) |
| RootSIFT + RANSAC (1600 × 128 floats per template, O(N) pairwise geometry per comparison) | weak on repetitive ridges, partial prints, cross-sensor; slow identification | fixed-length L2-normalised **deep embedding** (D ≤ 512); identification = one matrix product |
| OpenCV runtime dependency | heavy, platform quirks | `numpy` + `Pillow` + `onnxruntime` only; models portable to ONNX / TFLite |

## Pipeline

```text
image ──► io.load_image (Pillow) ──┬─► cropped print ───────────────────────────┐
                                   └─► hand photo ─► hand.HandFingerDetector ───┤ list[FingerCrop]
                                                                                ▼
                                              embedding.FingerprintEmbedder (ONNX Runtime)
                                                                                ▼
                                             (N, D) float32, L2-normalised embeddings
                                                                                ▼
                     store (SQLite schema v4: embeddings + metadata, no pixels) ─► cosine scores
                                                                                ▼
                         identify / verify (calibrated thresholds, one-to-one finger assignment)
```

## Module contracts

All new modules live under `src/fingerprint_from_image/` and must not import
`cv2`. Images are `numpy.ndarray`: grayscale `(H, W) uint8` or RGB
`(H, W, 3) uint8` (RGB order, not BGR).

```python
# io.py
def load_image(path: str | Path, *, mode: Literal["L", "RGB"] = "L") -> np.ndarray: ...

# types.py
@dataclass(frozen=True)
class FingerCrop:
    image: np.ndarray            # (H, W) uint8 grayscale, fingertip upright
    label: str                   # "right_index", ..., or "unknown"
    quality: float               # [0, 1]
    polygon: tuple[tuple[float, float], ...] = ()   # crop corners in source pixels
    source: str | None = None

# hand/  — learned hand + fingertip localisation
class HandFingerDetector:
    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None): ...
    def __call__(self, rgb: np.ndarray) -> list[FingerCrop]: ...

# embedding/ — learned fixed-length representation
class FingerprintEmbedder:
    model_version: str           # persisted in the store; mismatches are refused
    dim: int
    def __init__(self, model_path: Path | None = None, *, providers: Sequence[str] | None = None): ...
    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray: ...   # (N, dim) float32, unit norm

# evaluation/ — numpy-only biometric metrics and protocols
def fvc_pairs(n_fingers: int, n_impressions: int, *, impostor: Literal["fvc", "all"]) -> Pairs: ...
def verification_metrics(genuine: np.ndarray, impostor: np.ndarray) -> VerificationMetrics: ...
    # EER, FMR100, FMR1000, ZeroFMR, TAR@FAR=1e-2/1e-3, ROC/DET points
def identification_metrics(scores: np.ndarray, labels_q, labels_g) -> IdentificationMetrics: ...
    # CMC rank-1/5/10
```

## Model training and release

- Training code lives in `training/` (outside the runtime package; PyTorch
  only there). The runtime never imports torch.
- Backbone: modern pretrained vision backbone fine-tuned with a margin-based
  identity loss (ArcFace/CosFace family) on subject-disjoint splits, with heavy
  sensor/acquisition augmentation (contrast, blur, noise, elastic/perspective
  warp, partial occlusion, polarity inversion, resolution change) for
  cross-sensor robustness.
- **Primary metric:** cross-dataset EER / TAR@FMR=1e-3 on held-out FVC2004
  DB1–4 B (never used for training or threshold selection).
- Exports: ONNX FP32, ONNX INT8 (static or dynamic quantisation), TFLite
  FP16/INT8; parity tests compare each export with the PyTorch reference
  (cosine ≥ 0.99 on held-out images).
- Model cards record data, licences (SOCOFing is CC BY-NC-SA 4.0 → weights are
  non-commercial), metrics, and known limitations.

## Compatibility rules

- Store schema bumps to **v4**; v3 (RootSIFT) databases are refused with a
  clear message rather than mixed.
- `fingerprint_matching.py` / `fingerprint_extraction.py` shims keep working
  and keep the classic behaviour until the v3 matcher is promoted.
- The classic matcher stays importable as an optional extra
  (`pip install fingerprint-from-image[classic]`) for baseline comparison.
