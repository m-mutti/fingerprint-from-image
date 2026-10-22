# Model card — `fingerprint_embedder_v1`

_Status: draft. No v1 checkpoint exists yet: training restarts from scratch on the
manifest pipeline once the GPU is released (see `training/EXPERIMENTS.md`). Metrics are
filled from the registry (`python -m training.report`) after the long run._

## Summary

Fixed-length fingerprint embedding (256-d, L2-normalised) for verification and
identification by cosine similarity. Default backbone ConvNeXt-V2-Tiny (1-channel stem),
concat(avg, max) pooling → LayerNorm → Linear → BatchNorm, trained with ArcFace
(s 32, m 0.4) over finger identities, optionally with an in-batch SupCon term on P×K
batches. A DINOv2 ViT-S/14 variant is trained for the **open** track. Runtime:
`fingerprint_from_image.embedding.FingerprintEmbedder` (numpy + Pillow + onnxruntime).

## Two release tracks (licence lineage computed by `training/licences.py`)

| track | backbone weights | training data | use |
|---|---|---|---|
| research-only | ConvNeXt-V2 FCMAE (**CC-BY-NC-4.0**) or DINOv2 (Apache-2.0) | FVC2000/2002 (research use), SOCOFing (**CC-BY-NC-SA-4.0**), TDFD (non-commercial), L3-SF (CC-BY-4.0), generated synthetic prints | non-commercial research; share-alike obligations from SOCOFing |
| open | DINOv2 ViT-S/14 (Apache-2.0) | L3-SF (CC-BY-4.0, attribution required) + prints generated in this repo | releasable openly with L3-SF attribution |

Validation-only data (never trained on): FVC2000/2002 DB_B fingers 101–110, Neurotechnology
sample prints, L3-SF/SOCOFing val identities. Held-out test only: FVC2004 DB1–4 (A + B).
NIST SD300/SD302 were not used (they require accepting NIST terms via a request form).

## Intended use

* Research and non-commercial prototyping of fingerprint verification/identification on
  cropped fingerprint images (contact sensors, scanned prints, fingertip crops).
* Not for law-enforcement, border-control or other high-stakes decisions without
  independent, population-specific evaluation and calibrated thresholds.
* Not a liveness / presentation-attack detector.

## Input contract

Any grayscale or RGB fingerprint image. Preprocessing `pp-v3`
(`embedding/preprocess.py`, identical in training, evaluation and runtime; its version is
stored in the ONNX metadata and the runtime refuses a mismatch): trim uniform frame bars →
Welch ridge-period estimate (size prior: ≥ 16 ridges) → resample to 6 px per ridge →
square crop (256 px; 224 px for ViT) around the foreground centroid → masked z-score
(clip ±3, background → 0). Output `(N, D)` unit-norm float32.

## Evaluation protocol

Selection on manifest val identities only (pooled FNMR @ FMR 0.1 %, then mean EER).
Test, computed once per checkpoint: FVC2004 DB1–4 B (comparable with the classic and
SourceAFIS rows in `benchmarks/`) and FVC2004 full (110 fingers per DB; DB_A-only too).
Reference on FVC2004 DB_B (mean EER, all impostors): SourceAFIS 3.14 7.95 %, classic
RootSIFT 35.37 %; the lost legacy-protocol ConvNeXt model reached 20.24 %.

## Metrics

_Pending (FVC2004 DB_B and full; EER fvc/all, FMR100, FMR1000; per export variant)._

## Limitations

* Real multi-impression training data is limited to ~800 FVC fingers (+ 320 TDFD pairs);
  the rest is SOCOFing single captures with synthetic alterations, L3-SF and generated prints.
* Ridge-period normalisation can fail on very low-quality or tiny partial prints
  (2× sub-harmonic picks were observed on 2 / 80 FVC2004 DB2 images, not tuned on).
  Mendeley family prints (≈ 1229 dpi, thin ridges) are mis-estimated and are excluded.
* No minutiae-level interpretability; scores are not calibrated probabilities.
* Demographic performance differences have not been measured.
