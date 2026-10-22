<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wordmark-dark.svg">
    <img src="assets/wordmark-light.svg" alt="fingerprint-from-image" width="460">
  </picture>
</p>

<p align="center">
  <b>Deep-learning fingerprint recognition from photos and scans, optimized for ONNX Runtime and LiteRT on server, mobile and edge.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB">
  <img alt="Deep learning" src="https://img.shields.io/badge/deep%20learning-PyTorch%20%E2%86%92%20ONNX-EE4C2C">
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/runtime-ONNX%20Runtime-005CED">
  <img alt="LiteRT" src="https://img.shields.io/badge/edge-LiteRT%20(TFLite)-FF6F00">
  <img alt="Optimized" src="https://img.shields.io/badge/optimized-INT8%20%7C%20FP16%20%7C%20distilled-8B5CF6">
  <a href="docs/ROADMAP.md"><img alt="Roadmap" src="https://img.shields.io/badge/roadmap-v3-14B8A6"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-animated.svg">
    <img src="assets/pipeline-animated-light.svg" width="800"
         alt="Pipeline: a hand photo gets 21 learned landmarks and five upright fingertip crops; the index crop is enhanced, embedded as a unit vector, and matched against an enrolled gallery by cosine similarity.">
  </picture>
</p>

---

`fingerprint-from-image` turns a **phone photo of a hand** or a **fingerprint
scan** into identity decisions:

1. a **learned hand-landmark detector** finds each fingertip and cuts an upright,
   anatomically labelled crop (`right_index`, `left_thumb`, …);
2. a **deep fingerprint embedding network** maps each crop to a compact,
   L2-normalised vector;
3. a **versioned SQLite gallery** stores vectors (never pixels), and
   identification is a single matrix product plus an evidence-aware decision rule.

Inference is a lean `numpy` + `Pillow` + `onnxruntime` stack. The same models
ship as **ONNX (FP32/INT8)** and **LiteRT** (formerly TensorFlow Lite) for mobile
and edge devices, with custom optimizations described
[below](#edge-deployment-and-optimization).

> [!IMPORTANT]
> **v3 is under active development.** The recognizer, embedding store and
> foundations are on `main`; the embedding model, hand detector and benchmark
> harness are being trained and evaluated now (see [Status](#status)). The
> original classical pipeline is still fully available
> ([docs/CLASSIC.md](docs/CLASSIC.md)), and the pre-v2 code is preserved on the
> [`backup`](https://github.com/m-mutti/fingerprint-from-image/tree/backup) branch.

## Why deep learning

| | Classical v2 (hand-crafted) | Learned v3 |
|---|---|---|
| Finding fingers in a photo | HSV skin-colour thresholds; fails with lighting, skin tone and clutter; unstable `finger_N` labels | learned hand landmarks, so fingertips are located and labelled anatomically |
| Ridge representation | adaptive thresholding + 1,600 RootSIFT keypoints | deep embedding trained for invariance to sensor, pressure, rotation and partial prints |
| Comparison | RANSAC geometry for every pair (milliseconds) | cosine similarity (microseconds); one matrix product per gallery |
| Template size | up to ≈ 830 KB of floats (1,600 × 130 float32) | ≤ 2 KB per finger (D ≤ 512 float32) |
| Deployment | CPU-only feature matching | ONNX Runtime (CPU/GPU/DirectML/CoreML) and LiteRT on Android, iOS and embedded |
| Accuracy claims | none measured | measured on public benchmarks with a held-out protocol |

## Edge deployment and optimization

Models are trained in PyTorch, then optimized for each target rather than
exported once and left alone:

| Technique | What it does | Target |
|---|---|---|
| **Post-training INT8 quantization** (static, calibrated on fingerprint data) | ~4× smaller weights, integer kernels; accepted only if verification accuracy holds | ONNX Runtime, LiteRT |
| **FP16 weights** | halves model size; accuracy impact verified by the parity check | LiteRT GPU delegate, mobile GPUs |
| **Graph optimization** (operator fusion, constant folding, shape simplification) | fewer kernels and memory copies per inference | ONNX Runtime, LiteRT |
| **Knowledge distillation** into a compact student backbone | keeps the large teacher's embedding geometry at a fraction of the compute | phones, embedded boards |
| **Hardware delegates** (XNNPACK, GPU, NNAPI / Core ML) | runs on the fastest unit available | LiteRT, ONNX Runtime execution providers |
| **Embedding-level optimization** (compact dimension, optional INT8 vectors) | smaller galleries and faster matrix search | the SQLite gallery |

Every optimized export must pass a **parity check** against the PyTorch
reference: cosine similarity ≥ 0.99, and benchmark EER re-measured on the
held-out set. Size, latency and accuracy for each variant will be published in
`benchmarks/RESULTS.md`.

## Pipeline

```text
 hand photo ─► hand landmarks (ONNX) ─► upright fingertip crops ─┐
                                                                 ├─► embedding net (ONNX) ─► unit vectors ─► SQLite v4 gallery
 fingerprint scan ───────────────────────────────────────────────┘                                      │
                                                                                                        ▼
                                  identify / verify: one-to-one finger assignment · quorum · ambiguity rejection
```

Design contract and module interfaces: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Brand assets and usage rules: [docs/brand.md](docs/brand.md). The animation above is
illustrative; real scores come from the benchmarks.

## Status

| Component | Module | State |
|---|---|---|
| Image I/O (Pillow, EXIF-aware) | `fingerprint_from_image.io` | ✅ on `main` |
| Model registry with SHA-256 verified cache | `fingerprint_from_image.models` | ✅ on `main` |
| Embedding gallery, SQLite schema v4 | `fingerprint_from_image.recognition` | ✅ on `main` |
| Recognizer: enroll / identify / verify | `fingerprint_from_image.recognition` | ✅ on `main` |
| Deep fingerprint embedding model + ONNX / LiteRT export | `fingerprint_from_image.embedding`, `training/` | 🧪 training |
| Learned hand / fingertip detector (ONNX) | `fingerprint_from_image.hand` | 🧪 in development |
| Benchmark harness (EER, FMR100/1000, CMC) + SOTA comparison | `fingerprint_from_image.evaluation`, `benchmarks/` | 🧪 in development |
| Classic RootSIFT pipeline (baseline) | `fingerprint_from_image` (`FingerprintMatcher`) | ✅ stable |

## Installation

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1   ·   macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quick start (learned pipeline)

```python
from fingerprint_from_image.recognition import FingerprintRecognizer

# `embedder` is any model with .model_version, .dim and .embed(images) -> (N, dim);
# the bundled ONNX FingerprintEmbedder ships with the first trained release.
with FingerprintRecognizer("gallery.db", embedder) as recognizer:
    recognizer.enroll("alice", "scans/alice_right_index.png", finger="right_index")
    best = recognizer.identify("queries/unknown.png", top_k=3)[0]
    print(best.identity, best.score, best.accepted)

    claim = recognizer.verify("alice", "queries/claim.png")
    print(claim.accepted, claim.matched_fingers)
```

Decisions follow evidence rules carried over from the classic matcher. Query
fingertips are assigned one-to-one to enrolled finger labels. Below-threshold
fingers add no evidence. A quorum of matched fingers is required. A near-tie
between two identities is rejected as ambiguous instead of guessed.

The classic command-line tools still work:

```bash
fingerprint-extract photos/hand.jpg --output-dir extracted --no-display
fingerprint-match build dataset --db fingerprints.db
fingerprint-match identify queries/unknown.png --cropped --db fingerprints.db
```

See [docs/CLASSIC.md](docs/CLASSIC.md) for the full classic CLI and JSON reference.

## Benchmarks and datasets

Accuracy is reported **only** from reproducible runs on public data, with a
protocol that cannot leak:

- **Held-out test:** FVC2004 DB1–4 B (four sensors, 10 fingers × 8 impressions
  each). It is never used for training, model selection or threshold tuning.
- **Training:** SOCOFing (600 subjects × 10 fingers, real + altered), split by
  subject, optionally plus FVC2000/2002 DB_B.
- **Metrics:** EER, FMR100, FMR1000, ZeroFMR, ROC/DET, CMC rank-1/5, latency,
  and failure-to-enrol rate. The classic RootSIFT matcher and published
  state-of-the-art results are reported side by side.

Results will appear in `benchmarks/RESULTS.md` as runs complete. No number is
published here until it has been measured.

| Dataset | Role | Licence |
|---|---|---|
| FVC2000 / 2002 / 2004 DB1–4 B | benchmark | free research download (University of Bologna BioLab) |
| SOCOFing | training | CC BY-NC-SA 4.0, so trained weights are **non-commercial** |

## Project layout

```text
src/fingerprint_from_image/
  io.py, types.py, models.py   lightweight foundations (image I/O, FingerCrop, model cache)
  recognition/                 schema-v4 embedding gallery + recognizer (learned pipeline)
  embedding/                   ONNX fingerprint embedder                     (in progress)
  hand/                        ONNX hand-landmark fingertip detector          (in progress)
  evaluation/                  biometric protocols and metrics               (in progress)
  extraction.py, features.py, pairwise.py, matcher.py, store.py   classic v2 pipeline
  cli/                         fingerprint-extract and fingerprint-match
training/                      PyTorch training, evaluation, ONNX / LiteRT export and optimization
benchmarks/                    benchmark results and SOTA comparison
scripts/                       reproducible dataset and brand-asset tooling
docs/                          ARCHITECTURE, ROADMAP, CLASSIC manual
```

## Testing

```bash
python -m pytest
ruff check .
```

Unit tests use synthetic images and stand-in models, so they run in seconds
without datasets or GPUs. They establish functional correctness, not biometric
accuracy. Accuracy comes from the benchmark protocol above.

## Privacy, security, and liveness

Fingerprint embeddings are sensitive biometric data. They are not guaranteed to
be irreversible or unlinkable across databases. The SQLite gallery is not
encrypted, and it stores identity labels and source paths. It is created with
owner-only permissions where the OS supports them, uses SQLite secure delete,
and verifies a checksum on every vector. Protect it with access control and
encryption, collect consent, set retention and deletion rules, and follow
applicable privacy law.

There is no liveness or presentation-attack detection yet. A photo, printout or
synthetic pattern may be accepted. Do not use this project as the sole control
for access, payments, safety-critical or other high-impact decisions. It has not
been evaluated for forensic use.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=m-mutti/fingerprint-from-image&type=Date)](https://star-history.com/#m-mutti/fingerprint-from-image&Date)
