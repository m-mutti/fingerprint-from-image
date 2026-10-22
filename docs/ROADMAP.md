# Roadmap — fingerprint-from-image v2

Goal: modernise the project into a well-tested, installable package whose
accuracy is **measured on public benchmarks** and **generalises across sensors**,
with ONNX Runtime and LiteRT models optimized for server, mobile and edge.

Work lands on `main` (the original code is preserved on the `backup` branch).
The v3 design contract is in [ARCHITECTURE.md](ARCHITECTURE.md): learned models
replace the classical pipeline end to end. Every phase must keep the existing CLI, Python API, and
SQLite template format working (or bump their version identifiers explicitly).

## Status

| Phase | Scope | Status |
|------:|-------|--------|
| 0 | Package restructure (`src/fingerprint_from_image`), legacy shims, `pyproject.toml`, console scripts, CI matrix (Linux/Windows/macOS × Py 3.10–3.13, staged in `ci/` until a `workflow`-scoped push), logo + animated mark | ✅ done |
| 1 | Public-dataset tooling and benchmark harness: FVC2000/2002/2004 DB1–4 B, SOCOFing; FVC-protocol genuine/impostor pairs; EER, FMR100, FMR1000, ROC/DET, latency | 🚧 in progress |
| 1b | Training-throughput engineering: profiling, GPU-resident data + batched GPU augmentation, bf16/TF32, `torch.compile` (Inductor → Triton), CUDA graphs, custom fused Triton kernels where profiling justifies them | 🚧 in progress |
| 2 | Learned fixed-length embedding (pretrained backbone + margin loss) trained on SOCOFing, evaluated cross-dataset on unseen FVC2004 sensors; replaces RootSIFT | 🚧 in progress |
| 3 | Edge deployment and custom optimization: ONNX (FP32 + calibrated INT8), LiteRT (FP16/INT8, XNNPACK/GPU/NNAPI delegates), graph optimization, knowledge distillation to a mobile student, parity tests vs PyTorch, `onnxruntime` backend | 🚧 in progress |
| 4 | Hand-photo extraction v3: learned hand-landmark fingertip localisation (ONNX Runtime / LiteRT) with anatomical labels; replaces HSV contours | 🚧 in progress |
| 5 | Docs, model card, benchmark tables in README, interactive demo | ⏳ planned |

## Principles

- **Measure first.** No accuracy claim without a reproducible benchmark command
  and the dataset split it ran on.
- **Generalise, don't overfit.** Train on one corpus (SOCOFing), report on
  disjoint sensors (FVC). Thresholds are calibrated on development data only.
- **Backwards compatible.** Database schema and feature versions are bumped
  whenever persisted features change; old databases are refused, never mixed.
- **Small, reviewable diffs.** One concern per commit, Conventional Commits.

## Datasets

| Dataset | Use | Licence / access |
|---------|-----|------------------|
| FVC2000 / 2002 / 2004 DB1–4 **B** (10 fingers × 8 impressions each) | Benchmark only | Free download from the University of Bologna BioLab site; research use |
| SOCOFing (600 subjects × 10 fingers, real + synthetic alterations) | Training | CC BY-NC-SA 4.0 — models trained on it are **non-commercial** |

Datasets are never committed; `scripts/download_datasets.py` fetches them into
the git-ignored `data/` directory.
