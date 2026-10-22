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

## Scope re-evaluation against current SOTA (2026-10-22)

**Where we align.** v3 follows the established fixed-length deep-representation
line (DeepPrint, AFR-Net): a pretrained vision backbone (ConvNeXt-V2, ImageNet-22k)
fine-tuned with a margin-based identity loss, compact unit-norm embeddings, cosine
matching, and an FVC protocol with verified published numbers (`benchmarks/SOTA.md`).

**Where the 2024–2026 state of the art has moved, and what we change:**

| Gap | Recent evidence | Change to our plan |
|---|---|---|
| Global-only embeddings miss fine ridge detail | Dense fixed-length descriptors with alignment: FLARE (TIFS 2026, arXiv 2505.03597); LDRF (arXiv 2311.18576); IFViT (arXiv 2404.08237) | v3.1: two-stage matching, with a global embedding for retrieval plus a dense/local descriptor that re-ranks the top-K |
| No alignment | DeepPrint (spatial transformer); FLARE (pose-based alignment) | add a learned alignment (STN) / pose normalisation module and ablate it |
| Too little sensor diversity | SOTA trains on large multi-sensor and synthetic corpora; fingerprint foundation models use large-scale weakly supervised / unsupervised training (UoU, arXiv 2606.17436) | grow data with the open generator and L3SF synthetics; add domain-specific self-supervised pretraining on all unlabeled prints before identity fine-tuning |
| Backbone choice untested | General foundation backbones (DINOv2, Apache-2.0) adapt well to fine-grained texture tasks | backbone ablation: ConvNeXt-V2-T/B vs DINOv2 ViT-S/14 and ViT-B/14 (full fine-tune and LoRA, layer-wise LR decay) |
| Contactless phone photos | Ridgeformer (ICIP 2025, arXiv 2506.01806) uses multi-stage contrastive global→local training and releases weights (CC BY-NC 4.0) | use Ridgeformer as the contactless comparator; consider it as a warm start for the hand-photo path |
| Fair reference model | No official DeepPrint/AFR-Net weights are released | reproduce one published method under identical data and protocol (`baselines/`), with a documented deviations table |

**Pretrained models we can build on (availability verified 2026-10-22):**

| Model | What it gives | Weights | Licence | Use |
|---|---|---|---|---|
| ConvNeXt-V2 (timm) | ImageNet-22k CNN backbone | yes | CC BY-NC 4.0 (FCMAE weights) | current backbone |
| DINOv2 ViT-S/B/L | self-supervised ViT backbone | yes | Apache-2.0 | backbone ablation, preferred for permissive licensing |
| Ridgeformer | contactless↔contact transformer | yes (GitHub + HF) | CC BY-NC 4.0 | comparator, possible warm start |
| UoU | fingerprint foundation model (keypoints) | no (MIT training code only) | MIT | follow; nothing to load yet |
| FLARE | dense fixed-length descriptor | code announced, not confirmed | — | re-check later |
| MediaPipe Hand Landmarker | hand / fingertip localisation | yes | Apache-2.0 | already used in `hand/` |

Licence note: any CC BY-NC component (SOCOFing data, ConvNeXt-V2 FCMAE weights,
Ridgeformer) makes the resulting weights non-commercial; a DINOv2 backbone plus
openly licensed data is the path to a permissively licensed model.
