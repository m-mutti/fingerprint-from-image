# Fingerprint embedder — experiment log

Every number below comes from a command recorded next to it.

## Status (2026-10-22, evening)

* **All run artefacts were deleted** (the user removed `E:\Personal_tries\Mutti_Finger_Prints\runs\`
  at ~18:45: checkpoints, canvas caches, eval outputs). The legacy-protocol numbers below
  were measured before that and are kept from the command outputs; their checkpoints no
  longer exist and they are **not comparable** with the manifest protocol.
* Training restarts **from scratch** on the content-addressed pipeline (below), after the
  fair SOTA baseline (another agent) releases the GPU and after the design reviews
  (`docs/reviews/2026-10-22-*.md`) were folded in. No GPU job has run since.

## Protocol (manifest era — applies to every run from now on)

| item | choice |
|---|---|
| data | shared manifests `data/manifests/*.jsonl` (rows with `exclude: true` dropped; 157 cross-dataset duplicate groups resolved by the data agent) |
| held-out test | FVC2004 (`split == heldout`): DB1–4 B (10 fingers/DB, comparable with the classic/SourceAFIS table) **and** FVC2004 full (110 fingers/DB; DB_A = fingers 1–100). Never read by training, selection, thresholds or preprocessing choices; enforced in code (`training/guards.py`) |
| training | manifest `train` split; per-source epoch shares (`data.manifest_train`) |
| selection | manifest `val` identities only: contact FVC2000/2002 DB1–3 (fingers 101–110) + neurotech. **Primary** = pooled FNMR @ FMR 0.1 % with one global threshold; **secondary** = mean per-set EER; per-sensor EERs (also FVC DB4, l3sf, SOCOFing) logged every evaluation |
| test metrics | genuine = all same-finger pairs; impostor "fvc" = first impressions across fingers, "all" = every cross-finger pair; EER on both, FMR100/FMR1000/ZeroFMR on "all" (`training/eval_fvc.py`, `training/eval_full.py`, keyed in `training/evaluate.py`) |

## Legacy-protocol results (artefacts lost; for the record only)

Data: SOCOFing seed-0 train subjects (incl. CR alterations) + FVC2000 DB_B; selection on
FVC2002 DB_B (mean EER, all impostors). ConvNeXt-V2-T @256, ArcFace s32/m0.4, heavy aug,
B=128, stopped at epoch 12 of 30 (plateau). Commands:
`python -m training.summarize runs/embedder/convnext_heavy_20261022_175008` and
`python -m training.eval_full --checkpoint .../best.pt` (best = epoch 8).

| run | sel. epoch | FVC2002 EER all | FVC2004 DB_B EER fvc | FVC2004 DB_B EER all | DB_B FMR100 | DB_B FMR1000 | DB_B per-DB EER all | FVC2004 full EER all |
|---|---|---|---|---|---|---|---|---|
| `convnext_heavy_20261022_175008` | 8 | 15.59 | 21.36 | **20.24** | 63.0 | 83.3 | 17.2 / 26.6 / 21.4 / 15.7 | 19.05* |
| SourceAFIS 3.14 (reference) | — | 5.72 | 8.75 | **7.95** | 12.0 | 15.2 | 9.3 / 12.5 / 5.4 / 4.6 | — |
| classic v2 RootSIFT (reference) | — | 34.22 | 34.65 | **35.37** | 61.2 | 66.5 | 44.6 / 40.5 / 12.2 / 44.2 | — |

\*evaluated after the ridge size-prior change (20 → 16 ridges) that the checkpoint was not
trained with. Rotation TTA on FVC2002 (val): 15.59 → 15.51 % (3 angles) — not adopted.
Failure analysis on FVC2002 genuine pairs (scratch script): miss rate @ FMR 1 % 0.55 for
scale-matched impressions vs 0.73 when the estimated scales differ by > 10 % — the
representation, not preprocessing, is the main limit. ML review diagnosis: 89 % of the
SOCOFing training images are pixel-aligned alterations of one capture; the fix is
multi-impression data (manifest era).

## Data fixes (before any real training)

### Resolution gap → ridge-period normalisation

Spectral ridge-period estimates (median px per ridge): SOCOFing ≈ 3.0, FVC sets 7–13.
SOCOFing is ~3× down-sampled relative to 500 dpi. Every image is resampled so its
ridge period is **6 px**, then a 256×256 window is cut around the foreground centroid
and masked-z-score normalised (`src/fingerprint_from_image/embedding/preprocess.py`,
shared by training cache, evaluation and runtime). Training adds ±0.3 log-scale jitter
(≈ ×0.74–1.35) to absorb estimator noise.

Estimator robustness (vision-QA finding: ~5.5 % of SOCOFing images, mostly Altered-Obl,
got 10–24 px and shrank to a blob). Variants compared on SOCOFing train images
(every 4th Real, every 12th Altered) and FVC2000+FVC2002 DB_B (8 sets × 10 fingers × 8
impressions); fail = estimate > 4.5 px on SOCOFing; FVC columns = mean within-finger
relative std and 95th percentile of the per-finger max/min ratio.
`python -m training.period_study --no-2004`:

| estimator | SOCOFing Real fail | SOCOFing Altered fail | FVC within-finger rel. std | FVC p95 max/min ratio |
|---|---|---|---|---|
| global spectrum (v1) | 1.25 % | 6.79 % | 0.055 | 1.43 |
| global + size prior (≥ 20 ridges across the long side) | 0.92 % | 1.00 % | 0.057 | 1.43 |
| **Welch (unit-energy block spectra) + size prior (adopted)** | **0.75 %** | **0.88 %** | 0.037 | 1.29 |
| Welch without prior | 1.83 % | 6.67 % | 0.033 | 1.24 |

The choice uses SOCOFing and FVC2000/2002 only. Disclosure: an earlier scratch run of the
same comparison pooled all 12 FVC sets (incl. FVC2004) and gave the same ranking; it also
showed two FVC2004 DB2 impressions where Welch picks a 2× sub-harmonic (17.5 vs 8.5 px).
The estimator was **not** changed in response. In addition, SOCOFing Altered images (pixel-aligned edits of their
Real image) inherit the Real image's period in the training cache, and uniform
sensor-frame bars are trimmed before estimation (`trim_uniform_border`).

### SOCOFing duplicate prints (label noise + split leakage)

Ridge-band NCC (image minus Gaussian blur σ=2, 90×96) over all 6000 Real images
(`training/dedup.py`): 31 pairs > 0.5, next-highest pair 0.43 — a clean gap. Union-find gives
20 groups (e.g. 203/219/249/260/64 Left_little; 418/204/92; 511↔590/553/577). Each group is
owned by its most held-out split (test > val > train); copies in other splits are dropped and
same-split members share one label: 24 classes removed, 7 labels dropped across splits.
Final SOCOFing classes: train 4785, val 592, test 599. Training classes = 4785 + 40 FVC2000
fingers = **4825**.

### Augmentation (GPU, batched)

One fused `grid_sample` (rotation ±45°, log-scale ±0.3, anisotropy, translation ±12 %,
4×4 elastic field), partial-print windows and cutouts (keeping ≥ 35 % of the print area),
synthetic sensor-frame edges, wet/dry morphology, resolution loss, blur, unsharp masking,
gamma/contrast, illumination field and sensor noise **only on real pixels** (padding stays
constant as at inference), polarity inversion, then the torch port of the runtime
normaliser (`normalize_intensity_torch`, max |Δ| vs numpy 7e-7 in float32).

## Throughput phase (RTX 5060 Ti 16 GB, torch 2.11+cu128, triton-windows 3.7.1)

`python -m training.bench --batch B --compile MODE` — full step: pinned-RAM uint8 canvas
gather → H2D → GPU augmentation → bf16 autocast fwd/bwd → (fused) AdamW → EMA. CUDA-event
timing after 15–25 warm-up steps; GPU util sampled with NVML every 50 ms. ConvNeXt-V2-Tiny,
256×256 input, 5440-class ArcFace.

| # | config | step ms | img/s | GPU util % | peak VRAM GB |
|---|---|---|---|---|---|
| 1 | eager, B=96, AdamW (foreach), fp64 normaliser | 946 | 101.5 | 67.4 | 15.61 (exceeds free VRAM → WDDM spill) |
| 2 | eager, B=64 | 469 | 136.6 | 92.8 | 10.97 |
| 3 | eager, B=64, fused AdamW | 445 | 143.9 | 98.2 | 10.97 |
| 4 | eager, B=64, fused AdamW, no augmentation | 423 | 151.4 | 98.9 | 10.76 |
| 5 | torch.compile (whole model), B=64 | 165 | 387.8 | 93.2 | 5.38 |
| 6 | torch.compile (whole model), B=128 | 325 | 393.3 | 97.6 | 10.13 |
| 7 | #6, contiguous instead of channels_last | 324 | 394.8 | 97.6 | 10.14 |
| 8 | #6, no augmentation | 281 | 456.2 | 98.5 | 10.10 |
| 9 | #6 + float32 normaliser in augmentation | 306 | 418.8 | 98.0 | 10.12 |
| 10 | **compile backbone only (adopted), B=128, fp32 normaliser** | 311 | **411.7** | 95.5 | 10.14 |
| — | `max-autotune` | aborted: > 10 min autotuning with the GPU idle; Inductor reports "Not enough SMs to use max_autotune_gemm mode" on this 36-SM part | | | |

Findings:

* The step was **compute-bound** from the start (98 % util at B=64); the data path
  (whole decoded dataset preloaded as uint8 canvases in pinned RAM, augmentation on GPU)
  was never the bottleneck — no DataLoader workers are used.
* Eager ConvNeXt is dominated by unfused memory-bound ops (channels-last LayerNorm,
  GRN, GELU). Inductor fusion gives **2.7×** and halves activation memory.
* Augmentation stage timings at B=128 (CUDA events): geometry 7.2 ms, occlusion 3.9,
  texture 6.6, photometric 3.9, normaliser 21.7 ms in float64 → 4.3 ms in float32
  (consumer GPUs run fp64 at 1/64 rate); total 44 → 26.5 ms (≈ 8 % of a step).
* torch.profiler cannot record CUDA kernels on this machine (CUPTI initialisation fails
  under the Windows driver), so kernel attribution used CUDA-event A/B timing instead.
* Custom Triton kernels were **not** written: the margin-softmax head is
  128 × 256 × 4825 ≈ 0.3 GFLOP per step (< 1 % of the step), and augmentation is ≈ 8 %;
  neither justifies a hand-written kernel against the 2.7× already obtained.
* TF32 matmul (`set_float32_matmul_precision("high")`), `cudnn.benchmark`, bf16 autocast
  and fused AdamW are on in `training/train.py`. LR scaled 4e-4 → 5e-4 for B 96 → 128.

### NaN-loss root cause (smoke run `smoke_20261022_170707`)

Bisection (`scratchpad/nan3.py`, 5 steps each, same batch stream): eager → finite;
`torch.compile(net)` (backbone + head in one graph) → **all 202 parameter gradients NaN at
step 0**, forward finite; compile backbone only → finite; compile head only → finite;
compiling without channels_last → still NaN. Root cause: an Inductor/Triton codegen bug in
the backward of the fused backbone→head graph (amax/mean pooling + LayerNorm on the
channels-last ConvNeXt output) on torch 2.11 / triton-windows 3.7 / sm_120 — not the data,
the loss, or bf16. Fix: compile `backbone.forward_features` only (head stays eager;
throughput 411.7 vs 418.8 img/s). Guard: non-finite gradient norm → the update is skipped
and a `{"kind": "nonfinite"}` record is logged; > 20 such steps abort the run.

### Eager vs compiled equivalence (fixed seed, 2 × 100 steps)

Same config/seed/data stream, 2 epochs × 100 steps (B=128), only `compile` differs:

```
python -m training.train --config training/configs/convnext_heavy.yaml --set name=ab_eager \
  --set compile=none --set optim.epochs=2 --set optim.samples_per_epoch=12800 --set eval_every=2 --set log_every=10
python -m training.train ... --set name=ab_compiled --set compile=default ...
```

| run | epoch-1 loss | epoch-2 loss | FVC2002 EER(all) after 2 ep | median img/s (logged) | median GPU util |
|---|---|---|---|---|---|
| `ab_eager_20261022_173613` | 9.62 | 14.26 | 29.46 % | 142.7 | 100 % |
| `ab_compiled_20261022_174244` | 9.48 | 14.58 | 29.31 % | 334.1* | 80 %* |

Per-step losses (26 logged steps): Pearson r = 0.992, mean |Δloss| = 0.32 (losses 9–17);
differences come from dropout/drop-path RNG streams inside the compiled graph. Curves are
unchanged. (Loss rises in epoch 2 because the ArcFace margin switches on.)
\*The compiled leg overlapped with a CPU-heavy ONNX export test; uncontended throughput is
the 411.7 img/s benchmark above. One-off compile cost ≈ 4 min (epoch 1: 304 s).

## Pipeline (docs/TRAINING_PIPELINE.md, implemented in `training/`)

| piece | module | behaviour |
|---|---|---|
| content keys + `done.json` | `keys.py` | step key = SHA-256 of canonical JSON inputs; outputs hashed; verified skip; `invalidate` |
| canvas cache | `data.build_cache` | key = image SHA-256 list (manifest) / path+size+mtime (generated) + preprocessing version + **preprocessing source hash** + params |
| train | `train.py`, `trainer.py` | key = resolved config + cache key + **content hash of tracked+untracked `training/` and `src/` .py files** + held-out digest + resolved batch; skip if done; resume from `state.pt` (model, EMA, classifier, optimizer, counters, guards, sampler + all RNGs) — exact on CPU (test) |
| guards | `guards.py`, `sources.py`, `evaluate.py` | training refuses held-out identities/paths/hashes; val identities may not appear in training; selection reads `val` rows only; evaluation refuses checkpoints without a clean held-out record; metrics refuse NaN |
| queue | `queue.py` | `submit` / `submit-module` (any GPU job, e.g. the baseline) / `worker` / `status` / `rerun`; priorities; crash recovery (running → pending → resume); heartbeat stall guard (20 min); reasons oom / divergence / nonfinite / stall / exit |
| registry + leaderboard | `registry.py`, `report.py` | `runs/registry.sqlite`; `python -m training.report` → `benchmarks/LEADERBOARD.md` + plots, with licence lineage (`licences.py`: open vs research-only) |
| sweeps | `sweep.py`, `sweeps/hp_v1.yaml` | successive halving over queue jobs, ranked on val only, idempotent |
| export | `export.py`, `export_litert.py` | variants accepted on **val** (parity ≥ 0.99, pooled val metrics within 10 %); FVC2004 reported once per accepted file; `preprocess_version` in ONNX metadata, runtime refuses mismatches |

### Batch-size policy (user requirement)

* Requested B = 96 (ML review: B = 128 peaked at 16.0 / 16.3 GB in the legacy run, at the edge
  of WDDM spill). `batch_probe.py` (subprocess, static compile as in training, process capped
  with `set_per_process_memory_fraction`) measures peak reserved VRAM for ascending
  multiples of K up to the request, extrapolates and never runs a candidate predicted above
  **80 % of 16.3 GB**; the largest fitting B ≤ request is used and recorded in `done.json`.
  If B < requested, gradient accumulation keeps the effective batch.
* P×K sampler (`sampler.py`): P identities × K images, identities drawn per source share;
  sources with < K images repeat images (fresh augmentation views).
* LR = `optim.lr` (5e-4 at B = 128) × sqrt(B_eff / 128); warm-up ≥ 1 epoch; clip 5.0 over
  backbone + head + classifier; EMA 0.9995 with warm-up.
* Stability: per-step loss mean/std, grad-norm, img/s, GPU util and peak VRAM logged every
  50 steps; non-finite loss is checked **before** backward (BN/AdaFace buffers restored,
  step skipped, abort after 20); divergence guard (window loss > 2× best for 3 windows);
  early stop after epoch 40 if the pooled val metric does not improve in 3 evaluations.
* Chosen B, peak VRAM and img/s per run: _pending the first GPU run_.

### Review responses

| review item | status |
|---|---|
| code H1 stale heartbeat kills resumed jobs | fixed (heartbeat unlinked, `max(start, mtime)`), test |
| code H2 code key | fixed (content hash of tracked + untracked `.py`), test |
| code H3 NaN poisons BN/EMA/validation | fixed (finite check before backward, buffer restore, metrics/validator raise), tests |
| code H4 probe spill / leaks | fixed (subprocess, static compile, memory cap, extrapolation, `empty_cache` after val) |
| code M1–M5, L3, L4, L6, L7, L8 | fixed; L1 (uint8 cache rounding) and L2 left as documented |
| ML #1 data construction | `socofing_drop_variants: [CR]`, review shares, mendeley excluded, synthetic_v2 (8 impr./id, 4000 ids, generating on CPU) |
| ML #2 selection metric | pooled FNMR@FMR0.1 % + mean EER; `stats.paired_bootstrap`; A2 seed repeat |
| ML #3 positives in batch | P×K (K=4) + SupCon/MS aux loss (A3); AdaFace switch (A5) |
| ML #4 distortion aug | A4 retune (elastic 0.06, translate 0.15, partial 0.4, polarity 0); no sweep-sensor aug |
| ML #7 schedule | B 96 via probe, EMA 0.9995, eval every 4, early stop |

## Plan (queued when the GPU is released)

1. Ablations A0–A5 (`training/configs/a*.yaml`, 8 epochs × 40k, one variable each; A2 = seed
   noise floor) → adopt a change only if it beats A1 on both val metrics by more than A2's
   seed difference (or its bootstrap CI excludes 0).
2. Backbone: B0 ConvNeXt-V2-T @224, B1 DINOv2-S/14, B2 DINOv2-B/14 (Apache-2.0) on the A1 mix.
3. `sweeps/hp_v1.yaml` (24 sampled points, rungs 4 → 8 → 12 epochs) on the winner.
4. `v1_long` (≤ 80 epochs, early stop) → `python -m training.evaluate` (FVC2004 DB_B + full)
   → export + parity → leaderboard.
5. Open track: `open_dinov2_s.yaml` (DINOv2-S + L3-SF CC-BY + our synthetic prints only),
   reported next to the research-licensed model.

## Ablations (manifest protocol)

_pending — the GPU is owned by the baseline run._
