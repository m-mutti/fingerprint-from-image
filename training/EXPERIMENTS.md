# Fingerprint embedder — experiment log

Every number below comes from a command recorded next to it. Run artefacts
(checkpoints, `metrics.jsonl`, `train.log`, `plots/`) live in
`E:\Personal_tries\Mutti_Finger_Prints\runs\embedder\<run>`; `runs/embedder/LATEST`
points at the active run.

## Protocol

| item | choice |
|---|---|
| held-out test | FVC2004 DB1–4 B — never used for training, checkpoint selection, thresholds or preprocessing choices |
| training data | SOCOFing train subjects (shared seed-0 split, `benchmarks/splits/socofing_subject_split_seed0.json`: 480 train / 60 val / 60 test subjects) Real + Altered, after duplicate merging (below); FVC2000 DB1–4 B (40 fingers × 8 impressions) |
| model selection | FVC2002 DB1–4 B, mean EER over all impostor pairs (best epoch of the EMA weights). SOCOFing-val EER (Altered-Hard probes vs Real gallery of the 60 val subjects) is logged but not used for selection |
| FVC metrics | per set: genuine = 280 same-finger pairs; impostor "fvc" = 45 first-impression pairs; impostor "all" = 2880 cross-finger pairs; EER on both, FMR100/FMR1000/ZeroFMR on "all". `training/eval_fvc.py` (self-contained) |
| SOCOFing test split | never touched by this work |

Discard every metric from `runs/embedder/smoke_20261022_162949` (random 540/60 subject split
that did not match the shared split, and no duplicate handling).

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
(every 4th Real, every 12th Altered) and all FVC images
(`scratchpad/period_eval.py`; fail = estimate > 4.5 px on SOCOFing):

| estimator | SOCOFing Real fail | SOCOFing Altered fail | FVC within-finger rel. std | FVC p95 max/min ratio |
|---|---|---|---|---|
| global spectrum (v1) | 1.25 % | 6.79 % | 0.061 | 1.49 |
| global + size prior (≥ 20 ridges across the long side) | 0.92 % | 1.00 % | 0.061 | 1.47 |
| **Welch (unit-energy block spectra) + size prior (adopted)** | **0.75 %** | **0.88 %** | **0.045** | **1.33** |
| Welch without prior | 1.83 % | 6.67 % | 0.042 | 1.28 |

The choice is made on SOCOFing and FVC2000/2002 behaviour. Disclosure: the FVC rows
above pooled all 12 FVC sets including FVC2004; a post-hoc look showed two FVC2004 DB2
impressions where Welch picks a 2× sub-harmonic (17.5 vs 8.5 px). The estimator was **not**
changed in response. In addition, SOCOFing Altered images (pixel-aligned edits of their
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

## Ablations (FVC2002 = selection, FVC2004 = held-out test)

_pending._
