# Design review — ML method vs SOTA (2026-10-22)

Independent read-only review before the single long training run. Paper claims
cite arXiv id + section/table; run numbers come from `plan/task_plan.md`
(the `runs/` folder was deleted by the user during the review).

## Verdict
Do not launch the long run on the current recipe. The cross-sensor gap is a
**training-data construction** problem, not a backbone problem. Highest gain per
GPU-hour: (1) re-weight to multi-impression identities, (2) a selection metric
that can resolve differences, (3) identity-balanced batches + a positive-pair
loss, (4) realistic distortion augmentation. Alignment, a minutiae auxiliary head
and DINOv2 are well supported but second-order for this run.

## Findings (ranked)
1. **P0 — signal is mostly alteration copies, not impressions.** 89 % of SOCOFing
   training images are pixel-aligned edits of a single real capture (CR alone 30 %,
   which teaches ignoring the core). Legacy run: SOCOFing-val 0.48 % vs FVC2002-val
   21.7 % at epoch 2; held-out FVC2004 DB_B mean EER 20.2 %, FMR1000 83 %
   (SourceAFIS 7.83 % / 16.49 %). Literature: DeepPrint trains on 455K rolled prints
   of 38,291 fingers (~12 impressions/finger; arXiv 1909.09901 §3.1); AFR-Net gains
   92.50 → 97.82 TAR@0.1 % on FVC2002 DB3A from 760K → 1.3M images (arXiv
   2211.13897 Table 5); PrintsGAN pretraining + SD302 fine-tuning lifts NIST SD4
   TAR@0.01 % 73.37 → 87.03 (arXiv 2201.03674). **Fix:** manifest data path; drop
   SOCOFing CR; cap SOCOFing ≈ 25 % of samples (kept as negatives); ≈ 60 % from
   multi-impression sources (FVC2000/2002 DB_A, l3sf, TDFD); exclude
   single-impression mendeley_family; regenerate synthetics with ≥ 8 impressions/id.
2. **P0 — selection metric too noisy.** 10 fingers per FVC val set; 1–2-point
   differences between short runs are not significant. **Fix:** primary selection =
   pooled FNMR @ FMR 0.1 % with one global threshold over the selection sets;
   secondary = mean EER; finger-level paired bootstrap; one repeat seed to measure
   the noise floor; adopt a change only if it beats both.
3. **P1 — no positive pairs in a batch.** Independent multinomial sampling over
   ~5.6k classes at B = 128. **Fix:** P×K identity-balanced sampler (K = 4) +
   auxiliary in-batch pair loss (SupCon, arXiv 2004.11362, or Multi-Similarity,
   arXiv 1904.06627; weight ≈ 0.5); keep ArcFace s 32 / m 0.4 (IFViT uses m 0.4,
   arXiv 2404.08237 §III-C); AdaFace (arXiv 2204.00964) as a config-only ablation;
   skip sub-center ArcFace (labels are deduplicated).
4. **P1 — augmentation texture-heavy, distortion weak.** Elastic ≈ 0.75 of a ridge
   period. **Fix:** elastic ≈ 0.06, translate 0.15, partial 0.4, polarity 0
   (no contact sensor inverts), keep lowres 0.35, add structured noise. Do NOT add a
   sweep-sensor-specific augmentation because FVC2004 DB3 is thermal sweep — that
   would be test-set tuning.
5. **P2 — ridge-period sub-harmonic failures** (2/80 FVC2004 DB2 images, disclosed).
   Do not tune on FVC2004; later add multi-scale TTA validated on val only.
6. **P2 — architecture:** keep ConvNeXt-V2-T @256 (42.7 ridges ≈ 19.9 mm window;
   covers DB2–DB4 frames fully, DB1 ≈ 61 % × 82 %). Strongest literature lever for
   v3.1 is a **minutiae-map auxiliary head**: DeepPrint FVC2004 DB1A TAR@0.1 % 72.86
   (no alignment/minutiae) → 88.0 (+alignment) → 97.5 (+minutiae; Table 12).
   STN: AFR-Net 97.82 → 98.43 (Table 5). DINOv2: no verified fingerprint evidence;
   ImageNet ViT-S ≈ ResNet-50 in AFR-Net Table 3; worthwhile mainly for licensing.
7. **P2 — schedule:** B = 96 (B = 128 peaked at 16.0/16.3 GB); plan on ≈ 250 img/s;
   EMA 0.9995; evaluate every 4 epochs; early-stop if the pooled val metric does not
   improve for 3 evaluations after epoch 40.

## Ablations before the long run (8 epochs × 40k samples each, ≈ 3 GPU-hours)

| # | Run | Hypothesis |
|---|---|---|
| A0 | `m1_fvcfull` as-is | reference for the manifest pipeline |
| A1 | new data mix (no CR, + l3sf/TDFD/synthetics, SOCOFing ≈ 25 %) | multi-impression identities drive impression invariance (largest gain) |
| A2 | A1, seed 1 | noise floor = minimum effect size |
| A3 | A1 + P×K (K = 4) + SupCon/MS aux loss | positives in-batch fix the genuine-score tail (FMR1000) |
| A4 | A1 + distortion-focused aug retune | realistic geometry beats photometric noise |
| A5 | A1 + AdaFace | quality-adaptive margin helps dry/wet prints |

Then one long run (≈ 80 epochs × 40k ≈ 3.6 h at 250 img/s) of the best combination,
selected on the pooled val metric only.
