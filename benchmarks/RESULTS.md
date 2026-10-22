# Benchmark results — FVC2000 / 2002 / 2004 DB_B

Every number on this page comes from `benchmarks/results/*.json`, which this
repository's code produced. Every set was run on one CPU process (environment
below). For published state-of-the-art numbers and why they are **not** directly
comparable, see [SOTA.md](SOTA.md).

## How to reproduce

```bash
uv venv .venv -p 3.12 && uv pip install -e ".[dev]"
python scripts/download_datasets.py                     # 12 FVC DB_B zips -> data/fvc/
python -m fingerprint_from_image.evaluation --data-root data --sets all --scorer classic
#   (same as `fingerprint-bench ...` once the console script is registered)
python benchmarks/comparators/sourceafis.py --data-root data --sets all   # needs .NET SDK >= 10
# a learned model: any object with .embed(list[np.ndarray]) -> (N, D) unit-norm array
python -m fingerprint_from_image.evaluation --scorer embedding \
    --embedder fingerprint_from_image.embedding:FingerprintEmbedder --tag v3
```

## Protocol

* **Data.** FVC2000/2002/2004 DB1–DB4, set **B**: 10 fingers × 8 impressions per set,
  80 images per set, 12 sets.
* **Genuine pairs.** Every unordered impression pair of the same finger, 280 per set.
  The symmetric comparison is never run, following FVC.
* **Impostor pairs.**
  * `FVC imp.`: the official protocol. Impression 1 of each finger is compared
    with impression 1 of every other finger, giving 45 pairs.
  * `all imp.`: all 2,880 cross-finger pairs. **FMR100, FMR1000 and ZeroFMR in the
    tables use this set.** With only 45 impostors, FMR ≤ 1% is the same as FMR = 0.
* **Failures.** A failure to enrol (FTE) or a failed comparison gets score `-inf`, so
  it is always rejected. These failures are counted in FNMR and reported in the FTE column.
* **EER.** The exact crossing of the linearly interpolated ROC. The JSON also stores
  the FVC `[EERlow, EERhigh]` interval and its midpoint (`eer_fvc`). FMR100 and FMR1000
  are the lowest FNMR with FMR ≤ 1% / 0.1% (FVC definition). Rank-1 is closed-set
  identification with impression 1 as the gallery (10 identities) and impressions
  2–8 as probes (70 per set). Ties count against the probe.
* **Latency.** Single process, image decoding excluded. "Enrol" is template extraction
  per image. "Compare" is one one-to-one comparison.
* **Statistical caveat.** Each set has 280 genuine scores from only 10 fingers, and the
  scores are correlated. An EER difference of a few points on one set is not
  significant. Compare methods on the 12-set average, or better, on pooled evidence.
  The JSON files keep the raw scores so you can bootstrap over fingers.

## Headline

| 12-set average | FTE % | EER % (FVC imp.) | EER % (all imp.) | FMR100 % | FMR1000 % | ZeroFMR % | Rank-1 % | Enrol ms/img | Compare us/pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classic (v2 RootSIFT + RANSAC) | 1.15 | 33.12 | 33.36 | 55.89 | 57.65 | 57.98 | 46.67 | 36.1 | 9226 |
| SourceAFIS 3.14.0 (.NET) | 0.00 | 7.97 | 7.83 | 11.37 | 16.49 | 18.42 | 92.26 | 27.7 | 4187 |

**Targets for the v3 learned embedding.** It should beat both rows on the 12-set
average EER and FMR1000, with FVC2004 DB1–4 B as the primary held-out signal.
Training or threshold calibration must never touch FVC.

## Classic baseline (`classic`, v2 RootSIFT + affine RANSAC) — committed in `results/classic_*.json`

| Set | FTE % | EER % (FVC imp.) | EER % (all imp.) | FMR100 % | FMR1000 % | ZeroFMR % | Rank-1 % | Enrol ms/img | Compare us/pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FVC2000_DB1_B | 0.00 | 18.37 | 18.37 | 22.50 | 22.50 | 22.50 | 70.00 | 19.8 | 7683.2 |
| FVC2000_DB2_B | 1.25 | 23.96 | 23.82 | 30.71 | 30.71 | 30.71 | 67.14 | 23.1 | 9365.4 |
| FVC2000_DB3_B | 7.50 | 29.32 | 29.82 | 40.71 | 40.71 | 41.07 | 57.14 | 48.4 | 7848.1 |
| FVC2000_DB4_B | 0.00 | 49.91 | 49.91 | 99.64 | 99.64 | 99.64 | 0.00 | 19.9 | 11024.0 |
| FVC2002_DB1_B | 0.00 | 25.73 | 25.73 | 34.64 | 34.64 | 34.64 | 68.57 | 48.6 | 13359.7 |
| FVC2002_DB2_B | 0.00 | 17.89 | 17.89 | 21.79 | 21.79 | 21.79 | 88.57 | 55.4 | 10474.6 |
| FVC2002_DB3_B | 1.25 | 43.71 | 43.24 | 75.71 | 75.71 | 75.71 | 31.43 | 21.7 | 6397.6 |
| FVC2002_DB4_B | 0.00 | 50.00 | 50.00 | 100.00 | 100.00 | 100.00 | 0.00 | 28.1 | 7985.6 |
| FVC2004_DB1_B | 3.75 | 41.70 | 44.56 | 78.93 | 78.93 | 78.93 | 25.71 | 84.9 | 7011.4 |
| FVC2004_DB2_B | 0.00 | 40.45 | 40.47 | 72.86 | 93.93 | 95.36 | 40.00 | 25.7 | 7299.8 |
| FVC2004_DB3_B | 0.00 | 12.23 | 12.23 | 13.93 | 13.93 | 16.07 | 91.43 | 30.9 | 10799.3 |
| FVC2004_DB4_B | 0.00 | 44.22 | 44.22 | 79.29 | 79.29 | 79.29 | 20.00 | 26.5 | 11458.7 |
| **Average** | 1.15 | 33.12 | 33.36 | 55.89 | 57.65 | 57.98 | 46.67 | 36.1 | 9225.6 |

### How to read the classic numbers

* **The score is hard-gated.** `match_templates` returns exactly `0.0` whenever any
  gate fails (mutual ratio-test matches < 8, RANSAC inliers < 6, affine scale, spatial
  coverage). Only **68 of 34,560** impostor pairs score above 0, 66 of them on
  FVC2004 DB2. Between 0% (FVC2002 DB4) and 86% (FVC2004 DB3) of genuine pairs pass
  the gates. The matcher therefore operates at a single point with FMR ≈ 0 and
  FNMR = the share of genuine pairs it gates out. **ZeroFMR ≈ FMR1000 ≈ FMR100 is the
  meaningful number** (57.98% FNMR at FMR = 0, averaged over the 12 sets).
* **The classic EER is mostly a tie artefact.** Nearly all scores are tied at 0, so
  the ROC is a straight segment from (FMR 1, FNMR 0) to (FMR 0, FNMR p). The
  interpolated crossing is p/(1+p). The FVC `[EERlow, EERhigh]` interval stored in the
  JSON is about `[0, p]`, e.g. `[0.0, 22.5]` % on FVC2000 DB1. Compare methods on
  FMR1000 / ZeroFMR, not on this EER.
* **Synthetic SFinGe sets (DB4) fail almost completely.** On FVC2000 DB4 and FVC2002 DB4,
  ≤ 0.4% of genuine pairs pass ("insufficient mutual matches"). The RootSIFT ratio test
  cannot separate locally repetitive synthetic ridges. Decoding was checked: Pillow
  and OpenCV give identical arrays.
* **Failure to enrol** happens on 11 of 960 images (1.15%), mostly on FVC2000 DB3 (6)
  and FVC2004 DB1 (3). Of the 11, 8 fail the orientation-coherence gate and 3 the
  minimum ridge-region gate. FTE images are scored `-inf`.
* **Determinism.** Two full runs produced bit-identical scores (0 differences over 37,920 scores).
* **Latency.** 20–85 ms per template and 6–13 ms per comparison. The comparison runs a
  Python-side mutual k-NN plus RANSAC over up to 1,600 × 128 descriptors. Latency
  differs by up to ~1.5× between two runs on this shared machine, so treat it as an
  order of magnitude.

## Open-source comparator (`sourceafis`, SourceAFIS 3.14.0 .NET) — `results/sourceafis_*.json`

| Set | FTE % | EER % (FVC imp.) | EER % (all imp.) | FMR100 % | FMR1000 % | ZeroFMR % | Rank-1 % | Enrol ms/img | Compare us/pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FVC2000_DB1_B | 0.00 | 2.86 | 3.93 | 4.64 | 8.93 | 9.64 | 92.86 | 23.9 | 1593.7 |
| FVC2000_DB2_B | 0.00 | 11.11 | 11.43 | 14.29 | 23.21 | 24.29 | 87.14 | 28.2 | 3105.6 |
| FVC2000_DB3_B | 0.00 | 18.57 | 17.19 | 25.00 | 33.21 | 35.71 | 82.86 | 50.4 | 18374.2 |
| FVC2000_DB4_B | 0.00 | 6.67 | 6.79 | 10.71 | 16.43 | 19.29 | 98.57 | 16.4 | 932.8 |
| FVC2002_DB1_B | 0.00 | 2.86 | 3.12 | 3.21 | 5.71 | 6.07 | 100.00 | 26.2 | 2536.4 |
| FVC2002_DB2_B | 0.00 | 1.07 | 1.56 | 1.79 | 3.57 | 3.57 | 98.57 | 27.1 | 4005.2 |
| FVC2002_DB3_B | 0.00 | 11.79 | 11.79 | 18.57 | 26.43 | 31.79 | 88.57 | 22.0 | 1087.1 |
| FVC2002_DB4_B | 0.00 | 5.71 | 6.42 | 10.36 | 19.64 | 20.36 | 88.57 | 21.6 | 1025.3 |
| FVC2004_DB1_B | 0.00 | 9.29 | 9.29 | 17.14 | 22.14 | 23.21 | 90.00 | 35.1 | 3207.1 |
| FVC2004_DB2_B | 0.00 | 14.64 | 12.50 | 17.86 | 22.50 | 24.29 | 88.57 | 28.6 | 2920.6 |
| FVC2004_DB3_B | 0.00 | 6.43 | 5.36 | 7.14 | 8.93 | 15.00 | 97.14 | 31.4 | 7800.4 |
| FVC2004_DB4_B | 0.00 | 4.64 | 4.64 | 5.71 | 7.14 | 7.86 | 94.29 | 22.0 | 3661.4 |
| **Average** | 0.00 | 7.97 | 7.83 | 11.37 | 16.49 | 18.42 | 92.26 | 27.7 | 4187.5 |

* Same pairs, metrics and code path as the classic run. `benchmarks/comparators/sourceafis.py`
  plugs a .NET driver into `run_fvc_set`. Images reach SourceAFIS as raw 8-bit
  grayscale at each database's nominal DPI (500, but 569 for FVC2002 DB2 and 512 for FVC2004 DB3).
* SourceAFIS scores are continuous, so its EER is a real crossing. It has no FTE on any set.
* Latency is SourceAFIS's in-process time. "Compare" **includes building a
  `FingerprintMatcher` for the reference template on every pair**, which dominates
  the cost; 1:N search would amortise it. Wall-clock figures including process
  start-up are stored under `latency_wall` in each JSON.
* Sanity anchor: the same library (3.14.0-net) is published at 8.193% EER on
  FVC-onGoing FV-HARD-1.0 (see [SOTA.md](SOTA.md#c-fvc-ongoing-successor-benchmark-different-data-not-fvc20002004)).
  Our 12-set DB_B average of 7.83% (all impostors) is consistent with that.

## SOCOFing training split (for the embedding agent)

The train/val/test folders in the HF mirror `ThanhQuy78/socofing` are **not
subject-disjoint**. Counting real and altered images together, all 600 subjects
appear in every shipped folder (pairwise subject overlap 600 / 600 / 600). On real
prints alone, `val` and `test` share 404 and 395 subjects with `train`. Training on
the shipped folders would leak identities into validation.

`benchmarks/splits/socofing_subject_split_seed0.json` is a subject-level re-split:

```python
from fingerprint_from_image.evaluation import socofing_subject_split
socofing_subject_split("data/socofing/socofing", "split.json", seed=0)  # + per-file list
```

It pools all images, then shuffles subject ids with `random.Random(0)` and cuts them
80/10/10. Result: 480 / 60 / 60 subjects. Real images: 4,800 / 600 / 600. Altered
(CR+Obl+Zcut, all levels): 39,378 / 4,936 / 4,956. That accounts for all
55,270 images. The committed file lists only subject ids. Pass a manifest path to
get the full per-file list, which is large and therefore not committed.

## Environment

* Classic run: 2026-10-22T07:49:54+00:00, package 2.0.0.dev0, git `eeb9175`, Python 3.12.13, numpy 2.5.4, OpenCV 5.0.0
* SourceAFIS run: 2026-10-22T07:43:22+00:00, .NET SDK 10.0.200, SourceAFIS 3.14.0 (NuGet)
* Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 198 Stepping 2, GenuineIntel, 20 logical CPUs; one process, no GPU.
* Full matcher configuration, per-set ROC points (subsampled), CMC curves and raw
  scores (`null` = FTE / failed comparison, ordered as
  `fvc_pairs(10, 8, impostor="all")`) are in each per-set JSON.

## Note on the classic matcher's EER (vision-QA finding, 2026-10-22)

The classic RootSIFT matcher is hard-gated: 97.7–100 % of impostor pairs and a share
*p* of genuine pairs score exactly 0. Its interpolated EER therefore reduces to
p / (1 + p) (e.g. FVC2000 DB1: p = 0.225 → 18.37 %), and it fails completely on
FVC2002 DB4 (100 % of genuine scores are 0) and FVC2000 DB4 (99.6 %). Read its EER as
an interval [0, p] and prefer FMR1000 / ZeroFMR when comparing against it. All 48
reported EERs (24 runs × 2 impostor protocols) were independently recomputed and match.
