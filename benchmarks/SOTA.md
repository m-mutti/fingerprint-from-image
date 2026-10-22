# State of the art on FVC2000 / 2002 / 2004 — and how it relates to our numbers

Every number below was read from the primary source linked next to it, on
2026-10-22 (official BioLab result pages, or the paper PDF on arXiv). Nothing is
filled in from memory. Items we could not verify are listed in the last section.

## Read this first: why published FVC numbers are not directly comparable to ours

| | Published FVC results | Our benchmark (`fingerprint-bench`) |
|---|---|---|
| Subset | **DB_A**: fingers 1–100 × 8 impressions | **DB_B**: fingers 101–110 × 8 impressions |
| Genuine / impostor pairs (FVC protocol) | 2,800 / 4,950 | 280 / 45 (we also report all 2,880 cross-finger impostors) |
| Smallest non-zero FMR (FVC protocol) | 1 / 4,950 ≈ 0.02 % | 1 / 45 ≈ 2.2 % (1 / 2,880 ≈ 0.035 % with all impostors) |
| Role of the subset | Sequestered test set | Released to participants **for tuning** |

Source for the subset definitions, identical on all three databases pages
(<http://bias.csr.unibo.it/fvc2000/databases.asp>,
<http://bias.csr.unibo.it/fvc2002/databases.asp>,
<http://bias.csr.unibo.it/fvc2004/databases.asp>): *"fingers from 101 to 110
(set B) have been made available to the participants to allow parameter tuning
before the submission of the algorithms; the benchmark is then constituted by
fingers numbered from 1 to 100 (set A)."* Only DB_B can be downloaded freely.
The full databases ship with the *Handbook of Fingerprint Recognition*.

What follows from this:

* DB_B has 10 fingers. An EER measured on it has a wide confidence interval. With 45
  impostor scores, FMR100 and FMR1000 under the FVC protocol collapse to ZeroFMR.
  We therefore report FMR100 and FMR1000 on all 2,880 cross-finger impostor pairs.
  That is a superset of the official protocol, so those values are not FVC-official numbers.
* Competition entries were tuned on DB_B. If they were run on DB_B, their numbers
  would be optimistic, not pessimistic.
* BioLab itself warns against cross-year comparison (FVC2004 page): *"one should
  neither compare error rates among different FVC competitions"*. FVC2004 was made
  deliberately harder (distortion, dry/wet fingers).
* Use the table below as an *order-of-magnitude target*. The only like-for-like
  comparison here is the locally run comparator (last section). It uses our exact
  DB_B protocol and code path.

Protocol (FVC2002/2004 `perfeval.asp`, e.g.
<http://bias.csr.unibo.it/fvc2002/perfeval.asp>): *"Each sample in the subset A
is matched against the remaining samples of the same finger to compute the False
Non Match Rate … If the matching g against h is performed, the symmetric one
(i.e., h against g) is not executed"*. Impostors are the first impression of
each finger against the first impression of every other finger.
`fingerprint_from_image.evaluation.fvc_pairs` implements exactly this.

## A. Official FVC competitions: best participant per database (DB_A, FVC protocol)

All rates are in %. "n/r" = not reported on the official page (FVC2000 tables give
only EER, EER\*, REJENROLL and REJMATCH). All winners listed have REJENROLL = 0.00 %.

| Database | Best participant (by EER) | EER | FMR100 | FMR1000 | ZeroFMR | Source |
|---|---|---:|---:|---:|---:|---|
| FVC2000 DB1_A | Sag1 (SAGEM SA) | 0.67 | n/r | n/r | 2.11 | [res_db1_a](http://bias.csr.unibo.it/fvc2000/results/res_db1_a.asp), [Sag1 detail](http://bias.csr.unibo.it/fvc2000/participants/results/SAG1_db1_a.asp) |
| FVC2000 DB2_A | Sag1 | 0.61 | n/r | n/r | 1.36 | [res_db2_a](http://bias.csr.unibo.it/fvc2000/results/res_db2_a.asp) |
| FVC2000 DB3_A | Sag1 | 3.64 | n/r | n/r | 6.82 | [res_db3_a](http://bias.csr.unibo.it/fvc2000/results/res_db3_a.asp) |
| FVC2000 DB4_A | Sag1 | 1.99 | n/r | n/r | 6.71 | [res_db4_a](http://bias.csr.unibo.it/fvc2000/results/res_db4_a.asp) |
| FVC2000 average | Sag1 | 1.73 | n/r | n/r | n/r | [results](http://bias.csr.unibo.it/fvc2000/results.asp) |
| FVC2002 DB1_A | PA15 (Bioscrypt) | 0.10 | 0.11 | 0.21 | 0.21 | [res_db1_a](http://bias.csr.unibo.it/fvc2002/results/res_db1_a.asp) |
| FVC2002 DB2_A | PA27 (anonymous, industrial) | 0.14 | 0.11 | 0.14 | 0.29 | [res_db2_a](http://bias.csr.unibo.it/fvc2002/results/res_db2_a.asp) |
| FVC2002 DB3_A | PA15 | 0.37 | 0.32 | 0.61 | 0.75 | [res_db3_a](http://bias.csr.unibo.it/fvc2002/results/res_db3_a.asp) |
| FVC2002 DB4_A | PA15 | 0.10 | 0.04 | 0.11 | 0.25 | [res_db4_a](http://bias.csr.unibo.it/fvc2002/results/res_db4_a.asp) |
| FVC2002 average | PA15 | 0.19 | 0.15 | 0.28 | 0.38 | [resultsAvg](http://bias.csr.unibo.it/fvc2002/results/resultsAvg.asp) |
| FVC2004 DB1_A (Open) | P047 (Sonda Ltd.) | 1.97 | 2.86 | 4.36 | 8.21 | [O_res_db1_a](http://bias.csr.unibo.it/fvc2004/results/O_res_db1_a.asp) |
| FVC2004 DB2_A (Open) | P039 (Jan Lunter) | 1.58 | 2.18 | 5.79 | 8.18 | [O_res_db2_a](http://bias.csr.unibo.it/fvc2004/results/O_res_db2_a.asp) |
| FVC2004 DB3_A (Open) | P047 | 1.18 | 1.68 | 2.68 | 4.89 | [O_res_db3_a](http://bias.csr.unibo.it/fvc2004/results/O_res_db3_a.asp) |
| FVC2004 DB4_A (Open) | P071 (Inst. of Automation, CAS) | 0.61 | 0.50 | 1.00 | 1.93 | [O_res_db4_a](http://bias.csr.unibo.it/fvc2004/results/O_res_db4_a.asp) |
| FVC2004 average (Open) | P101 (Bioscrypt) | 2.07 | 2.54 | 4.70 | 6.21 | [Open_resultsAvg](http://bias.csr.unibo.it/fvc2004/results/Open_resultsAvg.asp) |
| FVC2004 average (Light) | P009 (Suprema) | 3.51 | 5.21 | 8.71 | 12.38 | [Light_resultsAvg](http://bias.csr.unibo.it/fvc2004/results/Light_resultsAvg.asp) |

Excluded: post-competition entries (e.g. FVC2000 "Neur", FVC2002 "LA22"). BioLab
lists them separately because the full databases were known in advance.

## B. Learned / modern matchers reporting FVC numbers (all on DB_A)

Most deep fixed-length papers report **TAR at FAR = 0.1 %** (= 1 − FNMR at
FMR = 0.1 %, i.e. 1 − FMR1000), not EER. None of the TAR rows below trained on FVC.

| Method (source) | FVC sets | Protocol | Metric | Values (DB1 / DB2 / DB3) | Location in source |
|---|---|---|---|---|---|
| **DeepPrint**: Engelsma, Cao, Jain, "Learning a Fixed-Length Fingerprint Representation", IEEE TPAMI ([arXiv:1909.09901v2](https://arxiv.org/abs/1909.09901)) | FVC2004 DB1_A | FVC (4,950 impostors) | TAR@FAR=0.1 % | 97.5 (uncompressed 97.53) | Table 6 "Authentication Accuracy (FVC 2004 DB1 A)"; Table 4. *"TAR @ FAR of 0.1% is reported since there are only 4,950 imposter pairs in the FVC protocol"* |
| ↳ COTS A / COTS B (as reported by DeepPrint) | FVC2004 DB1_A | FVC | TAR@FAR=0.1 % | 96.75 / 96.57 | same Table 6 |
| **AFR-Net** (with re-alignment): Grosz & Jain, "AFR-Net: Attention-Driven Fingerprint Recognition Network" ([arXiv:2211.13897v2](https://arxiv.org/abs/2211.13897)) | FVC2002 DB1–3_A | *"Following the FVC protocol of 2,800 genuine pairs and 4,950 imposter pairs"* | TAR@FAR=0.1 % | 100 / 99.96 / 99.36 | Table 3, p. 9 |
| ↳ same | FVC2004 DB1–3_A | same | TAR@FAR=0.1 % | 100 / 99.64 / 100 | Table 3 |
| ↳ AFR-Net without re-alignment | FVC2002 / FVC2004 DB1–3_A | same | TAR@FAR=0.1 % | 99.86 / 99.96 / 98.43; 100 / 99.36 / 100 | Table 3 |
| ↳ VeriFinger v12.3 (as reported by AFR-Net) | FVC2002 / FVC2004 DB1–3_A | same | TAR@FAR=0.1 % | 99.96 / 99.86 / 99.54; 98.79 / 98.46 / 99.86 | Table 3 |
| **LDRF**: Duan, Pan, Feng, Zhou, "Fingerprint Matching with Localized Deep Representation" ([arXiv:2311.18576v2](https://arxiv.org/abs/2311.18576v2)) | FVC2000 DB1–3_A | FVC: *"100×C(8,2)=2,800, and impostor pairs C(100,2)=4,950"* | TAR@FAR=0.1 % | 98.79 / 99.46 / 97.29 | Table III, p. 11, row "Ours" |
| ↳ same | FVC2002 DB1–3_A | same | TAR@FAR=0.1 % | 99.18 / 99.68 / 94.93 | Table III |
| ↳ same | FVC2004 DB1–3_A | same | TAR@FAR=0.1 % | 98.64 / 98.29 / 97.50 | Table III |
| ↳ VeriFinger SDK 12.0 (as reported by LDRF) | FVC2000 / 2002 / 2004 DB1–3_A | same | TAR@FAR=0.1 % | 99.89 / 99.96 / 99.50; 99.89 / 99.93 / 99.43; 98.93 / 97.79 / 99.86 | Table III |
| **FLARE**: Pan, Guan, Duan, Feng, Zhou, "Fixed-Length Dense Fingerprint Representation" ([arXiv:2505.03597v1](https://arxiv.org/abs/2505.03597)) | FVC2002 DB3_A; FVC2004 DB1_A | 2,800 / 4,950 | TAR@FAR=0.1 % | 96.36 (FVC2002 DB3_A); 99.57 (FVC2004 DB1_A) | Table IV, p. 8 |
| **IFViT**: Qiu et al., "IFViT: Interpretable Fixed-Length Representation for Fingerprint Matching via Vision Transformer" ([arXiv:2404.08237v1](https://arxiv.org/abs/2404.08237)) | FVC2004 DB1–3_A | **not FVC**: 7,750 test pairs per DB; **trained on FVC2002 DB1–3_A plus NIST SD301a, SD302a and MOLF, after pretraining on 6.7M PrintsGAN images** (Sec. IV-A, Table I) | EER (%) | 1.87 / 1.83 / 1.49 (their DeepPrint re-impl. 2.70 / 3.39 / 2.05; AFRNet re-impl. 2.43 / 2.95 / 1.56) | Table IV, p. 10 |

Caveats. LDRF and FLARE also list DeepPrint and AFRNet *re-implementations*
trained on their own data. Those rows differ from the original papers and are
deliberately omitted. LDRF v2 was read. Later arXiv versions of 2311.18576 carry a
different title and their numbers may differ. AFR-Net, FLARE and IFViT were read
as arXiv preprints, not camera-ready versions.

## C. FVC-onGoing (successor benchmark; different data, not FVC2000–2004)

From <https://biolab.csr.unibo.it/fvcongoing/UI/Form/PublishedAlgs.aspx>
(Fingerprint Verification area). Columns are EER / FMR1000 / FMR10000.
FV-STD-1.0 has 27,720 genuine / 87,990 impostor attempts. FV-HARD-1.0 has
19,320 / 20,850 (from `BenchmarkAreas/BenchmarkAreaFV.aspx`).

| Benchmark | Best published | EER % | Open-source reference | EER % |
|---|---|---:|---|---:|
| FV-STD-1.0 | Dermalog DRM-SP22-FL2 (17/01/2025) | 0.000 | SourceAFIS 3.14.0 | 3.867 |
| FV-HARD-1.0 | Dermalog DRM-SP22-FL3 (11/06/2025) | 0.002 | SourceAFIS 3.14.0-net | 8.193 |

The SourceAFIS 3.14.0-net row is the same library and version we run locally
(section D). It anchors our local comparator to an independently published result.

## D. Like-for-like comparison on our DB_B protocol (run locally)

Both rows use the identical DB_B protocol, pairs and metric code. Details and per-set tables
are in [RESULTS.md](RESULTS.md). Rates are in %, averaged over the 12 sets. FMR100,
FMR1000 and ZeroFMR use all 2,880 cross-finger impostors per set.

| 12-set average | FTE % | EER % (FVC imp.) | EER % (all imp.) | FMR100 % | FMR1000 % | ZeroFMR % | Rank-1 % | Enrol ms/img | Compare us/pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classic (v2 RootSIFT + RANSAC) | 1.15 | 33.12 | 33.36 | 55.89 | 57.65 | 57.98 | 46.67 | 36.1 | 9226 |
| SourceAFIS 3.14.0 (.NET) | 0.00 | 7.97 | 7.83 | 11.37 | 16.49 | 18.42 | 92.26 | 27.7 | 4187 |

The DB_B average EER (all impostors) is 7.83% for SourceAFIS and
33.36% for the classic matcher, against roughly 0.1–3.6% for the
competition winners on DB_A (section A). The classic v2 pipeline's EER is about 4×
that of a free minutiae matcher, and one to two orders of magnitude above the
2002–2004 competition winners (the DB_A/DB_B caveats above apply). The classic EER
is inflated by score ties at 0 (see RESULTS.md). Its ZeroFMR of 57.98% vs 18.42% is
the fairer comparison.

**Other comparators considered:**

* **NIST NBIS (mindtct + bozorth3)**: not run. NBIS is C source only, and this
  machine has no C toolchain (no gcc/make/cmake, no WSL/Docker available to this
  agent). To add it, build NBIS 5.0 under WSL or MSYS2 and wrap `mindtct` →
  `.xyt` → `bozorth3` with the same `Scorer` interface as `sourceafis.py`.
* **Deep fixed-length extractors (DeepPrint, AFR-Net, LDRF, FLARE)**: not run. We did
  not search exhaustively for released weights, and we did not locate or verify any
  official ones. Running one would also need PyTorch on CPU, outside the runtime package.

## E. Not verified / not included, and why

* **The three competition papers** (Maio et al., TPAMI 2002 / ICPR 2002 / ICBA 2004)
  were not read. All competition numbers come from the official result pages above.
* **FVC2000 FMR100/FMR1000** are not published on the official pages.
* **FVC2004 Light-category per-DB winners** were not collected; only the average is listed.
* **JIPNet** (arXiv:2405.03959) is excluded. It evaluates cropped partial patches
  with its own 5,600/5,600 pairs and trains on FVC2004 DB1_A/DB2_A.
* **Bozorth3 / NBIS, MinNet, dense-registration papers**: no FVC number was
  verified from a fetched primary source, so none is quoted.
