# Evaluation-methodology and leakage audit (2026-10-22)

Read-only audit with independent scripts (manifests, near-duplicate search, SourceAFIS
cross-checks, finger-level bootstrap). No repository files were changed by the auditor.

## Leakage verdict: PASS for the FVC2004 test set, with two conditions
- Manifests (76,298 rows): 0 split-check problems; 0 identity overlap between splits;
  157 duplicate groups, none touching FVC2004, none live across splits; SOCOFing matches
  the shared seed-0 split.
- 3,520 FVC2004 images vs 111,592 training-side images (all manifest sources, synthetic_v1,
  impose_syn, synth_procedural, naijafingset, unsw_2d3d): **0** decoded-pixel hash matches
  (positive control 320/320); best ridge-normalised NCC 0.726, below the 0.75 seen between
  two *different* FVC2004 fingers.
- Neurotechnology (same CrossMatch V300 sensor as FVC2004 DB1): 204,160 SourceAFIS comparisons,
  max score 43.5 (≈ chance at FMR 1e-4) vs median 148 for neurotech same-finger pairs.
- **C1 (P0):** the legacy config path trains on FVC2000 DB_B, which are manifest *val* rows; the
  overlap check compares labels in different formats and always reports 0 → default to no legacy
  sets and compare paths + SHA-256.
- **C2 (P0):** the held-out guard checks SHA-256 only for manifest samples; pixel-identical copies of
  FVC2004 DB_B exist under `fvc_full/` → content-hash every training file against held-out content.

## Other P0
- FVC2004 informs decisions in `export.py`/`export_litert.py` (quantisation acceptance), `summarize.py`
  and `evaluate.py` → use val only; test frozen release candidates only.
- With manifests, FVC2000/2002 DB_B are validation sets of sensors seen in training: the old
  "12-set average" is no longer a held-out number. Report FVC2004 as the test.
- Run artefacts (`runs/`) are git-ignored and unprotected.

## Statistics (finger-level bootstrap, B = 2000)
| | Mean EER | 95 % CI |
|---|---:|---:|
| SourceAFIS, FVC2004 DB_B (4 × 10 fingers) | 7.95 % | 5.17–10.83 |
| Classic, FVC2004 DB_B | 35.37 % | 32.17–37.84 |
| SourceAFIS, **FVC2004 full** (4 × 110 fingers, FVC impostors) | **10.29 %** | **9.17–11.44** |

Minimum EER difference claimable at 95 %: ≈ 4.0–4.7 points on FVC2004 DB_B (unpaired) vs
≈ 1.6 points on FVC2004 full (≈ 1.3 paired). DB_B also under-states difficulty (DB3: 6.43 % on
DB_B vs 13.86 % on DB_A).

## Recommendations adopted
- Primary test: mean EER over FVC2004 DB1–4 **full**, FVC impostor protocol, with finger-level
  bootstrap 95 % CI and paired bootstrap differences vs SourceAFIS; also FMR1000 (all impostors),
  DB_A-only EER (published comparability), DB_B (continuity), raw scores stored.
- Selection noise: SourceAFIS 8-set val mean has bootstrap SD 1.25 points → do not trust
  sub-3-point selection differences; prefer `last.pt` unless `best` beats it by > 2 SD.
- Ridge-period estimator: ranking unchanged without FVC2004 → re-freeze on val-only evidence,
  exclude FVC2004 from `period_study` by default.
- Pipeline MVP: robust keys (typed config, relative paths, image hash + label lists, val/heldout
  digests, untracked-code hash), single OS-level GPU lock held by the trainer, atomic writes with
  WinError 5/32 retry, heartbeat in every phase, content-hash guards, defer sweeps/LiteRT.
