# Code review — embedder + training code (2026-10-22)

Read-only review of `src/fingerprint_from_image/embedding/` and `training/` (training-agent
worktree at 95688ad plus uncommitted pipeline files), with CPU verification scripts.

**Verdict:** block the long run until the 4 HIGH items are fixed. The core ML path is sound.

## Verified clean
- Torch vs numpy normaliser: max |Δ| 1.2e-6 (30 images across FVC2000/2002, SOCOFing, l3sf, TDFD).
- Runtime path vs cache: typical max 0.006–0.028 z-units (one TDFD outlier, see L1).
- EER/FMR100/FMR1000/ZeroFMR vs shared evaluator: max |Δ| 7e-15; identical FVC pair sets.
- Splits across 8 manifests: 0 identities in two splits; held-out guard covers identity, path and sha256.
- Weight decay excludes all backbone norm/bias tensors; ArcFace runs in fp32 with a finite acos edge.
- `state.pt` holds net, EMA, classifier, optimizer, counters, guards, sampler, RNGs.

## HIGH
- **H1** `training/queue.py:115-125` — stale heartbeat mtime kills resumed jobs as "stall" → crash-resume never works.
  Fix: unlink heartbeat before `Popen` or `last = max(start, mtime)`.
- **H2** `training/keys.py:88` — content key ignores untracked code and over-reacts to doc/YAML edits.
  Fix: commit before launch; hash contents of `git ls-files -co --exclude-standard` for `training/*.py` + `src/**/*.py`.
- **H3** `trainer.py:200-207`, `model.py:23`, `training/metrics.py:64` — a non-finite forward writes NaN into head
  BatchNorm running stats, EMA copies it, validation reports EER 0.0 % (one NaN) and best.pt is saved.
  Fix: check `torch.isfinite(loss)` before backward; snapshot/restore head BN buffers on skipped steps; metrics raise
  on NaN; assert finite embeddings in the validator.
- **H4** `training/batch_probe.py:40-42, 79-82` — probe runs an over-budget batch for 4 steps (WDDM spill) and leaves the
  reserved memory allocated for the whole run; compiles dynamic vs static. Fix: probe in a subprocess (or del + gc +
  dynamo reset + empty_cache), stop before the over-budget candidate, compile like training,
  `set_per_process_memory_fraction(0.9)`, `empty_cache()` after validation.

## MEDIUM
- **M1** `--force` resumes the old `state.pt` instead of retraining → move state/best/last aside.
- **M2** `export.py:186-191`, `export_litert.py` — quantisation variant chosen on FVC2004 (test set) → choose on val.
- **M3** ONNX metadata hard-codes `target_period` and lacks `preprocess_version`; cache key relies on a hand-bumped
  version → write the version into ONNX metadata, embedder refuses mismatches, hash preprocessing source into the key.
- **M4** EMA without warm-up biases short ablations by batch size → `use_warmup=True`.
- **M5** gradient clipping misses the ArcFace weights (10× LR); head norm/bias get weight decay → fix param groups.

## LOW
L1 uint8 cache rounding (rare large outlier; float16 optional) · L2 cited parity test file missing · L3 `cfg.input_size`
not propagated to validation · L4 calibration docstring says train but uses FVC2000 DB_B val (no FVC2004) ·
L5 registry/licence strings hard-coded (ConvNeXt-V2 weights CC-BY-NC-4.0) · L6 non-atomic best/last checkpoints ·
L7 duplicate metrics rows after resume · L8 ±5 % anisotropy not configurable.

## Next step
Fix H1–H4 (commit first), then a 2-epoch smoke run killed mid-run to prove resume before the long run.
