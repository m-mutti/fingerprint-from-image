"""Keyed, guarded test evaluation: FVC2004 DB1–4 B and FVC2004 full, once per checkpoint.

Key = checkpoint SHA-256 + protocol version + test-set list → ``runs/eval/<key[:12]>/``.
Guard: the checkpoint's ``train`` record must report zero held-out overlap and the same
held-out digest as the current manifests; otherwise evaluation is refused. Test
metrics are written to the registry for reporting only — nothing reads them back for
selection.

Usage::

    python -m training.evaluate runs/train/<key>/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from training.guards import ContaminationError, heldout_from_rows
from training.keys import StepDir, file_sha256, runs_root
from training.manifest import load_rows
from training.registry import Registry
from training.sources import HELDOUT_MANIFESTS

__all__ = ["PROTOCOL_VERSION", "evaluate_checkpoint", "main"]

PROTOCOL_VERSION = "protocol-v1-2026-10-22"
TEST_SETS = ("FVC2004_DB1_B", "FVC2004_DB2_B", "FVC2004_DB3_B", "FVC2004_DB4_B", "FVC2004_full")


def _train_record(checkpoint: Path) -> dict:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = state.get("train_key")
    if not key:
        raise ContaminationError(f"{checkpoint} has no train_key; cannot verify its training data")
    record = StepDir("train", key, runs_root()).record()
    if record is None:
        raise ContaminationError(f"no train record for key {key[:12]}")
    return record


def check_checkpoint(checkpoint: Path, manifest_dir: Path) -> str:
    """Held-out guard at evaluation time; returns the train key."""

    record = _train_record(checkpoint)
    guard = record.get("data", {}).get("heldout_guard", {})
    current = heldout_from_rows(load_rows(manifest_dir, HELDOUT_MANIFESTS)).digest
    if guard.get("heldout_overlap", 1) != 0 or guard.get("heldout_digest") != current:
        raise ContaminationError("checkpoint's training data was not verified against the "
                                 "current held-out set")
    return record["key"]


def protocol_v1(matrices: dict[str, np.ndarray], reference: dict[str, np.ndarray] | None,
                n_boot: int = 2000) -> dict:
    """Protocol v1 summary: finger-level bootstrap CIs (FVC impostors) for full / DB_A / DB_B."""

    from training.stats import fvc_bootstrap

    def subset(mats: dict[str, np.ndarray], lo: int, hi: int) -> dict[str, np.ndarray]:
        return {db: m[(lo - 1) * 8 : hi * 8, (lo - 1) * 8 : hi * 8] for db, m in mats.items()}

    out = {}
    for name, (lo, hi) in {"full": (1, 110), "A": (1, 100), "B": (101, 110)}.items():
        ref = subset(reference, lo, hi) if reference is not None else None
        out[name] = fvc_bootstrap(subset(matrices, lo, hi), n=n_boot, reference=ref)
    return out


def evaluate_checkpoint(checkpoint: Path, *, manifest_dir: Path | None = None,
                        device: str = "cuda", force: bool = False,
                        reference_scores: Path | None = None) -> dict:
    """Run (or reuse) the protocol-v1 test suite for one frozen release candidate."""

    from training.config import ROOT
    from training.eval_full import evaluate_full
    from training.eval_fvc import FVC_GROUPS, _torch_embed, evaluate_sets

    manifest_dir = manifest_dir or ROOT / "data" / "manifests"
    train_key = check_checkpoint(checkpoint, manifest_dir)
    inputs = {"checkpoint_sha256": file_sha256(checkpoint), "protocol": PROTOCOL_VERSION,
              "test_sets": list(TEST_SETS),
              "reference": file_sha256(reference_scores) if reference_scores else None}
    step = StepDir.for_inputs("eval", inputs)
    if step.is_done() and not force:
        return json.loads((step.path / "results.json").read_text(encoding="utf-8"))
    started = time.time()
    embed, naive, size = _torch_embed(checkpoint, device)
    matrices: dict[str, np.ndarray] = {}
    results = {"fvc2004_db_b": evaluate_sets(embed, FVC_GROUPS["FVC2004"], naive=naive, size=size),
               "fvc2004_full": evaluate_full(embed, naive=naive, size=size,
                                             keep_scores=matrices),
               "train_key": train_key}
    reference = None
    if reference_scores is not None:
        with np.load(reference_scores) as stored:
            reference = {k: stored[k] for k in stored.files}
    results["protocol_v1"] = protocol_v1(matrices, reference)
    step.path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(step.path / "scores.npz", **matrices)
    (step.path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    step.mark_done(inputs, ["results.json", "scores.npz"], {"train_key": train_key}, started)
    db_b, full = results["fvc2004_db_b"]["mean"], results["fvc2004_full"]["mean"]
    v1 = results["protocol_v1"]
    Registry().upsert("eval", step.key, "done", parent_key=train_key, metrics={
        "full_eer_fvc": v1["full"]["mean_eer"], "full_eer_fvc_ci": [v1["full"]["ci_low"],
                                                                    v1["full"]["ci_high"]],
        "db_a_eer_fvc": v1["A"]["mean_eer"], "db_b_eer_fvc": v1["B"]["mean_eer"],
        "db_b_eer_all": db_b["eer_all"], "db_b_fmr1000": db_b["fmr1000"],
        "db_b_eer_fvc_legacy": db_b["eer_fvc"], "db_b_fmr100": db_b["fmr100"],
        "full_eer_all": full["full"]["eer_all"], "full_fmr1000": full["full"]["fmr1000"],
        "db_a_eer_all": full["A"]["eer_all"],
    }, artifacts={"results": str(step.path / "results.json"),
                  "scores": str(step.path / "scores.npz"), "checkpoint": str(checkpoint)})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reference-scores", type=Path,
                        help="npz of FVC2004_DB{n} (880x880) reference score matrices "
                             "(e.g. SourceAFIS) for the paired bootstrap difference")
    parser.add_argument("--release-candidate", action="store_true",
                        help="required: confirms this checkpoint is FROZEN (protocol v1 test "
                             "hygiene - FVC2004 never informs a decision)")
    args = parser.parse_args(argv)
    if not args.release_candidate:
        parser.error("FVC2004 is evaluated only for frozen release candidates; "
                     "pass --release-candidate to confirm")
    results = evaluate_checkpoint(args.checkpoint, device=args.device, force=args.force,
                                  reference_scores=args.reference_scores)
    print(json.dumps({"protocol_v1": results["protocol_v1"],  # noqa: T201
                      "db_b_all_impostors": results["fvc2004_db_b"]["mean"],
                      "full_all_impostors": results["fvc2004_full"]["mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
