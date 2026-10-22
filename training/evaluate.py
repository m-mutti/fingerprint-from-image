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

from training.guards import ContaminationError, heldout_from_rows
from training.keys import StepDir, file_sha256, runs_root
from training.manifest import load_rows
from training.registry import Registry
from training.sources import HELDOUT_MANIFESTS

__all__ = ["PROTOCOL_VERSION", "evaluate_checkpoint", "main"]

PROTOCOL_VERSION = "fvc2004-dbb+full-v1"
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


def evaluate_checkpoint(checkpoint: Path, *, manifest_dir: Path | None = None,
                        device: str = "cuda", force: bool = False) -> dict:
    """Run (or reuse) the standard test suite for one checkpoint."""

    from training.config import ROOT
    from training.eval_full import evaluate_full
    from training.eval_fvc import FVC_GROUPS, _torch_embed, evaluate_sets

    manifest_dir = manifest_dir or ROOT / "data" / "manifests"
    train_key = check_checkpoint(checkpoint, manifest_dir)
    inputs = {"checkpoint_sha256": file_sha256(checkpoint), "protocol": PROTOCOL_VERSION,
              "test_sets": list(TEST_SETS)}
    step = StepDir.for_inputs("eval", inputs)
    if step.is_done() and not force:
        return json.loads((step.path / "results.json").read_text(encoding="utf-8"))
    started = time.time()
    embed, naive, size = _torch_embed(checkpoint, device)
    results = {"fvc2004_db_b": evaluate_sets(embed, FVC_GROUPS["FVC2004"], naive=naive, size=size),
               "fvc2004_full": evaluate_full(embed, naive=naive, size=size),
               "train_key": train_key}
    step.path.mkdir(parents=True, exist_ok=True)
    (step.path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    step.mark_done(inputs, ["results.json"], {"train_key": train_key}, started)
    db_b, full = results["fvc2004_db_b"]["mean"], results["fvc2004_full"]["mean"]
    Registry().upsert("eval", step.key, "done", parent_key=train_key, metrics={
        "db_b_eer_all": db_b["eer_all"], "db_b_eer_fvc": db_b["eer_fvc"],
        "db_b_fmr100": db_b["fmr100"], "db_b_fmr1000": db_b["fmr1000"],
        "full_eer_all": full["full"]["eer_all"], "full_eer_fvc": full["full"]["eer_fvc"],
        "db_a_eer_all": full["A"]["eer_all"], "full_fmr1000": full["full"]["fmr1000"],
    }, artifacts={"results": str(step.path / "results.json"), "checkpoint": str(checkpoint)})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    results = evaluate_checkpoint(args.checkpoint, device=args.device, force=args.force)
    print(json.dumps({"db_b": results["fvc2004_db_b"]["mean"],  # noqa: T201
                      "full": results["fvc2004_full"]["mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
