"""Successive-halving sweeps driven by the registry (every trial is an ordinary queue job).

Spec (``training/sweeps/<name>.yaml``)::

    name: lr_margin
    base: training/configs/a1_datamix.yaml
    sample: 24                   # optional: seeded random subset of the grid
    space:                       # grid over dotted config keys
      optim.lr: [3.0e-4, 5.0e-4, 8.0e-4]
      loss.margin: [0.3, 0.4, 0.5]
    rungs:                       # train each surviving trial for `epochs`, keep the top `keep`
      - {epochs: 4, keep: 0.5}
      - {epochs: 8, keep: 0.5}
      - {epochs: 12, keep: 1}

``python -m training.sweep step SPEC`` is idempotent: it submits the current rung's
missing jobs, and once all of them are ``done`` (or ``failed``) in the registry it ranks
them on the val selection metric (pooled FNMR@FMR0.1 %, then mean EER) and promotes the
best to the next rung. ``python -m training.sweep run SPEC`` repeats ``step`` until the
last rung finishes. State lives in ``runs/sweeps/<name>/state.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import yaml

from training.keys import runs_root

__all__ = ["expand", "promote", "step"]


def expand(space: dict[str, list]) -> list[dict]:
    """Full grid of override dicts (deterministic order)."""

    keys = sorted(space)
    return [dict(zip(keys, values, strict=True))
            for values in itertools.product(*(space[k] for k in keys))]


def promote(results: dict[int, tuple[float, float] | None], keep: float) -> list[int]:
    """Indices of the best ``ceil(keep * n)`` trials (failed trials rank last, never kept)."""

    ranked = sorted((r, i) for i, r in results.items() if r is not None)
    count = max(1, math.ceil(keep * len(results)))
    return sorted(i for _, i in ranked[:count])


def _trial_name(sweep: str, trial: int, rung: int) -> str:
    return f"{sweep}-t{trial:03d}-r{rung}"


def step(spec_path: Path, *, registry=None, queue=None) -> dict:
    """Advance the sweep by at most one rung; returns the state."""

    from training.config import load_config
    from training.queue import Queue
    from training.registry import Registry

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    registry = registry or Registry()
    queue = queue or Queue()
    state_path = runs_root() / "sweeps" / spec["name"] / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trials = expand(spec["space"])
    if spec.get("sample"):  # seeded random subset of the grid
        import random

        trials = random.Random(spec.get("seed", 0)).sample(trials, min(spec["sample"],
                                                                        len(trials)))
    state = (json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file()
             else {"rung": 0, "alive": list(range(len(trials))), "submitted": [], "history": []})
    if state["rung"] >= len(spec["rungs"]):
        return state
    rung = spec["rungs"][state["rung"]]
    by_name = {row["name"]: row for row in registry.rows("train")}
    results: dict[int, tuple[float, float] | None] = {}
    pending = False
    for trial in state["alive"]:
        name = _trial_name(spec["name"], trial, state["rung"])
        row = by_name.get(name)
        if row is None and name not in state["submitted"]:
            overrides = {**trials[trial], "optim.epochs": rung["epochs"], "name": name}
            cfg = load_config(spec["base"], overrides)
            queue.submit_config(cfg.to_dict(), name=name, priority=spec.get("priority", 0))
            state["submitted"].append(name)
            pending = True
        elif row is None or row["status"] in ("pending", "running"):
            pending = True
        elif row["status"] == "done":
            metrics = row["metrics"] or {}
            results[trial] = (metrics.get("val_select", math.inf),
                              metrics.get("val_select_eer", math.inf))
        else:
            results[trial] = None
    if not pending:
        survivors = promote(results, rung["keep"])
        state["history"].append({"rung": state["rung"], "epochs": rung["epochs"],
                                 "results": {str(k): v for k, v in results.items()},
                                 "promoted": survivors, "trials": [trials[i] for i in survivors],
                                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["rung"] += 1
        state["alive"] = survivors
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("step", "run"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("--poll", type=float, default=120.0)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    while True:
        state = step(args.spec)
        print(json.dumps({k: state[k] for k in ("rung", "alive")}))  # noqa: T201
        if args.command == "step" or state["rung"] >= len(spec["rungs"]):
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
