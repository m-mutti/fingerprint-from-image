"""Per-sensor validation on manifest ``val`` identities (the only model-selection signal).

Each val set (FVC2000/2002 DB1–4 DB_B fingers 101–110, neurotech, l3sf, SOCOFing) is
preprocessed once with the runtime pipeline. Every evaluation reports per-set all-pairs
EER/FMR100 and two selection numbers over :data:`SELECTION_SETS` (contact FVC DB1–3 of
2000/2002 + neurotech):

* ``select`` (primary) — pooled FNMR at FMR ≤ 0.1 % with **one global threshold** over
  the genuine/impostor scores of all selection sets;
* ``select_eer`` (secondary) — mean of the per-set EERs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from training.data import Sample, load_gray, model_input
from training.manifest import SELECTION_SETS, all_pair_scores
from training.metrics import verification_metrics

__all__ = ["Validator", "selection_metrics", "selection_validator", "torch_embed_fn"]


def _prep(args: tuple[str, str, bool, int]) -> np.ndarray:
    path, source, naive, size = args
    return model_input(load_gray(path, source), naive=naive, size=size)


def selection_metrics(scores: dict[str, dict[str, np.ndarray]],
                      selection: Sequence[str]) -> dict[str, float]:
    """Pooled FNMR@FMR0.1 % (global threshold) and mean EER over ``selection`` sets."""

    genuine = np.concatenate([scores[s]["genuine"] for s in selection])
    impostor = np.concatenate([scores[s]["impostor"] for s in selection])
    pooled = verification_metrics(genuine, impostor)
    per_set = [verification_metrics(scores[s]["genuine"], scores[s]["impostor"]).eer
               for s in selection]
    return {"select": pooled.fmr1000, "select_eer": float(np.mean(per_set)),
            "pooled_eer": pooled.eer, "pooled_fmr100": pooled.fmr100}


class Validator:
    """Holds preprocessed val inputs; ``__call__(embed)`` returns metrics and raw scores."""

    def __init__(self, groups: dict[str, list[Sample]], *, naive: bool = False,
                 size: int = 256, workers: int = 8,
                 selection: Sequence[str] = SELECTION_SETS) -> None:
        self.selection = [s for s in selection if s in groups]
        self.sets: dict[str, tuple[np.ndarray, list[str]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, samples in groups.items():
                jobs = [(s.path, s.source, naive, size) for s in samples]
                inputs = np.stack(list(pool.map(_prep, jobs, chunksize=16)))[:, None]
                self.sets[name] = (inputs.astype(np.float32), [s.label for s in samples])

    def embeddings(self, embed: Callable[[np.ndarray], np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name, (inputs, _) in self.sets.items():
            emb = embed(inputs)
            if not np.isfinite(emb).all():
                raise FloatingPointError(f"non-finite embeddings on val set {name}")
            out[name] = emb
        return out

    def __call__(
        self, embed: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
        metrics: dict[str, dict[str, float]] = {}
        scores: dict[str, dict[str, np.ndarray]] = {}
        for name, emb in self.embeddings(embed).items():
            genuine, impostor = all_pair_scores(emb, self.sets[name][1])
            result = verification_metrics(genuine, impostor)
            metrics[name] = {"eer": result.eer, "fmr100": result.fmr100,
                             "fmr1000": result.fmr1000}
            scores[name] = {"genuine": genuine, "impostor": impostor}
        summary = selection_metrics(scores, self.selection)
        return {**summary, "sets": metrics}, {s: scores[s] for s in self.selection}


def selection_validator(size: int = 256) -> Validator:
    """Validator over the selection sets only (export variant choice, parity)."""

    from training.config import ROOT
    from training.manifest import load_rows, val_groups

    rows = load_rows(ROOT / "data" / "manifests", ["FVC2000", "FVC2002", "neurotech"])
    return Validator(val_groups(rows, SELECTION_SETS), size=size)


def torch_embed_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """numpy → numpy embedding function for a CPU PyTorch model (batches of 32)."""

    import torch

    def embed(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[i : i + 32])).numpy()
                                   for i in range(0, len(x), 32)])

    return embed
