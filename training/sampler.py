"""P×K identity-balanced, source-balanced batch sampler.

Every batch holds ``P`` identities × ``K`` images, so each batch contains genuine
positives for every identity. Identities are drawn per source group (e.g. SOCOFing,
FVC2000, FVC2002, synthetic) according to the configured shares, uniformly within a
group; the ``K`` images of an identity are drawn without replacement when it has at
least ``K`` images and with replacement otherwise (augmentation makes repeats differ).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["PKSampler"]


class PKSampler:
    """Draws ``(P*K,)`` index batches; deterministic for a given seed and call sequence."""

    def __init__(self, labels: Sequence[int], groups: Sequence[str], shares: Mapping[str, float],
                 p: int, k: int, seed: int = 0) -> None:
        if p < 1 or k < 1:
            raise ValueError("P and K must be positive")
        self.p, self.k = p, k
        self.rng = np.random.default_rng(seed)
        by_identity: dict[int, list[int]] = defaultdict(list)
        identity_group: dict[int, str] = {}
        for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
            by_identity[int(label)].append(index)
            identity_group.setdefault(int(label), group)
        self.members = {ident: np.asarray(idx) for ident, idx in by_identity.items()}
        per_group: dict[str, list[int]] = defaultdict(list)
        for ident, group in identity_group.items():
            per_group[group].append(ident)
        self.group_ids = {g: np.asarray(sorted(ids)) for g, ids in per_group.items()}
        present = [g for g in self.group_ids if shares.get(g, 0.0) > 0]
        if not present:
            raise ValueError("no source group with a positive share")
        total = sum(shares[g] for g in present)
        self.groups = present
        self.probs = np.asarray([shares[g] / total for g in present])

    @property
    def batch_size(self) -> int:
        return self.p * self.k

    def batch(self) -> np.ndarray:
        """One batch of ``P*K`` sample indices (identities distinct within the batch)."""

        counts = self.rng.multinomial(self.p, self.probs)
        out: list[np.ndarray] = []
        for group, count in zip(self.groups, counts, strict=True):
            if count == 0:
                continue
            ids = self.group_ids[group]
            chosen = self.rng.choice(ids, size=count, replace=count > len(ids))
            for ident in chosen:
                members = self.members[int(ident)]
                out.append(self.rng.choice(members, size=self.k, replace=len(members) < self.k))
        return np.concatenate(out)

    def state(self) -> dict:
        return self.rng.bit_generator.state

    def load_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state
