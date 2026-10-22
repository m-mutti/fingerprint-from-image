"""Licence lineage: data + pretrained-weight licences → "open" vs "research-only"."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["OPEN_LICENCES", "lineage", "weights_licence"]

OPEN_LICENCES = frozenset({"CC-BY-4.0", "CC0-1.0", "Apache-2.0", "MIT", "BSD-3-Clause",
                           "generated-in-repo"})

# Licences of pretrained weights by timm model-name prefix (checked 2026-10: model cards).
_WEIGHTS = (
    ("convnextv2_", "CC-BY-NC-4.0"),  # Meta FCMAE ConvNeXt-V2 weights
    ("vit_small_patch14_dinov2", "Apache-2.0"),
    ("vit_base_patch14_dinov2", "Apache-2.0"),
    ("vit_small_patch14_reg4_dinov2", "Apache-2.0"),
    ("vit_base_patch14_reg4_dinov2", "Apache-2.0"),
    ("resnet", "Apache-2.0"),
    ("efficientnet", "Apache-2.0"),
    ("mobilenetv", "Apache-2.0"),
    ("repvit", "Apache-2.0"),
)

_DATA = {
    "SOCOFing": "CC-BY-NC-SA-4.0",
    "FVC2000": "FVC-research",
    "FVC2002": "FVC-research",
    "FVC2004": "FVC-research",
    "l3sf": "CC-BY-4.0",
    "mendeley_family": "CC-BY-4.0",
    "TDFD": "TDFD-non-commercial",
    "neurotech": "Neurotechnology-sample",
    "synthetic": "generated-in-repo",
}


def weights_licence(backbone: str, pretrained: bool = True) -> str:
    if not pretrained:
        return "generated-in-repo"
    for prefix, licence in _WEIGHTS:
        if backbone.startswith(prefix):
            return licence
    return "unknown"


def lineage(train_groups: Mapping[str, float], backbone: str, pretrained: bool = True) -> dict:
    """Licences of every training source with a positive share plus the backbone weights."""

    data = {g: _DATA.get(g, "unknown") for g, share in train_groups.items() if share > 0}
    weights = weights_licence(backbone, pretrained)
    is_open = all(v in OPEN_LICENCES for v in [*data.values(), weights])
    return {"data": data, "weights": weights, "open": is_open,
            "track": "open" if is_open else "research-only"}
