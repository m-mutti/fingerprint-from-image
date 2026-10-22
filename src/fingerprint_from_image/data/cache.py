"""On-disk cache of :class:`~.quality.ImageProbe` results keyed by path, size and mtime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .quality import DESCRIPTOR_SHAPE, ImageProbe

__all__ = ["ProbeCache"]

_DIM = 2 * DESCRIPTOR_SHAPE[0] * DESCRIPTOR_SHAPE[1]


def _stamp(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


class ProbeCache:
    """Probe results that stay valid while a file's size and mtime are unchanged.

    Stored as one ``.npz`` (descriptors as float16) so rebuilding manifests
    after adding a dataset only decodes the new images.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, tuple[tuple[int, int], ImageProbe]] = {}
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        with np.load(self.path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            descriptors = data["descriptors"]
        for index, item in enumerate(meta):
            probe = ImageProbe(
                path=item["path"], ok=item["ok"], sha256=item["sha256"], width=item["width"],
                height=item["height"], mode=item["mode"], issues=item["issues"],
                ridge_period=item["ridge"],
                descriptor=descriptors[index].astype(np.float32) if item["ok"] else None,
            )
            self._entries[item["path"]] = ((item["size"], item["mtime"]), probe)

    def get(self, path: str) -> ImageProbe | None:
        """Cached probe for ``path`` if the file is unchanged, else ``None``."""

        entry = self._entries.get(path)
        if entry is None:
            return None
        try:
            if _stamp(path) != entry[0]:
                return None
        except OSError:
            return None
        return entry[1]

    def put(self, probe: ImageProbe) -> None:
        try:
            stamp = _stamp(probe.path)
        except OSError:
            stamp = (-1, -1)
        self._entries[probe.path] = (stamp, probe)

    def save(self) -> None:
        """Write atomically next to the target path."""

        meta = []
        descriptors = np.zeros((len(self._entries), _DIM), dtype=np.float16)
        for index, (path, ((size, mtime), probe)) in enumerate(sorted(self._entries.items())):
            meta.append({"path": path, "size": size, "mtime": mtime, "ok": probe.ok,
                         "sha256": probe.sha256, "width": probe.width, "height": probe.height,
                         "mode": probe.mode, "issues": probe.issues, "ridge": probe.ridge_period})
            if probe.descriptor is not None:
                descriptors[index] = probe.descriptor
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(temporary, meta=np.array(json.dumps(meta)), descriptors=descriptors)
        temporary.replace(self.path)
