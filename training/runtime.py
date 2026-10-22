"""Process-level safety: single GPU lock, atomic file replace with retry, phase heartbeats.

* :class:`GpuLock` — one OS-level lock (``runs/gpu.lock``; ``msvcrt.locking`` on Windows,
  ``fcntl.flock`` elsewhere) held by whichever process trains on the GPU; any other job
  (ours, the baseline agent's, a human's) waits instead of sharing VRAM.
* :func:`atomic_replace` — ``os.replace`` with retries for WinError 5/32 (file briefly
  open in an evaluator, antivirus or indexer).
* :class:`Heartbeat` — touches the queue heartbeat file; ``phase(name)`` keeps beating from
  a thread during long non-step phases (cache build, probe, compile, validation) so the
  queue's stall guard only fires on genuine hangs of the step loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from training.keys import runs_root

__all__ = ["GpuLock", "Heartbeat", "atomic_replace"]

logger = logging.getLogger(__name__)


def atomic_replace(src: Path, dst: Path, retries: int = 20, delay: float = 0.5) -> None:
    """``os.replace`` retried on Windows sharing violations (WinError 5 / 32)."""

    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


class GpuLock:
    """Exclusive, blocking, cross-process GPU lock (released on process exit)."""

    def __init__(self, path: Path | None = None, poll: float = 15.0) -> None:
        self.path = path or runs_root() / "gpu.lock"
        self.poll = poll
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        waited = 0.0
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if waited == 0.0:
                    logger.info("GPU lock %s is held by another job; waiting", self.path)
                time.sleep(self.poll)
                waited += self.poll
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> GpuLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class Heartbeat:
    """Writes the queue heartbeat (``$FFI_HEARTBEAT``) on demand or from a phase thread."""

    def __init__(self, path: Path | None, interval: float = 30.0) -> None:
        self.path = path
        self.interval = interval

    def beat(self) -> None:
        if self.path is not None:
            self.path.write_text(str(time.time()), encoding="utf-8")

    @contextmanager
    def phase(self, name: str, max_seconds: float | None = None) -> Iterator[None]:
        """Beat every ``interval`` s while a long phase runs (at most ``max_seconds``).

        A cap keeps the stall guard meaningful inside the step loop: after it, only real
        step logs refresh the heartbeat.
        """

        stop = threading.Event()
        start = time.time()

        def loop() -> None:
            while not stop.wait(self.interval):
                if max_seconds is not None and time.time() - start > max_seconds:
                    return
                self.beat()

        self.beat()
        thread = threading.Thread(target=loop, name=f"heartbeat-{name}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)
            self.beat()
