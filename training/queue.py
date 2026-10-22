"""Single GPU job queue: ``submit`` / ``worker`` / ``status`` / ``rerun``.

Jobs are YAML files moved through ``runs/queue/{pending,running,done,failed}/``. One
worker owns the GPU and runs one job at a time (highest ``priority`` first, then oldest).
A job left in ``running/`` by a crash is moved back to ``pending/`` on worker start and
resumes from its ``state.pt``; an already-done ``train`` key is skipped by the trainer
itself. Guards: a job whose heartbeat (written every logged step) is older than the
stall limit is killed; the failure reason (stall / oom / divergence / nonfinite / exit
code) is stored in the job file and the registry.

Usage::

    python -m training.queue submit training/configs/m1_fvcfull.yaml --priority 5
    # any other GPU job, e.g. the SOTA baseline:
    python -m training.queue submit-module --name sota --cwd D:/repo baseline.train -- --epochs 40
    python -m training.queue worker              # long-running GPU owner
    python -m training.queue status
    python -m training.queue rerun <train-key> [--force]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from training.keys import runs_root

__all__ = ["Queue", "classify_failure", "main"]

STATES = ("pending", "running", "done", "failed")


def classify_failure(log_tail: str, returncode: int | None) -> str:
    """Map a failed job's log tail to a short reason."""

    text = log_tail.lower()
    if "outofmemoryerror" in text or "out of memory" in text:
        return "oom"
    for reason in ("divergence", "nonfinite"):
        if f"trainingaborted: {reason}" in text or f"{reason}:" in text:
            return reason
    if "contaminationerror" in text:
        return "contamination"
    return f"exit {returncode}"


class Queue:
    """Filesystem job queue rooted at ``runs/queue``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or runs_root() / "queue"
        for state in (*STATES, "logs", "configs", "heartbeat"):
            (self.root / state).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ submission
    def submit(self, *, name: str, argv: list[str], priority: int = 0,
               meta: dict | None = None) -> Path:
        """Queue a job running ``python <argv>``; returns the job file."""

        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.perf_counter_ns() % 1_000_000:06d}"
        job = {"name": name, "priority": int(priority), "argv": list(argv),
               "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
        path = self.root / "pending" / f"{stamp}_{name}.yaml"
        path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
        return path

    def submit_config(self, config: dict, *, name: str, priority: int = 0,
                      force: bool = False) -> Path:
        """Freeze a resolved training config and queue ``python -m training.train``."""

        cfg_path = self.root / "configs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        argv = ["-m", "training.train", "--config", str(cfg_path)] + (["--force"] if force else [])
        return self.submit(name=name, argv=argv, priority=priority,
                           meta={"config": str(cfg_path)})

    # ------------------------------------------------------------ state
    def jobs(self, state: str) -> list[Path]:
        return sorted((self.root / state).glob("*.yaml"))

    def next_job(self) -> Path | None:
        pending = self.jobs("pending")
        if not pending:
            return None
        def order(p: Path) -> tuple[int, str]:
            job = yaml.safe_load(p.read_text(encoding="utf-8"))
            return (-int(job.get("priority", 0)), p.name)
        return min(pending, key=order)

    def _move(self, job: Path, state: str, **updates: object) -> Path:
        data = yaml.safe_load(job.read_text(encoding="utf-8"))
        data.update(updates)
        target = self.root / state / job.name
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        if target != job:
            job.unlink()
        return target

    def recover(self) -> list[Path]:
        """Move jobs orphaned in ``running/`` back to ``pending/`` (they resume)."""

        return [self._move(job, "pending", recovered=True) for job in self.jobs("running")]

    # ------------------------------------------------------------ execution
    def run_one(self, job: Path, *, stall_seconds: float = 1200, poll: float = 5.0) -> Path:
        """Run one pending job to completion; returns its final job file."""

        running = self._move(job, "running", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
        spec = yaml.safe_load(running.read_text(encoding="utf-8"))
        log = self.root / "logs" / f"{running.stem}.log"
        heartbeat = self.root / "heartbeat" / f"{running.stem}.hb"
        env = {**os.environ, "FFI_HEARTBEAT": str(heartbeat), "PYTHONIOENCODING": "utf-8"}
        start = time.time()
        with log.open("ab") as out:
            proc = subprocess.Popen([sys.executable, *spec["argv"]], stdout=out,
                                    stderr=subprocess.STDOUT, env=env,
                                    cwd=spec.get("cwd") or Path(__file__).resolve().parents[1])
            reason = None
            while proc.poll() is None:
                time.sleep(poll)
                last = heartbeat.stat().st_mtime if heartbeat.is_file() else start
                if time.time() - last > stall_seconds:
                    reason = "stall"
                    _kill_tree(proc)
                    break
        code = proc.wait()
        finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        if reason is None and code == 0:
            return self._move(running, "done", finished=finished, returncode=0)
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        reason = reason or classify_failure(tail, code)
        return self._move(running, "failed", finished=finished, returncode=code, reason=reason)

    def worker(self, *, once: bool = False, poll: float = 10.0,
               stall_seconds: float = 1200) -> None:
        """Process jobs forever (or until the queue is empty with ``once``)."""

        self.recover()
        while True:
            job = self.next_job()
            if job is None:
                if once:
                    return
                time.sleep(poll)
                continue
            self.run_one(job, stall_seconds=stall_seconds)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       check=False)
    else:
        proc.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("config", type=Path)
    s.add_argument("--priority", type=int, default=0)
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--force", action="store_true")
    m = sub.add_parser("submit-module", help="queue `python -m MODULE ARGS` (options first)")
    m.add_argument("module")
    m.add_argument("--name", required=True)
    m.add_argument("--priority", type=int, default=0)
    m.add_argument("--cwd", type=Path)
    m.add_argument("args", nargs=argparse.REMAINDER)
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true")
    w.add_argument("--stall-minutes", type=float, default=20.0)
    sub.add_parser("status")
    r = sub.add_parser("rerun")
    r.add_argument("key")
    r.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    queue = Queue()
    if args.command == "submit":
        from training.config import load_config

        overrides = {k: yaml.safe_load(v) for k, v in (i.split("=", 1) for i in args.set)}
        cfg = load_config(args.config, overrides)
        print(queue.submit_config(cfg.to_dict(), name=cfg.name, priority=args.priority,  # noqa: T201
                                  force=args.force))
    elif args.command == "submit-module":
        rest = args.args[1:] if args.args[:1] == ["--"] else args.args
        meta = {"cwd": str(args.cwd)} if args.cwd else None
        print(queue.submit(name=args.name, argv=["-m", args.module, *rest],  # noqa: T201
                           priority=args.priority, meta=meta))
    elif args.command == "worker":
        queue.worker(once=args.once, stall_seconds=args.stall_minutes * 60)
    elif args.command == "status":
        for state in STATES:
            for job in queue.jobs(state):
                spec = yaml.safe_load(job.read_text(encoding="utf-8"))
                print(f"{state:8s} p={spec.get('priority', 0):3d} {job.name} "  # noqa: T201
                      f"{spec.get('reason', '')}")
    else:
        from training.registry import Registry

        rows = [row for row in Registry().rows("train") if row["key"].startswith(args.key)]
        if len(rows) != 1:
            raise SystemExit(f"key prefix {args.key!r} matches {len(rows)} train records")
        print(queue.submit_config(rows[0]["config"], name=rows[0]["name"] or "rerun",  # noqa: T201
                                  force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
