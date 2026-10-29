#!/usr/bin/env python3
"""Kill orphaned sweep / simulation worker processes.

Use when a sweep was interrupted and child Python workers are still holding GPU/CPU.
Safe to run anytime — only targets processes whose command line matches this repo's
runners and whose parent is init (PPID 1) or no longer alive.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time


def _repo_markers() -> tuple[str, ...]:
    return (
        "run_sweep_axes.py",
        "run_banding_comparison.py",
        "continuous_patterns.experiments.run",
        "slab_kinetic_jax",
    )


def _read_cmdline(pid: int) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def _read_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def find_stale_pids(*, include_all_matching: bool) -> list[int]:
    self_pid = os.getpid()
    out: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        cmd = _read_cmdline(pid)
        if not cmd or not any(m in cmd for m in _repo_markers()):
            continue
        if "cleanup_stale_workers.py" in cmd:
            continue
        if include_all_matching:
            out.append(pid)
            continue
        ppid = _read_ppid(pid)
        if ppid is None:
            continue
        if ppid == 1 or not _pid_alive(ppid):
            out.append(pid)
    return sorted(set(out))


def _terminate_pids(pids: list[int], *, grace_s: float) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_s
    alive = list(pids)
    while alive and time.monotonic() < deadline:
        alive = [p for p in alive if _pid_alive(p)]
        if alive:
            time.sleep(0.2)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kill stale continuous-patterns worker PIDs.")
    parser.add_argument(
        "--all-matching",
        action="store_true",
        help="Kill every matching Python process, not only orphaned ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print PIDs that would be killed.",
    )
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("cleanup_stale_workers: /proc scan is Linux-only", file=sys.stderr)
        return 2

    pids = find_stale_pids(include_all_matching=bool(args.all_matching))
    if not pids:
        print("No stale worker processes found.")
        return 0

    lines = []
    for pid in pids:
        lines.append(f"  {pid}  {_read_cmdline(pid).strip()}")
    print(f"Found {len(pids)} process(es):")
    print("\n".join(lines))

    if args.dry_run:
        return 0

    _terminate_pids(pids, grace_s=3.0)
    remaining = [p for p in pids if _pid_alive(p)]
    if remaining:
        print(f"WARNING: still alive after SIGKILL: {remaining}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
