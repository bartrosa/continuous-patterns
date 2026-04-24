#!/usr/bin/env python3
"""Reproduce eight canonical paper-v2 runs via ``continuous_patterns.experiments.run``.

Runs are driven as subprocesses (``uv run python -m …``) from the repository root
so layered config paths and ``experiments/solver_settings.yaml`` resolve correctly.

Environment:

- ``CP_REPRODUCE_MINI=1`` — run a three-run subset with ``CP_OVERRIDE_T=250`` for a quicker check.
- ``CP_OVERRIDE_T`` — optional horizon override (seconds); applied by ``run.py`` after load.
- ``CP_LOG_LEVEL`` — passed as ``--log-level`` (default ``INFO``).
- ``CP_NO_PROGRESS=1`` — adds ``--no-progress``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = _REPO_ROOT / "experiments" / "canonical"
RESULTS_DIR = _REPO_ROOT / "results" / "canonical"

CANONICAL_BASELINES = [
    "no_pinning",
    "ratchet_only",
    "medium_pinning",
    "strong_pinning",
    "medium_pinning_seed123",
    "stress_uniform_biaxial",
    "stress_flamant",
    "agate_stage2_gamma_5",
]

_mini = os.environ.get("CP_REPRODUCE_MINI", "").strip().lower()
if _mini in {"1", "true", "yes"}:
    CANONICAL_BASELINES = ["no_pinning", "medium_pinning", "agate_stage2_gamma_5"]
    OVERRIDE_T = 250.0
else:
    OVERRIDE_T = None


def run_one(name: str) -> int:
    cfg_path = CANONICAL_DIR / f"{name}.yaml"
    if not cfg_path.is_file():
        print(f"[skip] {name}: {cfg_path} not found", file=sys.stderr)
        return 1

    print(f"\n=== {name} ===")
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "continuous_patterns.experiments.run",
        "--config",
        str(cfg_path),
        "--out-dir",
        str(RESULTS_DIR),
        "--log-level",
        os.environ.get("CP_LOG_LEVEL", "INFO"),
    ]
    if os.environ.get("CP_NO_PROGRESS", "0") == "1":
        cmd.append("--no-progress")

    env = dict(os.environ)
    if OVERRIDE_T is not None:
        env["CP_OVERRIDE_T"] = str(OVERRIDE_T)

    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    return int(proc.returncode)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for name in CANONICAL_BASELINES:
        rc = run_one(name)
        if rc != 0:
            failed.append(name)

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {len(failed)} of {len(CANONICAL_BASELINES)}")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"SUCCESS: all {len(CANONICAL_BASELINES)} baselines completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
