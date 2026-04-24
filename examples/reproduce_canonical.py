#!/usr/bin/env python3
"""Reproduce canonical agate morphologies from nested templates.

Intended for:

- Phase 4 regeneration against archived baselines
- Demonstration of the programmatic API (vs CLI)
- Quick smoke when ``CP_REPRODUCE_MINI`` is left at default ``1`` (small ``n``, short ``T``)

Set ``CP_REPRODUCE_MINI=0`` for full template parameters (long runs, large grids).
"""

from __future__ import annotations

import os
from pathlib import Path

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one

_REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = _REPO_ROOT / "src" / "continuous_patterns" / "experiments" / "templates"

CANONICAL_BASELINES = [
    "agate_ch_baseline",
    "agate_stage2_gamma_5",
]


def reproduce_one(name: str, results_root: Path) -> None:
    """Load template ``{name}.yaml``, optionally downsize, run, write artifacts."""
    cfg_path = TEMPLATE_DIR / f"{name}.yaml"
    cfg = load_run_config(cfg_path)
    if os.environ.get("CP_REPRODUCE_MINI", "1") == "1":
        cfg["geometry"]["n"] = min(int(cfg["geometry"]["n"]), 32)
        cfg["time"]["T"] = min(float(cfg["time"]["T"]), 2.0)
    print(f"Running {name}...")
    result = run_one(cfg, results_root=results_root)
    assert result.paths is not None
    print(f"  → {result.paths.root}")


def main() -> None:
    """Run all entries in :data:`CANONICAL_BASELINES`."""
    results_root = _REPO_ROOT / "results" / "canonical"
    results_root.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_BASELINES:
        reproduce_one(name, results_root)


if __name__ == "__main__":
    main()
