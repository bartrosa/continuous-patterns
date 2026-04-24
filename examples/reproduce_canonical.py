#!/usr/bin/env python3
"""Reproduce canonical agate morphologies from nested templates.

Intended for:

- Phase 4 regeneration against archived baselines (GPU, production ``n``, ``T``)
- Demonstration of the programmatic API (vs CLI)

Default uses full template parameters (typically ``n=512``, ``T=10000``). For a
short local smoke, set ``CP_REPRODUCE_MINI=1`` to cap grid size and horizon.

Each run prints wall time (seconds) for the user's regeneration log.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one

_REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = _REPO_ROOT / "src" / "continuous_patterns" / "experiments" / "templates"

CANONICAL_BASELINES = [
    "agate_ch_baseline",
    "stress_off",
    "validation_uniform_uniaxial_calibrated",
    "validation_pure_shear_calibrated",
    "validation_biaxial_calibrated",
    "stress_flamant_B_0_25",
    "stress_pressure_gradient_0_25",
    "agate_stage2_gamma_5",
]


def reproduce_one(name: str, results_root: Path) -> float:
    """Load template ``{name}.yaml``, optionally downsize, run, write artifacts.

    Returns
    -------
    float
        Wall time in seconds for the ``run_one`` call.
    """
    cfg_path = TEMPLATE_DIR / f"{name}.yaml"
    cfg = load_run_config(cfg_path)
    if os.environ.get("CP_REPRODUCE_MINI", "0") == "1":
        cfg["geometry"]["n"] = min(int(cfg["geometry"]["n"]), 32)
        cfg["time"]["T"] = min(float(cfg["time"]["T"]), 2.0)
    print(f"Running {name}...")
    t0 = time.perf_counter()
    result = run_one(cfg, results_root=results_root)
    wall_s = time.perf_counter() - t0
    assert result.paths is not None
    print(f"  → {result.paths.root}")
    print(f"  wall time: {wall_s:.1f} s")
    return wall_s


def main() -> None:
    """Run all entries in :data:`CANONICAL_BASELINES` sequentially."""
    results_root = _REPO_ROOT / "results" / "canonical"
    results_root.mkdir(parents=True, exist_ok=True)
    total_s = 0.0
    for name in CANONICAL_BASELINES:
        total_s += reproduce_one(name, results_root)
    print(f"Total wall time (all runs): {total_s:.1f} s")


if __name__ == "__main__":
    main()
