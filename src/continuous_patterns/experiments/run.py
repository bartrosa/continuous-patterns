"""Canonical CLI: nested config → simulate → ``results/`` tree.

``python -m continuous_patterns.experiments.run --config …`` (``docs/ARCHITECTURE.md`` §6.1).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.core.io import (
    allocate_run_dir,
    load_run_config,
    save_final_state_npz,
    save_run_config,
    save_summary,
    write_figures_final,
)
from continuous_patterns.core.types import SimResult
from continuous_patterns.models import agate_ch, agate_stage2

MODEL_DISPATCH: dict[str, Any] = {
    "agate_ch": agate_ch.simulate,
    "agate_stage2": agate_stage2.simulate,
}


def run_one(
    cfg: dict[str, Any],
    *,
    results_root: Path | None = None,
    chunk_size: int = 2000,
    write_artifacts: bool = True,
) -> SimResult:
    """Run one simulation; optionally write ``results/`` artifacts.

    Dispatches on ``cfg["experiment"]["model"]``. When ``write_artifacts`` is
    true, ``results_root`` must be set (defaults in the CLI to ``./results``).
    """
    model_name = cfg["experiment"]["model"]
    if model_name not in MODEL_DISPATCH:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {sorted(MODEL_DISPATCH)}")
    simulate_fn = MODEL_DISPATCH[model_name]
    result = simulate_fn(cfg, chunk_size=chunk_size)

    if write_artifacts:
        if results_root is None:
            raise ValueError("results_root is required when write_artifacts=True")
        paths = allocate_run_dir(
            experiment_name=str(cfg["experiment"]["name"]),
            results_root=Path(results_root),
        )
        save_run_config(paths.config_yaml, cfg)
        save_summary(paths.summary_json, result.diagnostics)
        save_final_state_npz(
            paths.final_state_npz,
            phi_m=np.asarray(result.state_final.phi_m),
            phi_c=np.asarray(result.state_final.phi_c),
            c=np.asarray(result.state_final.c),
            chi=None,
        )
        gcfg = cfg["geometry"]
        write_figures_final(
            paths.root,
            phi_m=np.asarray(result.state_final.phi_m),
            phi_c=np.asarray(result.state_final.phi_c),
            c=np.asarray(result.state_final.c),
            L=float(gcfg["L"]),
            R=float(gcfg.get("R", 0.0)),
            chi=None,
        )
        return replace(result, paths=paths)

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--config``, ``--out-dir``, ``--chunk-size``, ``--no-write``."""
    parser = argparse.ArgumentParser(
        prog="continuous_patterns.experiments.run",
        description="Run one simulation from a nested YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to nested YAML config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Results root directory (default: ./results).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="IMEX chunk size.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing artifacts (smoke testing).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_run_config(Path(args.config))
    except ValueError as e:
        print(f"Config load failed: {e}", file=sys.stderr)
        return 2

    try:
        result = run_one(
            cfg,
            results_root=None if args.no_write else args.out_dir,
            chunk_size=int(args.chunk_size),
            write_artifacts=not args.no_write,
        )
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        return 1

    if result.paths is not None:
        print(f"Run complete: {result.paths.root}")
    else:
        print("Run complete (no artifacts written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
