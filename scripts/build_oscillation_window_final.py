#!/usr/bin/env python3
"""Build combined heatmaps + ``table_all_runs_final.md`` from main + fine sweeps."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_outputs():
    p = _REPO / "scripts" / "slab_kinetic_oscillation_window_outputs.py"
    spec = importlib.util.spec_from_file_location("osc_out", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("main_sweep_dir", type=Path)
    ap.add_argument("fine_sweep_dir", type=Path)
    ap.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=_REPO / "results/slab_kinetic/oscillation_window_final",
        help="Output directory for *_final.png and table_all_runs_final.md",
    )
    args = ap.parse_args(argv)

    mod = _load_outputs()
    mod.write_final_combined_artifacts(
        args.main_sweep_dir.resolve(),
        args.fine_sweep_dir.resolve(),
        args.out_dir.resolve(),
    )
    print(f"Wrote combined figures under {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
