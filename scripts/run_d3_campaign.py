#!/usr/bin/env python3
"""EXP-D3 orchestrator: re-render master figure, D3H sweep, analysis."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    full = f"uv run {' '.join(cmd)} 2>&1 | tee {log}"
    print(f"[run_d3_campaign] {full}")
    subprocess.run(full, shell=True, check=True, cwd=_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D3 campaign")
    parser.add_argument(
        "--master-dir",
        type=Path,
        default=_ROOT / "results/master_comparison/20260609T110139Z",
        help="Existing completed master comparison directory to re-render",
    )
    parser.add_argument("--skip-rerender", action="store_true")
    parser.add_argument("--skip-d3h", action="store_true")
    parser.add_argument("--d3h-sweep-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    master_dir = args.master_dir if args.master_dir.is_absolute() else _ROOT / args.master_dir

    if not args.skip_rerender:
        # Copy updated comparison config (Heaney labels) then re-evaluate statuses.
        import shutil

        cfg_src = _ROOT / "experiments/comparisons/d3_master_comparison.yaml"
        shutil.copy2(cfg_src, master_dir / "comparison_config.yaml")
        _run(
            [
                "python",
                "scripts/run_banding_comparison.py",
                "--finalize-outputs",
                str(master_dir),
            ],
            log=Path("logs/d3_rerender_master.log"),
        )

    d3h_dir = args.d3h_sweep_dir
    if not args.skip_d3h:
        _run(["python", "scripts/run_d3h_sweep.py"], log=Path("logs/d3h_sweep.log"))
        root = _ROOT / "results/slab_twophase/d3h_heaney_diag"
        subs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if not subs:
            raise RuntimeError("D3H sweep produced no output directory")
        d3h_dir = subs[-1]

    if d3h_dir is not None:
        d3h_dir = d3h_dir if d3h_dir.is_absolute() else _ROOT / d3h_dir
        _run(
            ["python", "scripts/analyze_d3.py", str(d3h_dir)],
            log=Path("logs/d3_analyze.log"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
