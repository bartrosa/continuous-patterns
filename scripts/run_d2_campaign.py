#!/usr/bin/env python3
"""EXP-D2 orchestrator: PART A D2H sweep + analysis, PART B master comparison figure."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    full = f"uv run {' '.join(cmd)} 2>&1 | tee {log}"
    print(f"[run_d2_campaign] {full}")
    subprocess.run(full, shell=True, check=True, cwd=_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D2 full campaign")
    parser.add_argument(
        "--d1-sweep-dir",
        type=Path,
        default=_ROOT / "results/slab_twophase/d1_discrimination/20260607T141449Z",
    )
    parser.add_argument(
        "--skip-d2h",
        action="store_true",
        help="Skip PART A compute (use existing D2H sweep dir via --d2h-sweep-dir)",
    )
    parser.add_argument("--d2h-sweep-dir", type=Path, default=None)
    parser.add_argument(
        "--e1-run-dir",
        type=Path,
        default=None,
        help="E1 run directory for A1 sawtooth verification",
    )
    parser.add_argument("--skip-master", action="store_true")
    args = parser.parse_args(argv)

    d2h_dir = args.d2h_sweep_dir
    if not args.skip_d2h:
        _run(
            ["python", "scripts/run_d2h_sweep.py"],
            log=Path("logs/d2h_sweep.log"),
        )
        root = _ROOT / "results/slab_twophase/d2h_heaney_movingfront"
        subs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
        if not subs:
            raise RuntimeError("D2H sweep produced no output directory")
        d2h_dir = subs[-1]

    if d2h_dir is None:
        raise RuntimeError("provide --d2h-sweep-dir or run PART A")

    d2h_dir = d2h_dir if d2h_dir.is_absolute() else _ROOT / d2h_dir

    analyze_cmd = ["python", "scripts/analyze_d2.py", str(d2h_dir)]
    if args.e1_run_dir:
        analyze_cmd += ["--e1-run-dir", str(args.e1_run_dir)]
    _run(analyze_cmd, log=Path("logs/d2_analyze.log"))

    if not args.skip_master:
        print(f"[run_d2_campaign] D1 ingest dir: {args.d1_sweep_dir}")
        _run(
            [
                "python",
                "scripts/run_banding_comparison.py",
                "experiments/comparisons/d2_master_comparison.yaml",
                "results/master_comparison",
            ],
            log=Path("logs/d2_master_comparison.log"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
