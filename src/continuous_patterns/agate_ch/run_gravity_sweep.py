"""Run Experiment 4 gravity sweep: six ``c0_alpha`` runs by default, fixed seed.

Writes ``results/agate_ch/gravity_sweep_<timestamp>/{alpha_0_00,...}/`` and
``manifest.json`` for :mod:`plot_gravity_sweep`. Requires the optional
``physics.c0_alpha`` parameter in YAML; other experiments do not need it.

Use ``--only alpha_0_40 alpha_0_80`` (for example) to run a subset without
recomputing earlier α values.

Progress: each child run uses **tqdm** on stderr when YAML has
``diagnostics.progress_stderr: true`` (default in ``configs/agate_ch/gravity/``).
Use ``--no-progress`` to silence bars (e.g. CI or log capture).

Example::

    uv run python -m continuous_patterns.agate_ch.run_gravity_sweep
    uv run python -m continuous_patterns.agate_ch.run_gravity_sweep --no-progress
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(
        description="Sequential gravity (c0_alpha) sweep — forwards --no-progress to each run.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm step bar in each subprocess (same flag as agate_ch.run).",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "If set, run only these YAML stems (e.g. alpha_0_40 alpha_0_80). "
            "Must be among the six standard gravity configs. Order is preserved."
        ),
    )
    args = ap.parse_args(argv)

    cfg_dir = root / "configs" / "agate_ch" / "gravity"
    all_names = [
        "alpha_0_00",
        "alpha_0_05",
        "alpha_0_10",
        "alpha_0_20",
        "alpha_0_40",
        "alpha_0_80",
    ]
    if args.only:
        names = []
        for raw in args.only:
            stem = raw[:-5] if raw.endswith(".yaml") else raw
            if stem not in all_names:
                raise SystemExit(
                    f"unknown --only {stem!r}; allowed: {all_names}",
                )
            names.append(stem)
    else:
        names = list(all_names)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = root / "results" / "agate_ch" / f"gravity_sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"timestamp": ts, "runs": []}
    exe = sys.executable

    print(
        f"Experiment 4 gravity sweep\n"
        f"  sweep directory: {sweep_root}\n"
        f"  configs:         {cfg_dir}\n"
        f"  progress bar:    {'off' if args.no_progress else 'on (stderr, from YAML)'}\n",
        flush=True,
    )

    for i, name in enumerate(names, start=1):
        cfg = cfg_dir / f"{name}.yaml"
        if not cfg.is_file():
            raise FileNotFoundError(cfg)
        out_dir = sweep_root / name
        cmd = [
            exe,
            "-m",
            "continuous_patterns.agate_ch.run",
            "--config",
            str(cfg),
            "--out-dir",
            str(out_dir),
        ]
        if args.no_progress:
            cmd.append("--no-progress")

        print(
            f"\n[{i}/{len(names)}] === {name} ===\n"
            f"       config: {cfg.relative_to(root)}\n"
            f"       output: {out_dir.relative_to(root)}",
            flush=True,
        )
        t0 = time.perf_counter()
        subprocess.run(
            cmd,
            cwd=str(root),
            check=True,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"       done in {elapsed / 60.0:.2f} min ({elapsed:.1f} s)",
            flush=True,
        )
        manifest["runs"].append(
            {"id": name, "out_dir": str(out_dir.relative_to(root)), "wall_seconds": elapsed}
        )

    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    runs_done: list[dict[str, Any]] = manifest["runs"]
    total_s = sum(float(r["wall_seconds"]) for r in runs_done)
    print(
        f"\nGravity sweep complete.\n"
        f"  Manifest: {sweep_root / 'manifest.json'}\n"
        f"  Total wall time ({len(runs_done)} runs): {total_s / 60.0:.2f} min ({total_s:.1f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
