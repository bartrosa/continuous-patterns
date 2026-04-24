"""Run three additional Experiment 6 stress modes (pure shear, pressure gradient, Kirsch).

Does not re-run Flamant or other prior stress sweep configs.

Example::

    uv run python -m continuous_patterns.agate_ch.run_stress_extra_sweep
    uv run python -m continuous_patterns.agate_ch.run_stress_extra_sweep --no-progress
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
    ap = argparse.ArgumentParser(description="Sequential Experiment 6 extra stress modes.")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm step bar in each subprocess.",
    )
    args = ap.parse_args(argv)

    cfg_dir = root / "configs" / "agate_ch" / "stress"
    names = [
        "pure_shear_sigma_1_0",
        "pressure_gradient_sigma_1_0",
        "kirsch_sigma_1_0",
    ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = root / "results" / "agate_ch" / f"stress_extra_sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"timestamp": ts, "runs": []}
    exe = sys.executable

    print(
        f"Experiment 6 extra stress sweep\n"
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
        subprocess.run(cmd, cwd=str(root), check=True)
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
    mod = "continuous_patterns.agate_ch.plot_stress_all_modes_comparison"
    print(
        f"\nExtra stress sweep complete.\n"
        f"  Manifest: {sweep_root / 'manifest.json'}\n"
        f"  Total wall time ({len(runs_done)} runs): "
        f"{total_s / 60.0:.2f} min ({total_s:.1f} s)\n"
        "  Next:\n"
        f"    uv run python -m {mod} \\\n"
        f"      --pure-shear-dir {runs_done[0]['out_dir']} \\\n"
        f"      --pressure-gradient-dir {runs_done[1]['out_dir']} \\\n"
        f"      --kirsch-dir {runs_done[2]['out_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
