"""Short calibration sweep: uniform_uniaxial at σ₀ ∈ {0.5, 1.0, 2.0}, T=3000.

Example::

    uv run python -m continuous_patterns.agate_ch.run_uniform_uniaxial_sigma_scan --no-progress
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
    ap = argparse.ArgumentParser(description="Uniform uniaxial σ₀ calibration scan.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cfg_dir = root / "configs" / "agate_ch" / "stress"
    names = [
        "validation_uniform_uniaxial_sigma_0_5",
        "validation_uniform_uniaxial_sigma_1_0",
        "validation_uniform_uniaxial_sigma_2_0",
    ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = root / "results" / "agate_ch" / f"uniform_uniaxial_sigma_scan_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"timestamp": ts, "batch": "uniform_uniaxial_sigma_scan", "runs": []}
    exe = sys.executable

    print(f"Uniform uniaxial σ₀ scan → {sweep_root}\n", flush=True)

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
        print(f"[{i}/{len(names)}] {name}", flush=True)
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=str(root), check=True)
        print(f"       {time.perf_counter() - t0:.1f} s", flush=True)
        manifest["runs"].append({"id": name, "out_dir": str(out_dir.relative_to(root))})

    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {sweep_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
