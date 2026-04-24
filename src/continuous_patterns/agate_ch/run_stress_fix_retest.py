"""Re-run stress configs after divergence-form ψ coupling fix (Phase 4).

Writes ``results/agate_ch/stress_fix_retest_<timestamp>/`` with one subdirectory per config.

Example::

    uv run python -m continuous_patterns.agate_ch.run_stress_fix_retest --no-progress
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
    ap = argparse.ArgumentParser(description="Stress coupling fix — full retest sweep.")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cfg_dir = root / "configs" / "agate_ch" / "stress"
    names = [
        "stress_off",
        "validation_uniform_uniaxial",
        "flamant_B_1_0",
        "pure_shear_sigma_1_0",
        "pressure_gradient_sigma_1_0",
        "kirsch_sigma_1_0",
    ]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = root / "results" / "agate_ch" / f"stress_fix_retest_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"timestamp": ts, "runs": []}
    exe = sys.executable

    print(f"Stress fix retest sweep → {sweep_root}\n", flush=True)

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
        print(f"[{i}/{len(names)}] {name} → {out_dir.relative_to(root)}", flush=True)
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=str(root), check=True)
        elapsed = time.perf_counter() - t0
        print(f"       {elapsed / 60.0:.2f} min", flush=True)
        manifest["runs"].append(
            {"id": name, "out_dir": str(out_dir.relative_to(root)), "wall_seconds": elapsed}
        )

    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {sweep_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
