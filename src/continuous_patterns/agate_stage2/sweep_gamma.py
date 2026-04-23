"""Run γ-scan sweep for agate_stage2 (one run per config, fixed output folder names)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GAMMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = _repo_root()
    config_dir = root / "configs" / "agate_stage2"
    results_root = root / "results" / "agate_stage2"
    results_root.mkdir(parents=True, exist_ok=True)

    for gamma in GAMMA_VALUES:
        gamma_str = str(int(gamma)) if gamma == int(gamma) else str(gamma).replace(".", "_")
        config_path = config_dir / f"gamma_{gamma_str}.yaml"
        out_dir = results_root / f"stage2_gamma_{gamma_str}_0"
        print(f"\n=== Running γ={gamma} → {out_dir.relative_to(root)} ===")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "continuous_patterns.agate_stage2.run",
                "--config",
                str(config_path.relative_to(root)),
                "--out-dir",
                str(out_dir.relative_to(root)),
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
