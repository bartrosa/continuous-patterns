"""Long Run B for Experiment 2 — observe where the system saturates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


CFG = _repo_root() / "configs" / "agate_ch" / "stage_sequence" / "run_b_long.yaml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run Experiment 2 Run B with run_b_long.yaml (default T=100000). "
            "Extra args are forwarded to continuous_patterns.agate_ch.run "
            "(e.g. --T 200 --no-progress — do not use an extra `--` separator)."
        )
    )
    ap.add_argument(
        "--no-timestamp-dir",
        action="store_true",
        help=(
            "write to results/agate_ch/run_<timestamp> instead of stage_seq_run_b_long_<timestamp>"
        ),
    )
    ns, passthrough = ap.parse_known_args(argv)

    root = _repo_root()
    if not CFG.is_file():
        raise FileNotFoundError(f"Missing config: {CFG}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ns.no_timestamp_dir:
        out_dir = root / "results" / "agate_ch" / f"run_{ts}"
    else:
        out_dir = root / "results" / "agate_ch" / f"stage_seq_run_b_long_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "continuous_patterns.agate_ch.run",
        "--config",
        str(CFG),
        "--out-dir",
        str(out_dir),
        *passthrough,
    ]

    print("Long Run B — Experiment 2 stage II long horizon")
    print("Config:", CFG)
    print("Output dir:", out_dir)
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
