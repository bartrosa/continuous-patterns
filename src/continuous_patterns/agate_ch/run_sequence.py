"""Run Stage I (agate_ch with front) then Stage II (high-γ, no reaction, from snapshot)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from continuous_patterns.agate_ch.run import _repo_root, agate_ch_results_dir

CONFIG_DIR = _repo_root() / "configs" / "agate_ch" / "stage_sequence"


def _run_cli(
    *,
    config_path: Path,
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> None:
    root = _repo_root()
    cfg_s = str(config_path.resolve())
    out_s = str(out_dir.resolve())
    cmd = [
        sys.executable,
        "-m",
        "continuous_patterns.agate_ch.run",
        "--config",
        cfg_s,
        "--out-dir",
        out_s,
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=root, check=True)


def run_stage_a(
    *,
    ts: str,
    extra_args: list[str] | None = None,
) -> Path:
    root = _repo_root()
    out = agate_ch_results_dir(root) / f"stage_seq_run_a_{ts}"
    print("=" * 60)
    print("STAGE I — concentric bands (γ=3, reaction + Dirichlet)")
    print("=" * 60)
    _run_cli(
        config_path=CONFIG_DIR / "run_a_stage1.yaml",
        out_dir=out,
        extra_args=extra_args,
    )
    npz = out / "final_state.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"expected {npz} (set diagnostics.save_final_state: true)")
    return out.resolve()


def run_stage_b(
    run_a_dir: Path,
    *,
    ts: str,
    extra_args: list[str] | None = None,
) -> Path:
    root = _repo_root()
    snapshot = Path(run_a_dir) / "final_state.npz"
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    template = yaml.safe_load((CONFIG_DIR / "run_b_stage2.yaml").read_text())
    ic = template.setdefault("initial_condition", {})
    ic["type"] = "from_snapshot"
    ic["snapshot_path"] = str(snapshot.resolve())

    out = agate_ch_results_dir(root) / f"stage_seq_run_b_{ts}"
    print("=" * 60)
    print("STAGE II — γ-driven relaxation (γ=6, no reaction / no Dirichlet)")
    print(f"Snapshot: {snapshot}")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        prefix="stage_seq_run_b_resolved_",
    ) as tf:
        yaml.safe_dump(template, tf, sort_keys=False)
        tmp_path = Path(tf.name)

    try:
        _run_cli(config_path=tmp_path, out_dir=out, extra_args=extra_args)
    finally:
        tmp_path.unlink(missing_ok=True)

    return out.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential Stage I → Stage II agate_ch runs")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="short horizons for pipeline checks (overrides via --T / snapshot on CLI runner)",
    )
    args_ns = ap.parse_args()

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    root = _repo_root()
    extra: list[str] | None = None
    if args_ns.smoke:
        extra = ["--T", "500", "--snapshot-every", "50"]

    run_a_dir = run_stage_a(ts=ts, extra_args=extra)
    print(f"\nStage I complete: {run_a_dir}\n")

    run_b_dir = run_stage_b(run_a_dir, ts=ts, extra_args=extra)
    print(f"\nStage II complete: {run_b_dir}\n")

    seq_dir = agate_ch_results_dir(root)
    seq_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seq_dir / "stage_sequence_latest.json"
    summary = {
        "experiment": "stage_sequence",
        "run_a": str(run_a_dir),
        "run_b": str(run_b_dir),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("SEQUENCE COMPLETE")
    print(f"Run A: {run_a_dir}")
    print(f"Run B: {run_b_dir}")
    print(f"Summary: {summary_path}")
    print("Comparison: uv run python -m continuous_patterns.agate_ch.plot_stage_sequence")
    print("=" * 60)


if __name__ == "__main__":
    main()
