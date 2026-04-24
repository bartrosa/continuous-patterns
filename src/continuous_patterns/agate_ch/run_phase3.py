"""Phase 3 — seven calibrated T=10000 stress runs, markdown report, composite ψ figure.

Usage::

    uv run python -m continuous_patterns.agate_ch.run_phase3 --no-progress
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[3]

RUNS: list[tuple[str, str]] = [
    (
        "validation_uniform_uniaxial_calibrated",
        "configs/agate_ch/stress/validation_uniform_uniaxial_calibrated.yaml",
    ),
    (
        "validation_pure_shear_calibrated",
        "configs/agate_ch/stress/validation_pure_shear_calibrated.yaml",
    ),
    ("validation_biaxial_calibrated", "configs/agate_ch/stress/validation_biaxial_calibrated.yaml"),
    ("stress_flamant_B_0_25", "configs/agate_ch/stress/flamant_B_0_25.yaml"),
    ("stress_flamant_B_0_1", "configs/agate_ch/stress/flamant_B_0_1.yaml"),
    ("stress_pressure_gradient_0_25", "configs/agate_ch/stress/pressure_gradient_0_25.yaml"),
    ("stress_pressure_gradient_0_5", "configs/agate_ch/stress/pressure_gradient_0_5.yaml"),
]


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary.json"
    return json.loads(p.read_text())


def _morphology_line(summ: dict[str, Any]) -> str:
    mf = summ.get("metrics_at_final") or {}
    sc = summ.get("stability_scan") or {}
    parts = [
        str(mf.get("classification", "")),
        str(sc.get("morphology_hint", "")),
    ]
    return "; ".join(p for p in parts if p)


def _write_report(out_root: Path, run_dirs: dict[str, Path]) -> None:
    lines = [
        "# Phase 3 — calibrated stress coupling (T=10000)",
        "",
        f"Output root: `{out_root}`",
        "",
        (
            "Diagnostics: **`main_silica_window_drifts`** in each `summary.json` reports "
            "χ-weighted cavity silica relative change (%) over three 100-step windows of the "
            "**main** run (transient early vs steady late). "
            "**`spectral_mass_conservation.leak_pct`** is still the short **auxiliary** "
            "periodic blob run (NOT the main trajectory)."
        ),
        "",
        "## Mass conservation",
        "",
        "| run id | max_phi_sum | Option B % | spectral aux leak % | "
        "first100 χ-silica % | mid100 χ-silica % | last100 χ-silica % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    corr_txt = ""
    for rid, _yp in RUNS:
        rd = run_dirs[rid]
        summ = _load_summary(rd)
        mc = summ.get("mass_conservation") or {}
        sm = summ.get("spectral_mass_conservation") or {}
        msw = summ.get("main_silica_window_drifts") or {}
        opt_d = sm.get("leak_pct", float("nan"))
        opt_b = summ.get("mass_balance_percent", float("nan"))
        mx = mc.get("max_phi_sum", float("nan"))
        f1 = msw.get("first_100_steps_leak_pct", float("nan"))
        mid = msw.get("middle_100_steps_leak_pct", float("nan"))
        la = msw.get("last_100_steps_leak_pct", float("nan"))
        lines.append(f"| {rid} | {mx} | {opt_b} | {opt_d} | {f1} | {mid} | {la} |")
        if rid == "validation_uniform_uniaxial_calibrated":
            pc = summ.get("psi_corr_vs_aniso_10x") or {}
            corr_txt = str(pc.get("pearson_r", "n/a"))

    lines += [
        "",
        "## ψ correlation (validation uniaxial vs aniso_10x reference)",
        "",
        f"- Pearson **r** = `{corr_txt}` (cavity mask `r < R`; "
        "see `summary.json` → `psi_corr_vs_aniso_10x`). "
        "Target was >0.8 for κ-anisotropy parity; ψ-stress is a weaker / different operator, "
        "so a lower **r** can still occur even when banding looks similar.",
        "",
        "## Morphology (final Jabłczyński classification + FFT hint)",
        "",
    ]
    for rid, _yp in RUNS:
        summ = _load_summary(run_dirs[rid])
        lines.append(f"- **{rid}:** {_morphology_line(summ)}")
    lines.append("")
    (out_root / "phase3_report.md").write_text("\n".join(lines))


def _composite_psi(out_root: Path, run_dirs: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 7, figsize=(18, 3.2), constrained_layout=True)
    for ax, (rid, _) in zip(axes, RUNS, strict=True):
        npz = run_dirs[rid] / "final_state.npz"
        if not npz.is_file():
            ax.axis("off")
            ax.set_title(rid[:18] + "\n(missing npz)")
            continue
        z = np.load(npz)
        psi = np.asarray(z["phi_m"], dtype=np.float64) - np.asarray(z["phi_c"], dtype=np.float64)
        ax.imshow(psi.T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(rid.replace("validation_", "").replace("stress_", ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("φ_m − φ_c at T=10000 (Phase 3)", fontsize=11)
    fig.savefig(out_root / "phase3_psi_composite.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 calibrated stress batch")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--report-only",
        type=str,
        default="",
        help=(
            "rebuild phase3_report.md (+ composite) from an existing phase3_calibrated_* directory"
        ),
    )
    args = ap.parse_args()

    if args.report_only:
        out_root = Path(args.report_only)
        if not out_root.is_absolute():
            out_root = _REPO / out_root
        run_dirs = {rid: out_root / rid for rid, _ in RUNS}
        _write_report(out_root, run_dirs)
        _composite_psi(out_root, run_dirs)
        print(f"Updated {out_root / 'phase3_report.md'}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _REPO / "results" / "agate_ch" / f"phase3_calibrated_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs: dict[str, Path] = {}
    for rid, rel_cfg in RUNS:
        yp = _REPO / rel_cfg
        if not yp.is_file():
            print(f"Missing config {yp}", file=sys.stderr)
            sys.exit(1)
        run_dir = out_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs[rid] = run_dir

    (out_root / "manifest.json").write_text(
        json.dumps({"timestamp": ts, "runs": [{"id": r, "config": c} for r, c in RUNS]}, indent=2)
    )

    if args.dry_run:
        print("dry-run", out_root)
        return

    for rid, rel_cfg in RUNS:
        yp = _REPO / rel_cfg
        cmd = [
            sys.executable,
            "-m",
            "continuous_patterns.agate_ch.run",
            "--config",
            str(yp),
            "--out-dir",
            str(run_dirs[rid]),
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        print("→", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(_REPO), check=True)

    _write_report(out_root, run_dirs)
    _composite_psi(out_root, run_dirs)
    print(f"Wrote {out_root / 'phase3_report.md'} and composite PNG.")


if __name__ == "__main__":
    main()
