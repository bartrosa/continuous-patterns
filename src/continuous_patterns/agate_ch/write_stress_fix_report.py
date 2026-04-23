"""Build ``stress_fix_report.md`` from a manifest + aniso validation correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def phi_diff_corr(dir_a: Path, dir_b: Path) -> float:
    za = np.load(dir_a / "final_state.npz")
    zb = np.load(dir_b / "final_state.npz")
    da = np.asarray(za["phi_m"] - za["phi_c"], dtype=np.float64).ravel()
    db = np.asarray(zb["phi_m"] - zb["phi_c"], dtype=np.float64).ravel()
    if da.shape != db.shape:
        return float("nan")
    return float(np.corrcoef(da, db)[0, 1])


def _fmt(x: float | str | None) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if x != x:
        return "nan"
    return f"{x:.6g}"


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="stress_fix_retest_<ts>/manifest.json",
    )
    ap.add_argument(
        "--aniso-dir",
        type=str,
        default="results/agate_ch/anisotropy_sweep_20260423_180434/aniso_10x",
    )
    ap.add_argument(
        "--batch",
        type=str,
        default="",
        help="Optional subtitle (e.g. stress_sign_validation).",
    )
    args = ap.parse_args(argv)

    manifest_path = root / args.manifest
    man = json.loads(manifest_path.read_text())

    rows: list[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
        ]
    ] = []
    val_uni_dir: Path | None = None

    for run in man.get("runs", []):
        rid = str(run["id"])
        od = root / run["out_dir"]
        summ_path = od / "summary.json"
        npz_path = od / "final_state.npz"
        if not summ_path.is_file():
            continue
        summ = json.loads(summ_path.read_text())
        mc = summ.get("mass_conservation") or {}
        mx_sum = mc.get("max_phi_sum", float("nan"))
        drift = mc.get("option_D_drift_pct", float("nan"))
        mb_pct = summ.get("mass_balance_percent", float("nan"))
        ovs = summ.get("overshoot_fraction_final", float("nan"))
        morph = str(summ.get("classification_at_final", summ.get("classification", "")))
        max_pm = max_pc = float("nan")
        if npz_path.is_file():
            z = np.load(npz_path)
            pm = np.asarray(z["phi_m"], dtype=np.float64)
            pc = np.asarray(z["phi_c"], dtype=np.float64)
            max_pm = float(np.max(pm))
            max_pc = float(np.max(pc))
        rows.append(
            (
                rid,
                _fmt(ovs),
                _fmt(mb_pct),
                _fmt(max_pm),
                _fmt(max_pc),
                _fmt(mx_sum),
                _fmt(drift),
                morph,
            )
        )
        if rid == "validation_uniform_uniaxial":
            val_uni_dir = od

    corr_txt = "n/a"
    if val_uni_dir is not None:
        aniso = root / args.aniso_dir
        if (aniso / "final_state.npz").is_file():
            c = phi_diff_corr(val_uni_dir, aniso)
            corr_txt = f"{c:.6f}"

    bt = man.get("batch") or args.batch
    subtitle = f" ({bt})" if bt else ""

    lines = [
        "# Stress coupling fix report" + subtitle,
        "",
        f"*Manifest:* `{manifest_path.relative_to(root)}`",
        "",
        "## Summary",
        "",
        "| config | overshoot % | Option B leak % | max φ_m | max φ_c | "
        "max φ_m+φ_c | Option D drift % | morphology |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")

    lines.extend(
        [
            "",
            f"**Correlation φ_m−φ_c (validation_uniform_uniaxial vs aniso_10x):** {corr_txt}",
            "",
            "## Diagnosis summary",
            "",
            "- **Was broken:** `stress_mu_hat` was evaluated on **φ_m** and **φ_c** "
            "separately with full strength to each μ, instead of on **ψ = φ_m − φ_c** "
            "with **±½** splitting; `use_stress` gated on **σ_xx** only, skipping pure shear.",
            "- **Fixed:** `stress_contribution_to_mu` applies **−B∇·(σ∇ψ)** with "
            "**ψ = φ_m − φ_c**, **δμ_m = +½ μ_stress**, **δμ_c = −½ μ_stress**; stress "
            "activates if **any** of **σ_xx, σ_yy, σ_xy** is nonzero.",
            "",
            "- **Uniform uniaxial sign:** `uniform_uniaxial_field` uses **σ_xx = +σ₀** "
            "so **κ_x,eff ≈ κ₀ + B σ₀** (positive σ₀ stiffens **x**-gradients), matching "
            "**aniso_10x**-style horizontal banding without driving **κ** negative.",
            "",
            "## Known limitations",
            "",
            "- Kirsch stress is not a consistent interior field for solid-filled cavity; "
            "treat kirsch runs as exploratory.",
            "- Uniform uniaxial stress vs κ_x≠κ_y anisotropy is analogous but not "
            "guaranteed pixel-identical (different operators).",
            "- Spatially varying **σ** with large **B|σ|** can still make **κ + B σ** "
            "locally negative — use smaller **σ₀** (e.g. Flamant / pressure-gradient at 0.1).",
            "",
        ]
    )

    out = root / "results" / "agate_ch" / "stress_fix_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
