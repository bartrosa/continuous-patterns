"""Aggregate stage2 γ-scan runs: labyrinth metrics + λ_peak vs λ_CH theory."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.agate_stage2.labyrinth_analysis import analyze_stage2_run
from continuous_patterns.plot_captions import figure_save_png_with_params


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _results_dir() -> Path:
    return _repo_root() / "results" / "agate_stage2"


def find_gamma_sweep_dirs() -> list[Path]:
    """Directories named ``stage2_gamma_<n>_0`` (see ``sweep_gamma.py``)."""
    base = _results_dir()
    if not base.is_dir():
        return []
    out = sorted(base.glob("stage2_gamma_*_0"), key=lambda p: p.name)
    return [p for p in out if p.is_dir()]


def gamma_from_summary(run_dir: Path) -> float | None:
    sp = run_dir / "summary.json"
    if not sp.is_file():
        return None
    d = json.loads(sp.read_text())
    p = d.get("parameters") or {}
    g = p.get("gamma")
    try:
        return float(g) if g is not None else None
    except (TypeError, ValueError):
        return None


def lambda_ch_theory(gamma: float, kappa: float) -> float:
    """Linear CH scale λ = 2π √(κ/γ)."""
    return float(2.0 * np.pi * np.sqrt(kappa / max(float(gamma), 1e-30)))


def main() -> None:
    import matplotlib.pyplot as plt

    dirs = find_gamma_sweep_dirs()
    base = _results_dir()
    base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for d in dirs:
        g = gamma_from_summary(d)
        if g is None:
            print(f"(skip) no gamma in {d}/summary.json", file=sys.stderr)
            continue
        summ_path = d / "summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.is_file() else {}
        params = summ.get("parameters") or {}
        kappa = float(params.get("kappa", 0.5))

        try:
            ana = analyze_stage2_run(d)
        except Exception as exc:
            print(f"(skip) analyze failed {d}: {exc}", file=sys.stderr)
            continue

        spec = ana.get("spectral_peak_wavelength") or {}
        ani = ana.get("directional_anisotropy") or {}
        lam_p = spec.get("lambda_peak")
        lam_t = lambda_ch_theory(g, kappa)

        rows.append(
            {
                "gamma": g,
                "kappa": kappa,
                "lambda_peak": lam_p,
                "lambda_ch_theory": lam_t,
                "k_peak": spec.get("k_peak"),
                "anisotropy_ratio": ani.get("anisotropy_ratio"),
                "contrast_amplitude": ana.get("contrast_amplitude"),
                "run_dir": d.name,
            }
        )

    rows.sort(key=lambda r: float(r["gamma"]))

    out_json = base / "gamma_scan_summary.json"
    out_csv = base / "gamma_scan_summary.csv"
    out_json.write_text(json.dumps(rows, indent=2))

    if rows:
        keys = list(rows[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_json}\nWrote {out_csv}")

        gs = np.array([float(r["gamma"]) for r in rows], dtype=np.float64)
        lp = np.array(
            [
                float(r["lambda_peak"])
                if r.get("lambda_peak") is not None and r["lambda_peak"] == r["lambda_peak"]
                else np.nan
                for r in rows
            ],
            dtype=np.float64,
        )
        lt = np.array([float(r["lambda_ch_theory"]) for r in rows], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(gs, lp, "o-", label=r"$\lambda_{\mathrm{peak}}$ (spectral)")
        ax.plot(gs, lt, "s--", label=r"$\lambda_{\mathrm{CH}} = 2\pi\sqrt{\kappa/\gamma}$")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\lambda$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = base / "gamma_scan_lambda_comparison.png"
        agg_cfg = {
            "script": "agate_stage2.aggregate_sweep",
            "results_base": str(base.resolve()),
            "points": [{k: v for k, v in r.items() if k != "run_dir"} for r in rows],
        }
        figure_save_png_with_params(fig, fig_path, agg_cfg, dpi=150)
        plt.close(fig)
        print(f"Wrote {fig_path}")
    else:
        print("No sweep directories found; wrote empty JSON.", file=sys.stderr)
        out_json.write_text("[]")


if __name__ == "__main__":
    main()
