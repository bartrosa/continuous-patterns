"""Inspect Option B surface flux samples (v1 bilinear circle + 2·dx stencil)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models import agate_ch
from continuous_patterns.models.agate_ch import _compute_surface_flux_balance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to run YAML (will re-simulate).",
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=Path("flux_inspect"))
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm step progress (default: off for readable logs).",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {args.config}...")
    cfg = load_run_config(args.config)

    print(
        f"Simulating {cfg['experiment']['name']} "
        f"(T={cfg['time']['T']}, n={cfg['geometry']['n']})..."
    )
    result = agate_ch.simulate(cfg, chunk_size=args.chunk_size, show_progress=args.show_progress)

    flux = result.meta["flux_samples"]
    times = np.asarray(flux["times"])
    m_diss = np.asarray(flux["M_dissolved"])
    f_rate = np.asarray(flux["flux_rate"])
    phi_pack = np.asarray(flux["phi_pack_rfix"])
    c_in = np.asarray(flux.get("c_in_circle", []))
    c_out = np.asarray(flux.get("c_out_circle", []))

    budget = _compute_surface_flux_balance(flux)
    diag = result.diagnostics.get("surface_flux_balance")
    R = float(cfg["geometry"]["R"])
    rff = float(result.meta.get("option_b_r_fix_frac", 0.75))
    r_fix = rff * R

    print("\n=== Option B surface flux balance (v1 method) ===\n")
    print(f"r_fix:               {r_fix:.6f}")
    print(f"front_threshold:     {budget.get('front_threshold', float('nan'))}")
    print(f"front_arrival_idx:   {budget.get('front_arrival_idx', -1)}")
    print(f"front_arrival_t:     {budget.get('front_arrival_t', float('nan'))}")
    print(f"front_reached:       {budget.get('front_reached', False)}")
    print(f"leak_pct:            {budget.get('leak_pct', float('nan')):.4f}%")
    print(f"dissolved_change:    {budget.get('dissolved_change', float('nan'))}")
    print(f"flux_integrated:     {budget.get('flux_integrated', float('nan'))}")
    print(f"residual:            {budget.get('residual', float('nan'))}")
    print(f"n_pre_front_samples: {budget.get('n_samples', 0)}")
    if isinstance(diag, dict):
        print("\n--- simulate() diagnostics['surface_flux_balance'] ---")
        for k in (
            "leak_pct",
            "dissolved_change",
            "flux_integrated",
            "residual",
            "n_samples",
            "front_arrival_t",
        ):
            if k in diag:
                print(f"  {k}: {diag[k]}")

    thr = float(budget.get("front_threshold", 0.3))
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(times, phi_pack, color="C0", label=r"$\phi_m+\phi_c$ at $r_{\mathrm{fix}}$")
    axes[0].axhline(thr, color="k", linestyle="--", linewidth=0.8, label=f"threshold={thr}")
    if budget.get("front_reached") and np.isfinite(budget.get("front_arrival_t", float("nan"))):
        axes[0].axvline(
            float(budget["front_arrival_t"]),
            color="C3",
            linestyle=":",
            label="front arrival",
        )
    axes[0].set_ylabel(r"$\phi$ pack")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    if c_in.size == times.size and c_out.size == times.size:
        axes[1].plot(times, c_in, label=r"$c$ at $r_{\mathrm{fix}}-\Delta x$")
        axes[1].plot(times, c_out, label=r"$c$ at $r_{\mathrm{fix}}+\Delta x$")
    axes[1].set_ylabel(r"$c$ circle means")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, m_diss, color="C2")
    axes[2].set_ylabel(r"$M_{\mathrm{dissolved}}$ ($r<r_{\mathrm{fix}}$)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(times, f_rate, color="C4", label=r"$F(t)$")
    if times.size >= 2:
        cumtrap = np.zeros_like(times)
        for i in range(1, times.size):
            cumtrap[i] = cumtrap[i - 1] + 0.5 * (f_rate[i] + f_rate[i - 1]) * (
                times[i] - times[i - 1]
            )
        axb = axes[3].twinx()
        axb.plot(times, cumtrap, color="C5", linestyle="--", label=r"$\int F\,\mathrm{d}t$")
        axb.set_ylabel(r"cumulative $\int F\,\mathrm{d}t$")
        axb.legend(loc="upper right", fontsize=8)
    axes[3].set_ylabel("flux rate")
    axes[3].set_xlabel("t")
    axes[3].legend(loc="upper left", fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"Surface flux samples — {cfg['experiment']['name']} "
        f"(leak {budget.get('leak_pct', float('nan')):+.3f}%)"
    )
    fig.tight_layout()
    fig_path = args.out_dir / "flux_samples_inspection.png"
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)
    print(f"\nFigure: {fig_path}")


if __name__ == "__main__":
    main()
