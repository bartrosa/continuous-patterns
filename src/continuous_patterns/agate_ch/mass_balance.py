"""Mass balance diagnostics: fixed-radius surface flux vs dissolved-disk budget."""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from continuous_patterns.agate_ch.diagnostics import azimuthal_mean_at_radius_numpy


def silica_mass_disk_numpy(
    c: np.ndarray,
    pm: np.ndarray,
    pc: np.ndarray,
    *,
    L: float,
    r_disk: float,
    rho_m: float,
    rho_c: float,
) -> float:
    n = int(c.shape[0])
    dx = L / n
    xc = L / 2.0
    xs = (np.arange(n, dtype=np.float64) + 0.5) * dx
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    rv = np.sqrt((xv - xc) ** 2 + (yv - xc) ** 2)
    mask = rv < r_disk
    tot = np.asarray(c) + rho_m * np.asarray(pm) + rho_c * np.asarray(pc)
    return float(np.sum(tot * mask) * dx**2)


def dissolved_mass_disk_numpy(c: np.ndarray, *, L: float, r_disk: float) -> float:
    """∫ c dA over disk r < r_disk (matches Fick flux of dissolved silica)."""
    n = int(c.shape[0])
    dx = L / n
    xc = L / 2.0
    xs = (np.arange(n, dtype=np.float64) + 0.5) * dx
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    rv = np.sqrt((xv - xc) ** 2 + (yv - xc) ** 2)
    mask = rv < r_disk
    return float(np.sum(np.asarray(c) * mask) * dx**2)


def _surface_flux_budget_from_snapshots(
    snaps_full: list[tuple[int, Any, Any, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Sparse snapshot-based Option B (legacy)."""
    if not snaps_full:
        return {
            "error": "no_snapshots",
            "leak_pct": float("nan"),
            "flux_integrated_to_stop": float("nan"),
        }
    L = float(cfg["L"])
    R_geom = float(cfg["R"])
    n = int(cfg["grid"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    D_c = float(cfg["D_c"])
    dx = L / n

    if "mass_balance_r_measure_fixed" in cfg:
        r_fixed = float(cfg["mass_balance_r_measure_fixed"])
    elif "option_B_r_measure_fixed" in cfg:
        r_fixed = float(cfg["option_B_r_measure_fixed"])
    else:
        frac = float(
            cfg.get(
                "mass_balance_r_measure_fixed_fraction",
                cfg.get("option_B_r_measure_fixed_fraction", 0.75),
            )
        )
        r_fixed = frac * R_geom

    phi_thr = float(
        cfg.get(
            "mass_balance_front_phi_threshold",
            cfg.get("option_B_front_phi_threshold", 0.3),
        )
    )

    snaps_sorted = sorted(snaps_full, key=lambda x: int(x[0]))

    times_flux: list[float] = []
    flux_rates: list[float] = []
    silica_inside_initial = float("nan")
    silica_inside_at_stop = float("nan")
    front_reached_r_measure = False
    t_stop = float("nan")

    for step, c, pm, pc in snaps_sorted:
        c_np = np.asarray(c)
        pm_np = np.asarray(pm)
        pc_np = np.asarray(pc)
        phi_tot = pm_np + pc_np

        si_diss = dissolved_mass_disk_numpy(c_np, L=L, r_disk=r_fixed)
        if np.isnan(silica_inside_initial):
            silica_inside_initial = si_diss

        if not front_reached_r_measure:
            r_out = r_fixed + dx
            r_in = max(r_fixed - dx, 1e-6)
            c_o = azimuthal_mean_at_radius_numpy(c_np, L=L, r_abs=r_out)
            c_i = azimuthal_mean_at_radius_numpy(c_np, L=L, r_abs=r_in)
            dc_dr = (c_o - c_i) / max(2.0 * dx, 1e-12)
            perimeter = 2.0 * np.pi * r_fixed
            flux_rates.append(float(D_c * dc_dr * perimeter))
            times_flux.append(float(step) * dt)

            phi_mean = azimuthal_mean_at_radius_numpy(phi_tot, L=L, r_abs=r_fixed)
            if phi_mean > phi_thr:
                front_reached_r_measure = True
                silica_inside_at_stop = si_diss
                t_stop = float(step) * dt
                break

    if not front_reached_r_measure and snaps_sorted:
        _st, c, pm, pc = snaps_sorted[-1]
        silica_inside_at_stop = dissolved_mass_disk_numpy(
            np.asarray(c),
            L=L,
            r_disk=r_fixed,
        )
        t_stop = float(int(_st)) * dt

    if len(flux_rates) >= 2:
        t_arr = np.array(times_flux, dtype=np.float64)
        f_arr = np.array(flux_rates, dtype=np.float64)
        flux_integrated_to_stop = float(np.trapezoid(f_arr, t_arr))
    elif len(flux_rates) == 1:
        flux_integrated_to_stop = float(flux_rates[0]) * min(t_stop, T)
    else:
        flux_integrated_to_stop = 0.0

    delta_si = silica_inside_at_stop - silica_inside_initial
    residual_b = delta_si - flux_integrated_to_stop
    denom_b = float(
        np.max(
            np.abs(
                np.array(
                    [
                        silica_inside_initial,
                        silica_inside_at_stop,
                        flux_integrated_to_stop,
                    ],
                    dtype=np.float64,
                )
            )
        )
    )
    leak_pct_b = 100.0 * residual_b / max(denom_b, 1e-30)

    return {
        "r_measure_fixed": r_fixed,
        "silica_inside_initial": silica_inside_initial,
        "silica_inside_at_stop": silica_inside_at_stop,
        "silica_inside_change": delta_si,
        "dissolved_disk_initial": silica_inside_initial,
        "dissolved_disk_at_stop": silica_inside_at_stop,
        "dissolved_change": delta_si,
        "t_stop": t_stop,
        "flux_integrated_to_stop": flux_integrated_to_stop,
        "residual": residual_b,
        "leak_pct": leak_pct_b,
        "front_reached_r_measure": front_reached_r_measure,
        "times_valid": times_flux,
        "flux_rates_valid": flux_rates,
        "budget_is_dissolved_c_only": True,
        "budget_source": "snapshots_sparse",
        "used_dense_flux_sampling": False,
    }


def compute_surface_flux_budget(
    snaps_full: list[tuple[int, Any, Any, Any]],
    cfg: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve dense flux budget from ``meta`` or fall back to snapshot iteration.

    Prefer ``meta['mass_balance_surface_flux']`` populated by ``integrate_chunks``.
    """
    if meta is not None and meta.get("mass_balance_surface_flux"):
        return meta["mass_balance_surface_flux"]

    warnings.warn(
        "Option B fallback: sparse snapshot-based flux integration. "
        "Run with integrate_chunks flux_sample_dt (default 2.0) for dense sampling.",
        stacklevel=2,
    )
    return _surface_flux_budget_from_snapshots(snaps_full, cfg)


def plot_mass_balance_comparison(
    *,
    meta: dict[str, Any],
    surface_flux_budget: dict[str, Any],
    path: Any,
) -> None:
    """Diagnostic figure for smoke / QA."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    dt_b = np.asarray(surface_flux_budget.get("times_valid") or [], dtype=np.float64)
    fr_b = np.asarray(
        surface_flux_budget.get("flux_rates_valid") or [], dtype=np.float64
    )
    ax_a = axes[0, 0]
    if dt_b.size:
        ax_a.plot(dt_b, fr_b, color="tab:blue", lw=1.2)
        ts = surface_flux_budget.get("t_stop")
        if ts == ts and surface_flux_budget.get("front_reached_r_measure"):
            ax_a.axvline(float(ts), color="red", ls="--", label="front at r_fixed")
            ax_a.legend(fontsize=8)
    ax_a.set_title("Flux rate (fixed measurement radius)")
    ax_a.set_xlabel("time")
    ax_a.set_ylabel("flux rate")

    ax_b = axes[0, 1]
    r_fix = surface_flux_budget.get("r_measure_fixed")
    if dt_b.size and r_fix == r_fix:
        ax_b.axhline(float(r_fix), color="tab:green", lw=1.2)
    ax_b.set_title("Measurement radius")
    ax_b.set_xlabel("time")
    ax_b.set_ylabel("r")

    si0 = float(meta.get("silica_full_domain_initial", float("nan")))
    si1 = float(meta.get("silica_full_domain_final", float("nan")))
    ax_c = axes[1, 0]
    ax_c.axis("off")
    ax_c.set_title("Full grid silica total (reference)")
    ax_c.text(
        0.05,
        0.55,
        f"initial (full grid): {si0:.4f}\nfinal (full grid):   {si1:.4f}",
        transform=ax_c.transAxes,
        fontsize=10,
        family="monospace",
        va="top",
    )

    ax_d = axes[1, 1]
    direct_delta = si1 - si0 if si0 == si0 and si1 == si1 else float("nan")
    flux_b = float(surface_flux_budget.get("flux_integrated_to_stop", float("nan")))
    labs = [
        "Δ silica\n(full domain)",
        "Flux ∫dt\n(fixed r)",
    ]
    vals = [direct_delta, flux_b]
    colors = ["tab:gray", "tab:blue"]
    xpos = np.arange(len(vals))
    ax_d.bar(xpos, vals, color=colors)
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels(labs, fontsize=8)
    ax_d.set_title("Budget comparison")
    denom_annot = max(abs(direct_delta), 1e-30)
    for i, v in enumerate(vals):
        if v == v:
            ax_d.annotate(
                f"{v:.2f}",
                xy=(i, v),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    rel_lines: list[str] = []
    if direct_delta == direct_delta and flux_b == flux_b:
        rel_lines.append(
            f"flux vs Δdirect: {100.0 * (flux_b - direct_delta) / denom_annot:+.2f}%"
        )
    if rel_lines:
        ax_d.text(
            0.02,
            0.98,
            "\n".join(rel_lines),
            transform=ax_d.transAxes,
            va="top",
            fontsize=7,
            color="0.35",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def print_mass_balance_smoke_stdout(
    *,
    label: str,
    meta: dict[str, Any],
    surface_flux_budget: dict[str, Any],
    spectral_kernel_check: dict[str, Any] | None = None,
) -> None:
    """Formatted stdout block for smoke / CI.

    ``spectral_kernel_check`` — short periodic run (no Dirichlet / no χ projection):
    keys ``total_mass_initial``, ``total_mass_final``, ``leak_pct``, optional
    ``grid``, ``T``, ``dt``, ``n_steps``.
    """
    si0 = float(meta.get("silica_full_domain_initial", float("nan")))
    si1 = float(meta.get("silica_full_domain_final", float("nan")))
    delta = si1 - si0

    def _band_transport(pct: float) -> str:
        if pct != pct:
            return "UNKNOWN"
        ap = abs(pct)
        if ap < 1.0:
            return "EXCELLENT <1%"
        if ap < 5.0:
            return "ACCEPTABLE <5%"
        return "PROBLEMATIC >5%"

    def _band_spectral(pct: float) -> str:
        if pct != pct:
            return "UNKNOWN"
        ap = abs(pct)
        if ap < 0.1:
            return "PASS (<0.1% drift)"
        return "FAIL (kernel leak)"

    lb = float(surface_flux_budget.get("leak_pct", float("nan")))

    print("")
    print(f"=== MASS BALANCE ({label}) ===")
    print("")
    print("FULL GRID (reference totals):")
    print(f"  silica initial:         {si0:.6f}")
    print(f"  silica final:           {si1:.6f}")
    print(f"  delta:                  {delta:.6f}")
    print("")
    print("SURFACE FLUX vs DISSOLVED DISK (transport budget — physics):")
    rmfix = surface_flux_budget.get("r_measure_fixed", float("nan"))
    print(f"  r_measure_fixed:          {rmfix}")
    freached = surface_flux_budget.get("front_reached_r_measure", False)
    ts = surface_flux_budget.get("t_stop", float("nan"))
    print(f"  front_reached_r_measure:   {freached} at t={ts}")
    sii = float(surface_flux_budget.get("silica_inside_initial", float("nan")))
    sia = float(surface_flux_budget.get("silica_inside_at_stop", float("nan")))
    print(f"  dissolved disk initial:    {sii:.6f}")
    print(f"  dissolved disk at stop:    {sia:.6f}")
    fib = float(surface_flux_budget.get("flux_integrated_to_stop", float("nan")))
    sic = float(surface_flux_budget.get("silica_inside_change", float("nan")))
    print(f"  flux_integrated_to_stop: {fib:.6f}")
    print(f"  dissolved change:        {sic:.6f}")
    print(f"  leak_pct:                {lb:.6f}%")
    print(f"  → {_band_transport(lb)}")
    print("")
    if spectral_kernel_check:
        sk = spectral_kernel_check
        mi = float(sk.get("total_mass_initial", float("nan")))
        mf = float(sk.get("total_mass_final", float("nan")))
        lk = float(sk.get("leak_pct", float("nan")))
        g_ = sk.get("grid", "")
        t_ = sk.get("T", "")
        dt_ = sk.get("dt", "")
        ns = sk.get("n_steps", "")
        print("PERIODIC SPECTRAL KERNEL (numerics — no BC, blob IC, full-grid mass):")
        if g_ != "":
            print(f"  grid: {g_}")
        if t_ != "" and dt_ != "":
            print(f"  horizon: T={t_}, dt={dt_}, steps={ns}")
        print(f"  total_mass_initial: {mi:.9f}")
        print(f"  total_mass_final:   {mf:.9f}")
        print(f"  leak_pct:           {lk:.9f}%   [expect |·| ≪ 0.1%]")
        print(f"  → {_band_spectral(lk)}")
        print("")
