#!/usr/bin/env python3
"""EXP-A1: band-pacing drill-down on existing S1 Ostwald-window sweep data.

Scientific context (S1 falsification of spacing ∝ γ^(-1/2))
-----------------------------------------------------------
S1 falsified spacing ∝ gamma^(-1/2): means 16.6/16.4/14.2 for gamma=2/4/8
(medians ~12 across), prediction required 23/16.4/11.6. Band-formation
periods are gamma-independent (first intervals 1225/1200/1175) and spacing
grows with front velocity within runs (d ≈ v·tau). tau is NOT simple
diffusive recharge (tau/x² drops 6→0.4 along a run). Composition is a
passive tracer (no feedback on c or phi_tot), so all gamma=4 runs share ONE
universal c_front(t) trajectory; each run's banding = that trajectory ∩ its
Ostwald window. Open question: what sets tau ≈ 0.8-1.2e3?
Candidate: directional-quench demixing at the moving front (cf. directional
spinodal decomposition; Liu & Regenauer-Lieb Zebra-rock CH modeling).

S1 snapshots archive φ_tot, c_front, and x_front at evolution_dt=50 but not
ψ(t,x)=φ_m−φ_c. Per-run ψ kymographs use final ψ revealed as the deposit
mask advances (valid when local CH demix completes within one saved frame).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_patterns.core.banding_metrics import (  # noqa: E402
    alternating_zone_metrics,
    band_formation_times,
    front_velocity_series,
    interpolate_series,
    record_from_slab_twophase_result,
)
from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.experiments.checkpoint import load_sim_result_from_run_dir  # noqa: E402
from continuous_patterns.experiments.s1_ostwald_window import analyze_s1_run  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_s1_outputs_module():
    spec = importlib.util.spec_from_file_location(
        "s1_outputs", _SCRIPTS_DIR / "slab_twophase_ostwald_window_outputs.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linreg_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    if xv.size < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(xv, yv)
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    if xv.size < 2:
        return float("nan"), float("nan"), float("nan")
    return _linreg_r2(np.log(xv), np.log(yv))


def _activity_window(
    snap_t: np.ndarray,
    snap_xf: np.ndarray,
    form_t: np.ndarray,
) -> tuple[float, float]:
    t = np.asarray(snap_t, dtype=np.float64).ravel()
    xf = np.asarray(snap_xf, dtype=np.float64).ravel()
    if t.size == 0:
        return 0.0, 1.0
    t0 = float(t[0])
    if form_t.size:
        t0 = max(t0, float(np.min(form_t)) - 200.0)
    if xf.size >= 2:
        dx = np.diff(xf)
        dt = np.diff(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = dx / np.maximum(dt, 1e-30)
        active = v > 1e-4
        if np.any(active):
            idx = np.where(active)[0]
            t0 = min(t0, float(t[max(int(idx[0]) - 1, 0)]))
    t1 = float(t[-1])
    if form_t.size:
        t1 = min(t1, float(np.max(form_t)) + 300.0)
    return t0, t1


def _build_psi_kymograph(
    solid_xt: np.ndarray,
    psi_final: np.ndarray,
    *,
    phi_gate: float = 0.05,
) -> np.ndarray:
    """Final ψ masked by evolving φ_tot (see module docstring)."""
    solid = np.asarray(solid_xt, dtype=np.float64)
    psi = np.asarray(psi_final, dtype=np.float64).ravel()
    out = np.full(solid.shape, np.nan, dtype=np.float64)
    mask = solid > phi_gate
    out[mask] = psi[None, :].repeat(solid.shape[0], axis=0)[mask]
    return out


def _load_run_bundle(run_dir: Path, run_label: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    cfg = load_run_config(run_dir / "config.yaml")
    result = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    rec = record_from_slab_twophase_result(result, run_label)
    phys = cfg.get("physics") or {}
    geom = cfg.get("geometry") or {}
    L = float(geom.get("L", 100.0))
    n = int(geom.get("n", rec.band_positions.size or 256))
    dx = L / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx

    snap_path = run_dir / "snapshots.npz"
    snap = np.load(snap_path)
    snap_t = np.asarray(snap["snapshots_t"], dtype=np.float64)
    solid_xt = np.asarray(snap["snapshots_solid"], dtype=np.float64)
    snap_cf = np.asarray(snap["snapshots_c_front"], dtype=np.float64)
    snap_xf = np.asarray(snap["snapshots_x_front"], dtype=np.float64)

    fin = np.load(run_dir / "final_state.npz")
    psi_final = np.asarray(fin["phi_m"], dtype=np.float64) - np.asarray(
        fin["phi_c"], dtype=np.float64
    )

    zone = alternating_zone_metrics(rec.band_positions, rec.band_widths, rec.band_labels)
    zone_pos = np.asarray(zone["zone_band_positions"], dtype=np.float64)
    form_t = band_formation_times(zone_pos, snap_t, snap_xf) if zone_pos.size else np.asarray([])

    return {
        "run_dir": run_dir,
        "cfg": cfg,
        "result": result,
        "rec": rec,
        "x": x,
        "L": L,
        "D_c": float(phys.get("D_c", 1.0)),
        "c_ostwald": float(phys.get("c_ostwald", float("nan"))),
        "w_ostwald": float(phys.get("w_ostwald", 0.05)),
        "gamma": float(phys.get("gamma", float("nan"))),
        "snap_t": snap_t,
        "solid_xt": solid_xt,
        "snap_cf": snap_cf,
        "snap_xf": snap_xf,
        "psi_final": psi_final,
        "zone": zone,
        "zone_pos": zone_pos,
        "form_t": form_t,
    }


def _extract_band_events(bundle: dict[str, Any], run_label: str) -> list[dict[str, Any]]:
    zone_pos = bundle["zone_pos"]
    form_t = bundle["form_t"]
    snap_t = bundle["snap_t"]
    snap_xf = bundle["snap_xf"]
    snap_cf = bundle["snap_cf"]
    D_c = bundle["D_c"]
    if zone_pos.size == 0:
        return []

    t_v, v_front = front_velocity_series(snap_t, snap_xf)
    events: list[dict[str, Any]] = []
    for i, (x_form, t_form) in enumerate(zip(zone_pos, form_t, strict=True)):
        v_at = (
            float(interpolate_series(np.asarray([t_form]), t_v, v_front)[0])
            if t_v.size
            else float("nan")
        )
        c_at = (
            float(interpolate_series(np.asarray([t_form]), snap_t, snap_cf)[0])
            if snap_t.size
            else float("nan")
        )
        if i == 0:
            d_n = float("nan")
            tau = float("nan")
        else:
            d_n = float(zone_pos[i] - zone_pos[i - 1])
            tau = float(form_t[i] - form_t[i - 1])
        tau_v = float(tau * v_at) if np.isfinite(tau) and np.isfinite(v_at) else float("nan")
        events.append(
            {
                "run_label": run_label,
                "gamma": bundle["gamma"],
                "c_ostwald": bundle["c_ostwald"],
                "t_form": float(t_form),
                "x_form": float(x_form),
                "d_n": d_n,
                "v_front": v_at,
                "tau": tau,
                "tau_v": tau_v,
                "x_form2_over_D": float(x_form**2 / D_c) if D_c > 0 else float("nan"),
                "c_front": c_at,
            }
        )
    return events


def _plot_psi_kymograph(
    out_path: Path,
    *,
    x: np.ndarray,
    snap_t: np.ndarray,
    psi_kymo: np.ndarray,
    form_t: np.ndarray,
    t_window: tuple[float, float],
    run_label: str,
) -> None:
    t0, t1 = t_window
    t_mask = (snap_t >= t0) & (snap_t <= t1)
    t_plot = snap_t[t_mask]
    z = psi_kymo[t_mask, :]
    if t_plot.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    vmax = float(np.nanmax(np.abs(z))) if np.any(np.isfinite(z)) else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(t_plot[0]), float(t_plot[-1])],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"$\psi=\phi_m-\phi_c$")
    for t_ev in form_t:
        if t0 <= t_ev <= t1:
            ax.axhline(t_ev, color="k", lw=0.6, alpha=0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"{run_label}: ψ kymograph (final ψ on deposit mask)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_cfront(
    out_path: Path,
    *,
    snap_t: np.ndarray,
    snap_cf: np.ndarray,
    form_t: np.ndarray,
    c_ost: float,
    w_ost: float,
    run_label: str,
) -> None:
    fig, (ax_main, ax_zoom) = plt.subplots(
        2, 1, figsize=(10.0, 6.5), gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_main.fill_between(
        snap_t, c_ost - w_ost, c_ost + w_ost, color="#ff7f0e", alpha=0.2, label="Ostwald window"
    )
    ax_main.plot(snap_t, snap_cf, lw=1.0, color="#1f77b4", label=r"$c_{\mathrm{front}}(t)$")
    for t_ev in form_t:
        ax_main.axvline(t_ev, color="0.4", lw=0.5, alpha=0.5)
    ax_main.set_ylabel(r"$c_{\mathrm{front}}$")
    ax_main.set_title(f"{run_label}: front concentration (dt=50 saved frames)")
    ax_main.legend(fontsize=8, loc="best")

    if form_t.size >= 3:
        i_mid = int(form_t.size // 2)
        t_a = float(form_t[i_mid - 1])
        t_b = float(form_t[i_mid + 1])
    elif form_t.size >= 2:
        t_a = float(form_t[0])
        t_b = float(form_t[1])
    else:
        t_a = float(snap_t[0])
        t_b = float(snap_t[min(5, snap_t.size - 1)])

    zoom_mask = (snap_t >= t_a) & (snap_t <= t_b)
    ax_zoom.fill_between(
        snap_t[zoom_mask],
        c_ost - w_ost,
        c_ost + w_ost,
        color="#ff7f0e",
        alpha=0.2,
    )
    ax_zoom.plot(snap_t[zoom_mask], snap_cf[zoom_mask], "o-", ms=3, lw=1.0, color="#1f77b4")
    ax_zoom.set_xlabel("t")
    ax_zoom.set_ylabel(r"$c_{\mathrm{front}}$")
    ax_zoom.set_title(f"zoom: t ∈ [{t_a:.0f}, {t_b:.0f}]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fmt_cell(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.4g}"


def _write_events_table(out_path: Path, events: list[dict[str, Any]]) -> None:
    header = (
        "| run_label | gamma | c_ostwald | t_form | x_form | d_n | v_front | "
        "tau | tau·v | x²/D | c_front |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header.rstrip("\n")]
    for ev in events:
        lines.append(
            f"| {ev['run_label']} | {ev['gamma']:.4g} | {ev['c_ostwald']:.4g} | "
            f"{ev['t_form']:.4g} | {ev['x_form']:.4g} | {_fmt_cell(ev['d_n'])} | "
            f"{ev['v_front']:.4g} | {_fmt_cell(ev['tau'])} | {_fmt_cell(ev['tau_v'])} | "
            f"{ev['x_form2_over_D']:.4g} | {ev['c_front']:.4g} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_figure_a1(
    out_path: Path,
    events: list[dict[str, Any]],
    psi_profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    fits: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))

    # (a) tau vs gamma — c_ost=0.25 events only
    ax = axes[0, 0]
    ev25 = [e for e in events if abs(float(e["c_ostwald"]) - 0.25) < 1e-9 and np.isfinite(e["tau"])]
    if ev25:
        g = np.array([e["gamma"] for e in ev25], dtype=np.float64)
        tau = np.array([e["tau"] for e in ev25], dtype=np.float64)
        ax.scatter(g, tau, s=22, alpha=0.75, color="#2ca02c")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"band period $\tau$")
    ax.set_title(r"(a) $\tau$ vs $\gamma$ ($c_{\mathrm{ostwald}}=0.25$ events)")

    # (b) tau vs v — log-log
    ax = axes[0, 1]
    ev_v = [e for e in events if np.isfinite(e["tau"]) and e["v_front"] > 0]
    if ev_v:
        v = np.array([e["v_front"] for e in ev_v], dtype=np.float64)
        tau = np.array([e["tau"] for e in ev_v], dtype=np.float64)
        ax.loglog(v, tau, "o", ms=4, alpha=0.65, color="#1f77b4")
        slope = fits.get("tau_vs_v", {}).get("slope")
        intercept = fits.get("tau_vs_v", {}).get("intercept")
        if slope is not None and intercept is not None and np.isfinite(slope):
            v_line = np.logspace(np.log10(v.min()), np.log10(v.max()), 64)
            ax.loglog(v_line, np.exp(intercept) * v_line**slope, "--", color="0.4", lw=1.0)
    ax.set_xlabel(r"$v_{\mathrm{front}}$")
    ax.set_ylabel(r"$\tau$")
    ax.set_title(r"(b) kinetic pacing: $\tau$ vs $v_{\mathrm{front}}$")

    # (c) tau vs x_form
    ax = axes[1, 0]
    ev_x = [e for e in events if np.isfinite(e["tau"]) and np.isfinite(e["x_form"])]
    if ev_x:
        xform = np.array([e["x_form"] for e in ev_x], dtype=np.float64)
        tau = np.array([e["tau"] for e in ev_x], dtype=np.float64)
        ax.scatter(xform, tau, s=22, alpha=0.65, color="#d62728")
        slope = fits.get("tau_vs_x", {}).get("slope")
        intercept = fits.get("tau_vs_x", {}).get("intercept")
        if slope is not None and intercept is not None and np.isfinite(slope):
            x_line = np.linspace(xform.min(), xform.max(), 64)
            ax.plot(x_line, intercept + slope * x_line, "--", color="0.4", lw=1.0)
    ax.set_xlabel(r"$x_{\mathrm{form}}$")
    ax.set_ylabel(r"$\tau$")
    ax.set_title(r"(c) geometric pacing: $\tau$ vs $x_{\mathrm{form}}$")

    # (d) final psi profiles at c_ost=0.25, aligned at x=0
    ax = axes[1, 1]
    colors = {2.0: "#9467bd", 4.0: "#ff7f0e", 8.0: "#8c564b"}
    for label, (x_prof, psi_prof) in sorted(
        psi_profiles.items(), key=lambda kv: float(str(kv[0]).split("_g")[-1])
    ):
        gamma = float(label.split("_g")[-1])
        ax.plot(x_prof, psi_prof, lw=1.2, color=colors.get(gamma, "#333333"), label=f"γ={gamma:g}")
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.set_xlabel("x (aligned at deposit origin)")
    ax.set_ylabel(r"$\psi(x)$ final")
    ax.set_title(r"(d) Final $\psi(x)$ at $c_{\mathrm{ostwald}}=0.25$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _refresh_manifest_entry(entry: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    for key in (
        "n_bands_zone",
        "band_zone_extent",
        "mean_spacing_zone",
        "median_spacing_zone",
        "mean_spacing_zone_excl_last",
        "std_spacing_zone",
        "mass_residual_pct",
        "clipped_mass_frac",
        "row_validity",
        "spacing_pacing",
    ):
        if key in analysis:
            out[key] = analysis[key]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-A1 band-pacing analysis on S1 sweep data")
    parser.add_argument("sweep_dir", type=Path, help="S1 sweep directory with manifest_axes.json")
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / "manifest_axes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("entries", []) if e.get("status") == "success"]
    if not entries:
        raise RuntimeError("no successful runs in manifest")

    all_events: list[dict[str, Any]] = []
    psi_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for entry in tqdm(entries, desc="A1 runs", unit="run"):
        run_label = str(entry["run_label"])
        rel = entry.get("relative_path")
        if not rel:
            continue
        run_dir = sweep_dir / str(rel)

        analysis = analyze_s1_run(run_dir, run_label=run_label)
        entry.update(_refresh_manifest_entry(entry, analysis))

        bundle = _load_run_bundle(run_dir, run_label)
        events = _extract_band_events(bundle, run_label)
        all_events.extend(events)

        t_window = _activity_window(bundle["snap_t"], bundle["snap_xf"], bundle["form_t"])
        psi_kymo = _build_psi_kymograph(bundle["solid_xt"], bundle["psi_final"])
        _plot_psi_kymograph(
            sweep_dir / f"kymo_psi_{run_label}.png",
            x=bundle["x"],
            snap_t=bundle["snap_t"],
            psi_kymo=psi_kymo,
            form_t=bundle["form_t"],
            t_window=t_window,
            run_label=run_label,
        )
        _plot_cfront(
            sweep_dir / f"cfront_{run_label}.png",
            snap_t=bundle["snap_t"],
            snap_cf=bundle["snap_cf"],
            form_t=bundle["form_t"],
            c_ost=bundle["c_ostwald"],
            w_ost=bundle["w_ostwald"],
            run_label=run_label,
        )

        if abs(bundle["c_ostwald"] - 0.25) < 1e-9:
            x0 = float(bundle["snap_xf"][0]) if bundle["snap_xf"].size else 0.0
            x_aligned = bundle["x"] - x0
            psi_profiles[run_label] = (x_aligned, bundle["psi_final"])

    # Fitted slopes
    tau_v_events = [e for e in all_events if np.isfinite(e["tau"]) and e["v_front"] > 0]
    tau_x_events = [e for e in all_events if np.isfinite(e["tau"]) and np.isfinite(e["x_form"])]
    d_v_events = [e for e in all_events if np.isfinite(e["d_n"]) and e["v_front"] > 0]

    tau_v_slope, tau_v_intercept, tau_v_r2 = _loglog_fit(
        np.array([e["v_front"] for e in tau_v_events], dtype=np.float64),
        np.array([e["tau"] for e in tau_v_events], dtype=np.float64),
    )
    tau_x_slope, tau_x_intercept, tau_x_r2 = _linreg_r2(
        np.array([e["x_form"] for e in tau_x_events], dtype=np.float64),
        np.array([e["tau"] for e in tau_x_events], dtype=np.float64),
    )
    d_v_slope, d_v_intercept, d_v_r2 = _loglog_fit(
        np.array([e["v_front"] for e in d_v_events], dtype=np.float64),
        np.array([e["d_n"] for e in d_v_events], dtype=np.float64),
    )

    fits = {
        "tau_vs_v": {"slope": tau_v_slope, "intercept": tau_v_intercept, "r2": tau_v_r2},
        "tau_vs_x": {"slope": tau_x_slope, "intercept": tau_x_intercept, "r2": tau_x_r2},
        "d_vs_v": {"slope": d_v_slope, "intercept": d_v_intercept, "r2": d_v_r2},
    }

    _write_events_table(sweep_dir / "table_a1_events.md", all_events)
    summary = {
        "n_events": len(all_events),
        "n_events_with_tau": len(tau_v_events),
        "fits": fits,
        "gate": {"mass_residual_pct_max": 1e-4, "clipped_mass_frac_max": 1e-6},
    }
    (sweep_dir / "summary_a1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_figure_a1(sweep_dir / "figure_a1_pacing.png", all_events, psi_profiles, fits)

    updated_by_label = {str(e["run_label"]): e for e in entries}
    manifest["entries"] = [
        updated_by_label.get(str(e.get("run_label")), e) if e.get("status") == "success" else e
        for e in manifest.get("entries", [])
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    s1_mod = _load_s1_outputs_module()
    s1_mod.write_s1_artifacts(sweep_dir, manifest)

    print(f"[A1] {len(all_events)} band events → {sweep_dir / 'table_a1_events.md'}", flush=True)
    print(f"[A1] summary → {sweep_dir / 'summary_a1.json'}", flush=True)
    print(f"[A1] figure → {sweep_dir / 'figure_a1_pacing.png'}", flush=True)
    for e in manifest["entries"]:
        if str(e.get("run_label")) == "S1_cost0p25_g2":
            print(f"[A1] gamma=2 validity → {e.get('row_validity')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
