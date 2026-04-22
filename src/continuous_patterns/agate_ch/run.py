"""CLI entry; default config ``configs/agate_ch/baseline.yaml``."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
import yaml

from continuous_patterns.agate_ch.diagnostics import (
    analyse_all_snapshots,
    band_metrics,
    labyrinth_heuristic,
    moganite_chalcedony_anticorr,
    overshoot_fraction,
    physical_mass_balance_from_snapshots,
    radial_profile,
    total_silica_numpy,
)
from continuous_patterns.agate_ch.mass_balance import compute_surface_flux_budget
from continuous_patterns.agate_ch.plotting import (
    choose_pub_field,
    plot_band_count_evolution,
    plot_canonical_slice,
    plot_canonical_slice_grid,
    plot_comparison_grid,
    plot_fields_final,
    plot_gamma_phase_diagram,
    plot_gamma_scan_fields,
    plot_jablczynski,
    plot_kymograph,
    plot_radial,
    plot_sweep_compare,
    plot_sweep_kymographs,
    save_final_pub,
    write_animation,
    write_evolution_gif_phi_m,
)
from continuous_patterns.agate_ch.publication import (
    generate_paper_figures,
    write_results_markdown,
)
from continuous_patterns.agate_ch.solver import (
    cfg_to_sim_params,
    initial_state,
    simulate_to_host,
    spectral_mass_conservation_diagnostic,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def agate_ch_results_dir(root: Path | None = None) -> Path:
    """Where Agate CH CLI writes runs: ``<repo>/results/agate_ch``."""
    return (root if root is not None else _repo_root()) / "results" / "agate_ch"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def merge_cfg(base: dict, overlay: dict) -> dict:
    out = dict(base)
    out.update(overlay)
    return out


SnapList = list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def load_snapshots_h5(path: Path) -> SnapList:
    out: SnapList = []
    with h5py.File(path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            step = int(k.split("_")[1])
            g = h5[k]
            out.append(
                (
                    step,
                    np.asarray(g["c"]),
                    np.asarray(g["phi_m"]),
                    np.asarray(g["phi_c"]),
                )
            )
    return out


def _csv_val(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x)


def sweep_summary_row(summ: dict[str, Any]) -> dict[str, str]:
    mf = summ.get("metrics_at_final") or {}
    cvq = mf.get("cv_q", float("nan"))
    cvd = mf.get("cv_d_spacings", float("nan"))
    cvq_pct = (cvq * 100.0) if cvq == cvq else float("nan")
    cvd_pct = (cvd * 100.0) if cvd == cvd else float("nan")
    return {
        "config_id": _csv_val(summ.get("label")),
        "N_peak": _csv_val(summ.get("peak_band_count")),
        "N_final": _csv_val(summ.get("final_band_count", summ.get("N_b"))),
        "persist": _csv_val(summ.get("bands_persist")),
        "peak_t": _csv_val(summ.get("peak_band_count_time")),
        "mean_q": _csv_val(mf.get("mean_q")),
        "CV_q_pct": _csv_val(cvq_pct if cvq_pct == cvq_pct else ""),
        "CV_d_pct": _csv_val(cvd_pct if cvd_pct == cvd_pct else ""),
        "spearman_rho": _csv_val(mf.get("spearman_d_vs_index")),
        "classification": _csv_val(
            summ.get("classification_at_final", mf.get("classification"))
        ),
        "anticorrelation": _csv_val(summ.get("moganite_chalcedony_anticorrelation")),
        "overshoot_pct": _csv_val(summ.get("overshoot_fraction_final")),
        "mass_balance_pct": _csv_val(summ.get("mass_balance_percent")),
        "wall_seconds": _csv_val(summ.get("wall_seconds")),
    }


def seed_robustness_classification(
    results_per_seed: list[dict[str, Any]],
) -> str:
    """ROBUST / QUALITATIVELY-ROBUST / SEED-SENSITIVE (seed suite rule)."""
    N_vals = [
        float(r["N_final"]) for r in results_per_seed if r.get("N_final") is not None
    ]
    if not N_vals:
        return "SEED-SENSITIVE"
    classes = [str(r.get("classification", "")) for r in results_per_seed]
    anticorrs = []
    for r in results_per_seed:
        a = r.get("anticorrelation")
        if a is not None and float(a) == float(a):
            anticorrs.append(float(a))
    N_mean = float(np.mean(N_vals))
    N_cv = float(np.std(N_vals) / N_mean) if N_mean > 1e-9 else 1.0
    class_same = len(set(classes)) == 1
    ac_spread = max(anticorrs) - min(anticorrs) if len(anticorrs) >= 2 else 0.0
    if class_same and ac_spread < 0.05 and N_cv < 0.25:
        return "ROBUST"
    if class_same and ac_spread < 0.10:
        return "QUALITATIVELY-ROBUST"
    return "SEED-SENSITIVE"


def report_main_sweep(
    summaries: dict[str, dict[str, Any]],
    *,
    sweep_dir: Path,
    ids_order: list[str],
    total_wall_s: float,
) -> None:
    """Stdout: Part I physical mass balance + seeds + Q1–Q3."""

    def _lbl(cid: str) -> tuple[Any, Any, Any]:
        s = summaries.get(cid, {})
        mf = s.get("metrics_at_final") or {}
        return (
            s.get("final_band_count"),
            mf.get("classification") or s.get("classification_at_final"),
            s.get("moganite_chalcedony_anticorrelation"),
        )

    lines: list[str] = []
    lines.append("")
    lines.append("=== AGATE CH — MAIN SWEEP COMPLETE ===")
    lines.append(f"Total wall-clock: {fmt_hms(total_wall_s)}")
    lines.append("")
    lines.append("PART I — physical mass balance |residual| %:")
    mb_list: list[float] = []
    for cid in ids_order:
        mbp = summaries.get(cid, {}).get("mass_balance_percent_physical")
        if mbp is None:
            mbp = summaries.get(cid, {}).get("mass_balance_percent")
        v = float(mbp) if mbp is not None and mbp == mbp else float("nan")
        if v == v:
            mb_list.append(abs(v))
        sv = f"{v:.4f}" if v == v else "nan"
        lines.append(f"  {cid + ':':<28} {sv:>10}%")
    if mb_list:
        lines.append(f"  Max |residual|:{'':15} {max(mb_list):.4f}%  [pass if <5%]")
    lines.append("")
    lines.append("MASS CONSERVATION:")
    mb_phys_vals: list[float] = []
    mb_spec_vals: list[float] = []
    for cid in ids_order:
        s = summaries.get(cid, {})
        mbp = s.get("mass_balance_percent_physical")
        if mbp is None:
            mbp = s.get("mass_balance_percent")
        vb = float(mbp) if mbp is not None and mbp == mbp else float("nan")
        if vb == vb:
            mb_phys_vals.append(abs(vb))
        sm = s.get("spectral_mass_conservation") or s.get(
            "option_D_spectral_conservation"
        )
        leak_s = (
            sm.get("leak_pct", float("nan")) if isinstance(sm, dict) else float("nan")
        )
        vs = float(leak_s)
        if vs == vs:
            mb_spec_vals.append(abs(vs))
    max_phys = max(mb_phys_vals) if mb_phys_vals else float("nan")
    max_spec = max(mb_spec_vals) if mb_spec_vals else float("nan")
    phys_s = f"{max_phys:.4f}" if max_phys == max_phys else "nan"
    spec_s = f"{max_spec:.6f}" if max_spec == max_spec else "nan"
    lines.append(
        f"  Physical flux residual (wall gradient): {phys_s}%    [pass if <5%]"
    )
    lines.append(
        f"  Spectral kernel mass drift (periodic check): {spec_s}% [pass if <0.1%]"
    )
    lines.append("")
    pass_phys = max_phys == max_phys and max_phys < 5.0
    pass_spec = max_spec == max_spec and max_spec < 0.1
    if pass_phys and pass_spec:
        lines.append("Mass conservation checks passed.")
    elif max_phys == max_phys or max_spec == max_spec:
        lines.append(
            "Mass conservation: "
            + ("physical OK, " if pass_phys else "physical FAIL, ")
            + ("spectral OK." if pass_spec else "spectral FAIL.")
        )
    lines.append("")
    lines.append("OVERSHOOT CONTROL:")
    for cid in ids_order:
        ov = summaries.get(cid, {}).get("overshoot_fraction_final", float("nan"))
        ov_s = f"{float(ov):.2f}" if ov == ov else "nan"
        lines.append(f"  {cid + ':':<28} {ov_s:>6}%")
    lines.append("")
    lines.append("BAND COUNTS (final):")
    for cid in ids_order:
        nn = summaries.get(cid, {}).get("final_band_count")
        lines.append(f"  {cid + ':':<28} {nn}")
    lines.append("")
    lines.append("SEED ROBUSTNESS (new rule, medium_pinning × 3 seeds):")
    n42, c42, a42 = _lbl("medium_pinning")
    n123, c123, a123 = _lbl("medium_pinning_seed2")
    n999, c999, a999 = _lbl("medium_pinning_seed3")
    seed_results = [
        {"N_final": n42, "classification": c42, "anticorrelation": a42},
        {"N_final": n123, "classification": c123, "anticorrelation": a123},
        {"N_final": n999, "classification": c999, "anticorrelation": a999},
    ]
    ac_spread = (
        max(float(a42), float(a123), float(a999))
        - min(float(a42), float(a123), float(a999))
        if all(x is not None and x == x for x in (a42, a123, a999))
        else float("nan")
    )
    lines.append(f"  medium_pinning: N={n42}, class={c42}, anticorr={a42}")
    lines.append(f"  seed2:           N={n123}, class={c123}, anticorr={a123}")
    lines.append(f"  seed3:           N={n999}, class={c999}, anticorr={a999}")
    if ac_spread == ac_spread:
        lines.append(f"  anticorr spread: {ac_spread:.4f}")
    else:
        lines.append("  anticorr spread: nan")
    flag = seed_robustness_classification(seed_results)
    lines.append(f"  → {flag}")
    lines.append("")
    lines.append("KEY QUESTION ANSWERS:")
    nnp, _, _ = _lbl("no_pinning")
    nmp, _, _ = _lbl("medium_pinning")
    mf_np = summaries.get("no_pinning", {}).get("metrics_at_final") or {}
    mf_mp = summaries.get("medium_pinning", {}).get("metrics_at_final") or {}
    mf_ro = summaries.get("ratchet_only", {}).get("metrics_at_final") or {}
    cv_np = mf_np.get("cv_q")
    cv_mp = mf_mp.get("cv_q")
    cv_ro = mf_ro.get("cv_q")
    cv_np_pct = (cv_np * 100) if cv_np == cv_np else float("nan")
    cv_mp_pct = (cv_mp * 100) if cv_mp == cv_mp else float("nan")
    cv_ro_pct = (cv_ro * 100) if cv_ro == cv_ro else float("nan")
    lines.append("  Q1 (does pinning matter?):")
    lines.append(f"      no_pinning: N={nnp}, CV_q={cv_np_pct:.1f}%")
    lines.append(f"      medium_pinning: N={nmp}, CV_q={cv_mp_pct:.1f}%")
    q1_yes = cv_mp_pct == cv_mp_pct and cv_np_pct == cv_np_pct and cv_mp_pct < cv_np_pct
    lines.append(
        "  → "
        + (
            "YES, pinning increases regularity"
            if q1_yes
            else "NO, pinning has little effect (by CV_q)"
        )
    )
    cls_np = mf_np.get("classification")
    lines.append("  Q2 (is overshoot fix alone sufficient?):")
    lines.append(f"      no_pinning classification: {cls_np}")
    ok_bands = str(cls_np) not in ("INSUFFICIENT BANDS", "")
    lines.append(
        "  → "
        + (
            "YES, barrier_fix alone gives bands"
            if ok_bands
            else "NO, bands require pinning"
        )
    )
    nro, _, _ = _lbl("ratchet_only")
    lines.append("  Q3 (isolated ratchet effect):")
    lines.append("      no_pinning vs ratchet_only:")
    lines.append(f"      N: {nnp} vs {nro}, CV_q: {cv_np_pct:.1f}% vs {cv_ro_pct:.1f}%")
    q3_eff = cv_ro_pct == cv_ro_pct and cv_np_pct == cv_np_pct and cv_ro_pct < cv_np_pct
    lines.append(
        "  → "
        + (
            "ratchet alone increases regularity"
            if q3_eff
            else "ratchet alone has no clear effect (by CV_q)"
        )
    )
    lines.append("")
    lines.append(f"Outputs: {sweep_dir}")
    print("\n".join(lines))


def report_gamma_scan(
    gamma_rows: list[dict[str, Any]],
    *,
    sweep_dir: Path,
    total_wall_s: float,
) -> None:
    """Stdout after γ scan."""
    lines: list[str] = []
    lines.append("")
    lines.append("=== AGATE CH — GAMMA SCAN COMPLETE ===")
    lines.append(f"Total wall-clock: {fmt_hms(total_wall_s)}")
    lines.append("")
    lines.append("PART II — gamma scan:")
    for r in sorted(gamma_rows, key=lambda x: x["gamma"]):
        lines.append(
            f"  gamma={r['gamma']:.1f}: N={r['N_bands']}, "
            f"CV_q={r['CV_q_pct']:.1f}%, class={r.get('classification', '')}"
        )
    cv_vals = [(r["gamma"], r["CV_q_pct"]) for r in gamma_rows]
    cv_vals.sort(key=lambda z: z[0])
    reg_gamma = None
    for g, cv in cv_vals:
        if cv < 50.0:
            reg_gamma = g
            break
    lab_gamma = None
    for r in sorted(gamma_rows, key=lambda x: x["gamma"]):
        if r.get("labyrinth"):
            lab_gamma = r["gamma"]
            break
    lines.append("")
    lines.append(f"  Transition toward regular regime (CV_q<50%): gamma={reg_gamma}")
    lines.append(f"  Labyrinth detected at gamma≥: {lab_gamma}")
    lines.append("")
    lines.append(f"Outputs: {sweep_dir}")
    print("\n".join(lines))


def write_sweep_summaries_csv_txt(rows: list[dict[str, str]], sweep_dir: Path) -> None:
    cols = [
        "config_id",
        "N_peak",
        "N_final",
        "persist",
        "peak_t",
        "mean_q",
        "CV_q_pct",
        "CV_d_pct",
        "spearman_rho",
        "classification",
        "anticorrelation",
        "overshoot_pct",
        "mass_balance_pct",
        "wall_seconds",
    ]
    csv_path = sweep_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    txt_path = sweep_dir / "sweep_summary.txt"
    widths = {
        c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
        for c in cols
    }
    lines = []
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    lines.append(header)
    lines.append("-|-".join("-" * widths[c] for c in cols))
    for r in rows:
        lines.append(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    txt_path.write_text("\n".join(lines) + "\n")


def mass_balance_percent_from_meta(meta: dict[str, Any]) -> float:
    """Prefer physical flux residual; fall back to direct bookkeeping."""
    mp = meta.get("mass_balance_percent_physical")
    if mp is not None:
        return float(mp)
    si = meta.get("silica_initial")
    sf = meta.get("silica_final")
    fl = meta.get("cumulative_boundary_flux_mass")
    try:
        if si is not None and sf is not None and fl is not None:
            si_f, sf_f, fl_f = float(si), float(sf), float(fl)
            denom = max(abs(si_f), abs(sf_f), abs(fl_f))
            return abs(sf_f - si_f - fl_f) / max(denom, 1e-30) * 100.0
    except (TypeError, ValueError):
        pass
    old = meta.get("mass_balance_percent_direct")
    return float(old) if old is not None else float("nan")


def _json_safe_surface_flux_budget(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop dense time series from surface flux budget for summary.json."""
    skip = {"times_valid", "flux_rates_valid"}
    return {k: v for k, v in raw.items() if k not in skip}


def enrich_meta_physical_flux(
    cfg: dict[str, Any],
    meta: dict[str, Any],
    snaps_full: SnapList,
) -> None:
    phy = physical_mass_balance_from_snapshots(
        snaps_full,
        cfg,
        silica_initial=float(meta["silica_initial"]),
        silica_final=float(meta["silica_final"]),
    )
    meta.update(phy)

    odx = float(cfg.get("physical_flux_outer_dx", 3.0))
    idx = float(cfg.get("physical_flux_inner_dx", 5.0))
    meta["option_A_ring_at_r_minus_3dx"] = {
        "note": (
            "Invalid once the front passes the ring; legacy ring-gradient comparison."
        ),
        "physical_flux_outer_dx": odx,
        "physical_flux_inner_dx": idx,
        "influx_total_physical": phy["influx_total_physical"],
        "residual_physical": phy["residual_physical"],
        "mass_balance_percent_physical": phy["mass_balance_percent_physical"],
        "mass_balance_method": phy.get(
            "mass_balance_method_physical", "physical_flux_integral"
        ),
    }

    meta["mass_balance_surface_flux"] = compute_surface_flux_budget(snaps_full, cfg)


def save_h5(path: Path, snaps: SnapList) -> None:
    with h5py.File(path, "w") as h5:
        for step, c, pm, pc in snaps:
            g = h5.create_group(f"t_{step:07d}")
            g.create_dataset("c", data=c, compression="gzip")
            g.create_dataset("phi_m", data=pm, compression="gzip")
            g.create_dataset("phi_c", data=pc, compression="gzip")


def run_postprocess(
    cfg: dict,
    out_dir: Path,
    label: str,
    snaps_full: SnapList,
    meta: dict[str, Any],
    wall_seconds: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[float], list[float]]:
    """Plots, HDF5, summary, diagnostics from an in-memory snapshot list."""
    if not snaps_full:
        raise ValueError("snapshots list is empty")
    _step, c, pm, pc = snaps_full[-1]

    L, R = float(cfg["L"]), float(cfg["R"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])

    phi_tot = pm + pc
    rc, pm_r = radial_profile(pm, L=L, R=R)
    _, pc_r = radial_profile(pc, L=L, R=R)
    _, pt_r = radial_profile(phi_tot, L=L, R=R)
    radial_metrics = band_metrics(rc, pt_r, R)

    plot_fields_final(c, pm, pc, L=L, R=R, path=out_dir / "fields_final.png")
    plot_radial(
        rc,
        {"phi_m": pm_r, "phi_c": pc_r, "phi_m+phi_c": pt_r},
        out_dir / "radial_profile.png",
    )

    frames = [np.asarray(a + b) for _, _, a, b in snaps_full]
    mx = max(float(np.max(f)) for f in frames) if frames else 1.0
    frames_n = [np.clip(f / max(mx, 1e-9), 0.0, 1.0) for f in frames]
    write_animation(frames_n, out_dir / "evolution.mp4", fps=30.0)
    write_evolution_gif_phi_m(snaps_full, out_dir / "evolution.gif")

    h5_path = out_dir / "snapshots.h5"
    save_h5(h5_path, snaps_full)

    ts = analyse_all_snapshots(h5_path, L=L, R=R, dt=dt, skip_before=500)
    plot_band_count_evolution(ts["records"], dt, out_dir / "band_count_evolution.png")
    plot_kymograph(
        ts["kymograph_t"],
        ts["kymograph_r"],
        out_dir / "kymograph.png",
        title=label,
    )

    mp = ts["metrics_at_peak"]
    mf = ts["metrics_at_final"]
    peak_nb = int(ts["peak_band_count"])
    final_nb = int(ts["final_multislice_band_count"])
    persist = bool(peak_nb > 0 and final_nb >= 0.5 * peak_nb)

    pk_time = ts["peak_band_count_time"]
    peak_title = (
        f"Jabłczyński at peak band count (t={pk_time}, N={peak_nb}) — "
        f"{mp.get('classification', '')}"
    )
    plot_jablczynski(
        mp,
        out_dir / "jablczynski_timeresolved.png",
        title=peak_title,
        radial_centers=ts["peak_radial_centers"],
        radial_profile_arr=ts["peak_radial_profile"],
    )

    final_title = f"Jabłczyński at final — {mf.get('classification', '')}"
    plot_jablczynski(
        mf,
        out_dir / "jablczynski.png",
        title=final_title,
        radial_centers=rc,
        radial_profile_arr=pt_r,
    )

    anticorr = moganite_chalcedony_anticorr(pm, pc, L, R)
    plot_canonical_slice(
        pm,
        pc,
        L=L,
        R=R,
        path=out_dir / "canonical_slice.png",
        n_bands=final_nb,
        classification=str(mf.get("classification", "")),
        anticorr=anticorr,
    )

    pub_field, _pub_src = choose_pub_field(pm, pc)
    save_final_pub(pub_field, L=L, R=R, path=out_dir / "final_pub.png")

    ovs = overshoot_fraction(pm, pc)
    mb_physical = mass_balance_percent_from_meta(meta)
    mb_direct = float(meta.get("mass_balance_percent_direct", float("nan")))
    lab_detect = labyrinth_heuristic(pm, pc, L=L, R=R, final_band_count=final_nb)

    n_steps = int(round(T / dt))
    g_ = int(cfg["grid"])

    summary: dict[str, Any] = {
        "label": label,
        "parameters": cfg,
        "N_b": final_nb,
        "mean_q": mf.get("mean_q"),
        "std_q": mf.get("std_q"),
        "cv_q_percent": mf.get("cv_q"),
        "cv_d_spacings_percent": mf.get("cv_d_spacings"),
        "spearman_d_vs_index": mf.get("spearman_d_vs_index"),
        "classification": mf.get("classification"),
        "wall_seconds": wall_seconds,
        "silica_initial": meta.get("silica_initial"),
        "silica_final": meta.get("silica_final"),
        "cumulative_boundary_flux_mass_direct": meta.get(
            "cumulative_boundary_flux_mass"
        ),
        "mass_balance_percent": mb_physical,
        "mass_balance_percent_physical": meta.get("mass_balance_percent_physical"),
        "mass_balance_percent_direct": meta.get("mass_balance_percent_direct"),
        "influx_total_physical": meta.get("influx_total_physical"),
        "residual_physical": meta.get("residual_physical"),
        "mass_balance_method": "physical_flux_integral",
        "mass_balance_note": (
            "physical: residual = silica_final − silica_initial − ∫flux_dt "
            "(trapezoidal on snapshot influx rates from wall gradient); "
            "direct: chunk sum of Δ(silica) ≡ (final−initial) is tautological."
        ),
        "labyrinth_detected": lab_detect,
        "bands_r_outer_in": mf.get("r_outer_in"),
        "peak_band_count": peak_nb,
        "peak_band_count_time": pk_time,
        "final_band_count": final_nb,
        "classification_at_peak": mp.get("classification"),
        "classification_at_final": mf.get("classification"),
        "bands_persist": persist,
        "overshoot_fraction_final": ovs,
        "moganite_chalcedony_anticorrelation": anticorr,
        "metrics_at_peak": mp,
        "metrics_at_final": mf,
        "radial_azimuthal_aux": {
            "N_b": radial_metrics["N_b"],
            "classification": radial_metrics["classification"],
        },
        "option_A_ring_at_r_minus_3dx": meta.get("option_A_ring_at_r_minus_3dx"),
        "mass_balance_surface_flux": _json_safe_surface_flux_budget(
            meta.get("mass_balance_surface_flux")
            or meta.get("option_B_fixed_surface")
            or {}
        ),
        "spectral_mass_conservation": meta.get(
            "spectral_mass_conservation",
            meta.get("option_D_spectral_conservation"),
        ),
        "silica_full_domain_initial": meta.get("silica_full_domain_initial"),
        "silica_full_domain_final": meta.get("silica_full_domain_final"),
        "ring_mask_sanity": meta.get("ring_mask_sanity"),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    d_list = mf.get("d") or []
    sp_str = ", ".join(f"{x:.4f}" for x in d_list) if d_list else "(none)"
    mq = mf.get("mean_q", float("nan"))
    cvq = mf.get("cv_q", float("nan"))
    cvq_pct = (cvq * 100) if cvq == cvq else float("nan")
    cvd = mf.get("cv_d_spacings", float("nan"))
    cvd_pct = (cvd * 100) if cvd == cvd else float("nan")
    rho_d = mf.get("spearman_d_vs_index", float("nan"))
    klass = mf.get("classification", "")

    print("")
    print(f"=== AGATE CH — {label} ===")
    print(
        f"Grid: {g_}×{g_}  T: {T}  steps: {n_steps}  wall-clock: "
        f"{fmt_hms(wall_seconds)}"
    )
    print(f"Overshoot final: {ovs:.2f}%                     [pass if <1%]")
    mbd = mb_direct if mb_direct == mb_direct else float("nan")
    mbp = mb_physical if mb_physical == mb_physical else float("nan")
    print(f"Mass balance (physical flux): {mbp:.4f}%            [pass if |·|<5%]")
    print(f"Mass balance (direct/chunk): {mbd:.6f}%        [tautology check]")
    print("")
    print("MULTI-SLICE BAND COUNT:")
    print(f"  Peak: {peak_nb} at t={pk_time}")
    print(f"  Final (median of 16 slice/field samples): {final_nb}")
    print(f"  Persist: {persist}")
    print("")
    print("CANONICAL SLICE JABŁCZYŃSKI (final):")
    print(f"  N peaks: {int(mf.get('N_b', 0))}, spacings: [{sp_str}]")
    if mq == mq:
        print(f"  mean(q): {mq:.2f}, CV(q): {cvq_pct:.1f}%, CV(d_n): {cvd_pct:.1f}%")
    else:
        print("  mean(q): n/a")
    if rho_d == rho_d:
        print(f"  Spearman ρ(d, n): {rho_d:.2f}")
    else:
        print("  Spearman ρ(d, n): n/a")
    print(f"  Classification: {klass}")
    print("")
    ac_show = f"{anticorr:.2f}" if anticorr == anticorr else "nan"
    print(f"MOGANITE/CHALCEDONY ANTI-CORRELATION: {ac_show}")
    print("  [Agate-like if < -0.3]")
    print("")
    print(f"Outputs: {out_dir}/")

    return summary, rc, pt_r, ts["kymograph_t"], ts["kymograph_r"]


def run_simulation(
    cfg: dict, out_dir: Path, *, label: str = "BASELINE"
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[float], list[float]]:
    chunk = int(cfg.get("chunk_size", 2000))
    t0 = time.perf_counter()
    c, pm, pc, meta, snaps = simulate_to_host(cfg, chunk_size=chunk)
    wall = time.perf_counter() - t0

    geom = meta["geom"]
    key = jax.random.PRNGKey(int(cfg.get("seed", 0)))
    prm = cfg_to_sim_params(cfg)
    ic = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )
    snaps_full: SnapList = [
        (
            0,
            np.asarray(jax.device_get(ic[0])),
            np.asarray(jax.device_get(ic[1])),
            np.asarray(jax.device_get(ic[2])),
        )
    ]
    snaps_full.extend(snaps)
    enrich_meta_physical_flux(cfg, meta, snaps_full)
    _rec = cfg.get(
        "record_spectral_mass_diagnostic",
        cfg.get("record_option_D_spectral", False),
    )
    if _rec:
        meta["spectral_mass_conservation"] = spectral_mass_conservation_diagnostic(cfg)

    return run_postprocess(cfg, out_dir, label, snaps_full, meta, wall)


def run_reanalyze(
    src_dir: Path,
    cfg: dict,
    out_dir: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[float], list[float]]:
    h5_src = src_dir / "snapshots.h5"
    if not h5_src.is_file():
        raise FileNotFoundError(h5_src)
    snaps_full = load_snapshots_h5(h5_src)
    summ_path = src_dir / "summary.json"
    meta: dict[str, Any] = {}
    if summ_path.is_file():
        old = json.loads(summ_path.read_text())
        meta = {
            "silica_initial": old.get("silica_initial"),
            "silica_final": old.get("silica_final"),
            "cumulative_boundary_flux_mass": old.get(
                "cumulative_boundary_flux_mass_direct",
                old.get("cumulative_boundary_flux_mass"),
            ),
            "mass_balance_percent_direct": old.get(
                "mass_balance_percent_direct", old.get("mass_balance_percent")
            ),
        }
        cfg = merge_cfg(cfg, old.get("parameters", {}))
    rho_m = float(cfg.get("rho_m", 1.0))
    rho_c = float(cfg.get("rho_c", 1.0))
    L_, R_ = float(cfg["L"]), float(cfg["R"])
    st0, st1 = snaps_full[0], snaps_full[-1]
    si = total_silica_numpy(
        st0[1], st0[2], st0[3], L=L_, R=R_, rho_m=rho_m, rho_c=rho_c
    )
    sf = total_silica_numpy(
        st1[1], st1[2], st1[3], L=L_, R=R_, rho_m=rho_m, rho_c=rho_c
    )
    meta.setdefault("silica_initial", si)
    meta.setdefault("silica_final", sf)
    enrich_meta_physical_flux(cfg, meta, snaps_full)
    return run_postprocess(cfg, out_dir, label, snaps_full, meta, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agate Cahn–Hilliard falsification runner")
    ap.add_argument("--config", type=str, default="configs/agate_ch/baseline.yaml")
    ap.add_argument("--sweep", type=str, default="", help="Optional sweep YAML")
    ap.add_argument("--quick", action="store_true", help="small grid / short time")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm step bar (stdout stays clean for logs)",
    )
    ap.add_argument(
        "--reanalyze-dir",
        type=str,
        default="",
        help="skip simulation; rebuild diagnostics from snapshots.h5 in this folder",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "output directory for --reanalyze-dir "
            "(default: results/agate_ch/run_<timestamp>)"
        ),
    )
    ap.add_argument(
        "--generate-paper",
        nargs=2,
        metavar=("MAIN_SWEEP_DIR", "GAMMA_SWEEP_DIR"),
        help="build paper_figures/ under MAIN from sweep outputs",
    )
    ap.add_argument(
        "--write-results",
        nargs=2,
        metavar=("MAIN_SWEEP_DIR", "GAMMA_SWEEP_DIR"),
        help="write RESULTS.md; use 'none' for gamma if unavailable",
    )
    args = ap.parse_args()
    root = _repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    baseline = load_yaml(cfg_path)
    if args.no_progress:
        baseline["progress"] = False
    if args.quick:
        baseline["grid"] = 128
        baseline["T"] = 2.0
        baseline["snapshot_every"] = 100

    if args.generate_paper:
        ma = Path(args.generate_paper[0])
        ga = Path(args.generate_paper[1])
        if not ma.is_absolute():
            ma = root / ma
        if not ga.is_absolute():
            ga = root / ga
        generate_paper_figures(ma, ga)
        print(f"Paper figures: {ma / 'paper_figures'}/")
        return

    if args.write_results:
        ma = Path(args.write_results[0])
        g_arg = args.write_results[1]
        if not ma.is_absolute():
            ma = root / ma
        ga: Path | None = None
        if g_arg.lower() != "none":
            ga = Path(g_arg)
            if not ga.is_absolute():
                ga = root / ga
        write_results_markdown(root, ma, ga)
        print(f"Wrote {root / 'RESULTS.md'}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.reanalyze_dir:
        src = Path(args.reanalyze_dir)
        if not src.is_absolute():
            src = root / src
        if args.out_dir:
            out_dir = Path(args.out_dir)
            if not out_dir.is_absolute():
                out_dir = root / out_dir
        else:
            out_dir = agate_ch_results_dir(root) / f"run_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        lab = cfg_path.stem
        run_reanalyze(src, baseline, out_dir, label=lab)
        print(f"(reanalyze) wrote {out_dir}/")
        return

    if args.sweep:
        sw_path = Path(args.sweep)
        if not sw_path.is_absolute():
            sw_path = root / sw_path
        sweep = load_yaml(sw_path)
        sweep_dir = agate_ch_results_dir(root) / f"sweep_{ts}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
        kymos: list[tuple[str, list[float], list[float]]] = []
        sweep_entries: list[tuple[str, Path]] = []
        summaries: dict[str, dict[str, Any]] = {}
        table_rows: list[dict[str, str]] = []
        ids_order: list[str] = []
        total_wall = 0.0
        for item in sweep["runs"]:
            cid = item["id"]
            merged = merge_cfg(baseline, item.get("params", {}))
            out = sweep_dir / cid
            out.mkdir(parents=True, exist_ok=True)
            summ, rc, pt_r, kt, kr = run_simulation(merged, out, label=cid)
            sweep_entries.append((cid, out))
            summaries[cid] = summ
            ids_order.append(cid)
            total_wall += float(summ.get("wall_seconds") or 0.0)
            table_rows.append(sweep_summary_row(summ))
            lab = f"{cid}: {summ['classification_at_final']}"
            profiles.append((lab, rc, pt_r))
            kymos.append((cid, kt, kr))

        write_sweep_summaries_csv_txt(table_rows, sweep_dir)
        plot_sweep_compare(profiles, sweep_dir / "sweep_comparison.png")
        plot_sweep_kymographs(kymos, sweep_dir / "sweep_kymographs.png")
        plot_comparison_grid(sweep_entries, sweep_dir / "comparison_grid.png")
        plot_canonical_slice_grid(sweep_entries, sweep_dir / "canonical_slice_grid.png")
        sweep_kind = str(sweep.get("kind", "main"))
        if sweep_kind == "gamma_scan":
            paired = [(cid, summaries[cid]) for cid in ids_order]
            paired.sort(
                key=lambda z: float((z[1].get("parameters") or {}).get("gamma", 0.0))
            )
            gamma_rows: list[dict[str, Any]] = []
            ordered_entries: list[tuple[str, Path]] = []
            for cid, summ in paired:
                mf = summ.get("metrics_at_final") or {}
                prm = summ.get("parameters") or {}
                gamma = float(prm.get("gamma", float("nan")))
                cv_q = mf.get("cv_q", float("nan"))
                mq = float(mf.get("mean_q") or 1e-9)
                std_q = float(mf.get("std_q") or 0.0)
                std_pct = abs(std_q / mq * 100.0) if mq else 0.0
                gamma_rows.append(
                    {
                        "gamma": gamma,
                        "N_bands": int(summ.get("final_band_count", 0)),
                        "CV_q_pct": (cv_q * 100.0) if cv_q == cv_q else 0.0,
                        "spearman_rho": mf.get("spearman_d_vs_index"),
                        "anticorrelation": summ.get(
                            "moganite_chalcedony_anticorrelation"
                        ),
                        "classification": summ.get("classification_at_final"),
                        "labyrinth": bool(summ.get("labyrinth_detected", False)),
                        "std_q_pct_err": std_pct,
                    }
                )
                ordered_entries.append((cid, sweep_dir / cid))
            plot_gamma_scan_fields(
                ordered_entries,
                sweep_dir / "gamma_scan_fields.png",
                titles_gamma=[rf"$\gamma={r['gamma']:.1f}$" for r in gamma_rows],
            )
            plot_gamma_phase_diagram(
                gamma_rows,
                sweep_dir / "gamma_phase_diagram.png",
                sweep_dir / "gamma_phase_diagram.csv",
            )
            report_gamma_scan(
                gamma_rows,
                sweep_dir=sweep_dir.resolve(),
                total_wall_s=total_wall,
            )
        else:
            report_main_sweep(
                summaries,
                sweep_dir=sweep_dir.resolve(),
                ids_order=ids_order,
                total_wall_s=total_wall,
            )
        return

    out_dir = agate_ch_results_dir(root) / f"run_{ts}"
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_simulation(baseline, out_dir, label=cfg_path.stem)


if __name__ == "__main__":
    main()
