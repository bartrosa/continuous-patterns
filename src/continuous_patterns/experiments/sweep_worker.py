"""Process-pool worker entry point for axis sweeps (CPU-only JAX in child processes)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def ensure_cpu_jax_backend() -> None:
    """Keep worker subprocesses off the GPU (avoids CUDA OOM with ``--workers N``)."""
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _analytic_strip_predict(Da: float, kappa: float, c1: float, c2: float) -> bool:
    lo = 1.0 / (1.0 + float(kappa) * float(Da))
    hi = 1.0 / (1.0 + float(Da))
    return (float(c2) > lo) and (float(c1) < hi) and (float(c1) > float(c2))


def _kinetic_skipped_entry(
    *,
    run_label: str,
    combo: dict[str, Any],
    Da_v: float,
    ka_v: float,
    c1v: float,
    c2v: float,
    Tv: float,
) -> dict[str, Any]:
    osc_p = _analytic_strip_predict(Da_v, ka_v, c1v, c2v)
    return {
        "run_label": run_label,
        "parameters": combo,
        "status": "skipped",
        "relative_path": None,
        "band_count_axial": None,
        "residual_pct": None,
        "front_origin": None,
        "wall_time_s": None,
        "error": None,
        "Da": Da_v,
        "kappa": ka_v,
        "c1": c1v,
        "c2": c2v,
        "T": Tv,
        "n_transitions": None,
        "dwell_L_mean": None,
        "dwell_H_mean": None,
        "peak_period": None,
        "thickness_final": None,
        "osc_predicted": osc_p,
        "osc_observed": False,
        "osc_match": not osc_p,
    }


def _kinetic_manifest_extras(cfg: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    phys = cfg.get("physics", {})
    Da = float(phys.get("Da", float("nan")))
    kappa = float(phys.get("kappa", float("nan")))
    c1 = float(phys.get("c1", float("nan")))
    c2 = float(phys.get("c2", float("nan")))
    osc_pred = _analytic_strip_predict(Da, kappa, c1, c2)
    n_tr = int(diag.get("n_transitions", 0))
    osc_obs = n_tr >= 2

    def _opt(key: str) -> float | None:
        v = diag.get(key)
        return None if v is None else float(v)

    return {
        "Da": Da,
        "kappa": kappa,
        "c1": c1,
        "c2": c2,
        "T": float(cfg.get("time", {}).get("T", float("nan"))),
        "n_transitions": n_tr,
        "dwell_L_mean": _opt("dwell_L_mean"),
        "dwell_H_mean": _opt("dwell_H_mean"),
        "peak_period": _opt("peak_period"),
        "mean_dwell_period": _opt("mean_dwell_period"),
        "mean_dwell_freq": _opt("mean_dwell_freq"),
        "thickness_final": _opt("thickness_final"),
        "osc_predicted": osc_pred,
        "osc_observed": osc_obs,
        "osc_match": bool(osc_pred == osc_obs),
    }


def _front_origin_from_h5_or_final(
    run_dir: Path, *, rim_left_width_px: int, rim_width_px: int
) -> str:
    import numpy as np

    h5_path = run_dir / "snapshots.h5"
    phi_total: Any = None
    if h5_path.is_file():
        import h5py

        with h5py.File(h5_path, "r") as h5:
            keys = sorted([k for k in h5.keys() if k.startswith("snap_")])
            if keys:
                g = h5[keys[len(keys) // 2]]
                phi_total = g["phi_m"][()] + g["phi_c"][()]
    if phi_total is None:
        arr = __import__("numpy").load(run_dir / "final_state.npz")
        phi_total = arr["phi_m"] + arr["phi_c"]
    n = int(phi_total.shape[0])
    left_hi = min(n, int(rim_left_width_px) + 10)
    right_lo = max(0, n - int(rim_width_px) - 10)
    left_mass = float(np.mean(phi_total[:left_hi, :]))
    right_mass = float(np.mean(phi_total[right_lo:, :]))
    if left_mass > right_mass + 0.1:
        return "LEFT"
    if right_mass > left_mass + 0.1:
        return "RIGHT"
    return "BOTH"


def run_s1_combo(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one explicit S1 run with post-run zone analysis."""
    ensure_cpu_jax_backend()
    from continuous_patterns.experiments.run import run_one
    from continuous_patterns.experiments.s1_ostwald_window import analyze_s1_run

    cfg = payload["cfg"]
    run_label = str(payload["run_label"])
    sweep_dir = Path(payload["sweep_dir"])
    chunk_size = int(payload["chunk_size"])
    analyze_s1 = bool(payload.get("analyze_s1", True))

    try:
        t0 = time.perf_counter()
        result = run_one(
            cfg,
            results_root=sweep_dir,
            chunk_size=chunk_size,
            write_artifacts=True,
            show_progress=False,
            log_level="WARNING",
        )
        assert result.paths is not None
        rel_path = result.paths.root.relative_to(sweep_dir).as_posix()
        wall_time_s = float(time.perf_counter() - t0)
        if analyze_s1:
            analysis = analyze_s1_run(result.paths.root, run_label=run_label)
            return {
                "run_label": run_label,
                "status": "success",
                "relative_path": rel_path,
                "wall_time_s": wall_time_s,
                "c_ostwald": analysis.get("c_ostwald"),
                "w_ostwald": analysis.get("w_ostwald"),
                "gamma": analysis.get("gamma"),
                "n_bands_zone": analysis.get("n_bands_zone"),
                "band_zone_extent": analysis.get("band_zone_extent"),
                "mean_spacing_zone": analysis.get("mean_spacing_zone"),
                "median_spacing_zone": analysis.get("median_spacing_zone"),
                "mean_spacing_zone_excl_last": analysis.get("mean_spacing_zone_excl_last"),
                "std_spacing_zone": analysis.get("std_spacing_zone"),
                "mass_residual_pct": analysis.get("mass_residual_pct"),
                "clipped_mass_frac": analysis.get("clipped_mass_frac"),
                "row_validity": analysis.get("row_validity"),
                "spacing_pacing": analysis.get("spacing_pacing"),
                "error": None,
            }
        diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        return {
            "run_label": run_label,
            "status": "success",
            "relative_path": rel_path,
            "wall_time_s": wall_time_s,
            "error": None,
            "mass_residual_pct": diag.get("mass_residual_pct"),
        }
    except Exception as exc:
        return {
            "run_label": run_label,
            "status": "failed",
            "relative_path": None,
            "wall_time_s": None,
            "error": str(exc),
        }


def run_combo(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one sweep combination; intended for ``ProcessPoolExecutor`` workers."""
    ensure_cpu_jax_backend()
    from continuous_patterns.experiments.run import run_one

    cfg = payload["cfg"]
    combo = payload["combo"]
    run_label = payload["run_label"]
    sweep_dir = Path(payload["sweep_dir"])
    chunk_size = int(payload["chunk_size"])
    exp_m = str(cfg.get("experiment", {}).get("model", ""))
    phys = cfg.get("physics", {})
    c1v = float(phys.get("c1", float("nan")))
    c2v = float(phys.get("c2", float("nan")))
    Da_v = float(phys.get("Da", float("nan")))
    ka_v = float(phys.get("kappa", float("nan")))
    Tv = float(cfg.get("time", {}).get("T", float("nan")))

    if exp_m == "slab_kinetic" and not (c1v > c2v):
        return _kinetic_skipped_entry(
            run_label=run_label,
            combo=combo,
            Da_v=Da_v,
            ka_v=ka_v,
            c1v=c1v,
            c2v=c2v,
            Tv=Tv,
        )

    try:
        t0 = time.perf_counter()
        result = run_one(
            cfg,
            results_root=sweep_dir,
            chunk_size=chunk_size,
            write_artifacts=True,
            show_progress=False,
            log_level="WARNING",
        )
        assert result.paths is not None
        rel_path = result.paths.root.relative_to(sweep_dir).as_posix()
        wall_time_s = float(time.perf_counter() - t0)
        diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        slab_ax = diag.get("slab_axial", {})
        dmb = diag.get("dirichlet_mass_balance", {})
        band_count = int(slab_ax.get("band_count_axial", 0)) if isinstance(slab_ax, dict) else None
        residual_pct = float(dmb.get("residual_pct", 0.0)) if isinstance(dmb, dict) else None
        row: dict[str, Any] = {
            "run_label": run_label,
            "parameters": combo,
            "status": "success",
            "relative_path": rel_path,
            "band_count_axial": band_count,
            "residual_pct": residual_pct,
            "front_origin": None,
            "wall_time_s": wall_time_s,
            "error": None,
        }
        if exp_m == "slab_kinetic":
            row.update(_kinetic_manifest_extras(cfg, diag))
        else:
            row["front_origin"] = _front_origin_from_h5_or_final(
                sweep_dir / rel_path,
                rim_left_width_px=int(cfg.get("geometry", {}).get("rim_left_width_px", 0)),
                rim_width_px=int(cfg.get("geometry", {}).get("rim_width_px", 0)),
            )
        return row
    except ValueError as exc:
        msg = str(exc)
        if exp_m == "slab_kinetic" and "c1 must be > c2" in msg:
            return _kinetic_skipped_entry(
                run_label=run_label,
                combo=combo,
                Da_v=Da_v,
                ka_v=ka_v,
                c1v=c1v,
                c2v=c2v,
                Tv=Tv,
            )
        return {
            "run_label": run_label,
            "parameters": combo,
            "status": "failed",
            "relative_path": None,
            "band_count_axial": None,
            "residual_pct": None,
            "front_origin": None,
            "wall_time_s": None,
            "error": msg,
        }
    except Exception as exc:
        return {
            "run_label": run_label,
            "parameters": combo,
            "status": "failed",
            "relative_path": None,
            "band_count_axial": None,
            "residual_pct": None,
            "front_origin": None,
            "wall_time_s": None,
            "error": str(exc),
        }
