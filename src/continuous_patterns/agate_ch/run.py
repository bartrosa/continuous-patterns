"""CLI entry; default config ``configs/baseline_v15.yaml``."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import h5py
import jax
import numpy as np
import yaml

from continuous_patterns.agate_ch.diagnostics import (
    analyse_all_snapshots,
    band_metrics,
    overshoot_fraction,
    radial_profile,
)
from continuous_patterns.agate_ch.plotting import (
    plot_band_count_evolution,
    plot_fields_final,
    plot_jablczynski,
    plot_kymograph,
    plot_radial,
    plot_sweep_compare,
    plot_sweep_kymographs,
    write_animation,
)
from continuous_patterns.agate_ch.solver import (
    cfg_to_sim_params,
    initial_state,
    simulate_to_host,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def save_h5(path: Path, snaps: SnapList) -> None:
    with h5py.File(path, "w") as h5:
        for step, c, pm, pc in snaps:
            g = h5.create_group(f"t_{step:07d}")
            g.create_dataset("c", data=c, compression="gzip")
            g.create_dataset("phi_m", data=pm, compression="gzip")
            g.create_dataset("phi_c", data=pc, compression="gzip")


def run_one(
    cfg: dict, out_dir: Path, *, label: str = "BASELINE"
) -> tuple[dict, np.ndarray, np.ndarray, list[float], list[float]]:
    chunk = int(cfg.get("chunk_size", 2000))
    t0 = time.perf_counter()
    c, pm, pc, meta, snaps = simulate_to_host(cfg, chunk_size=chunk)
    wall = time.perf_counter() - t0

    L, R = float(cfg["L"]), float(cfg["R"])
    dt = float(cfg["dt"])
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

    phi_tot = pm + pc
    rc, pm_r = radial_profile(pm, L=L, R=R)
    _, pc_r = radial_profile(pc, L=L, R=R)
    _, pt_r = radial_profile(phi_tot, L=L, R=R)

    metrics_final = band_metrics(rc, pt_r, R)

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
    final_nb = int(mf.get("N_b", 0))
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

    ovs = overshoot_fraction(pm, pc)
    mb_err = float(meta.get("mass_balance_percent", 0.0))

    n_steps = int(round(float(cfg["T"]) / dt))
    T = float(cfg["T"])
    g_ = int(cfg["grid"])

    summary: dict = {
        "label": label,
        "parameters": cfg,
        "N_b": metrics_final["N_b"],
        "mean_q": metrics_final["mean_q"],
        "std_q": metrics_final["std_q"],
        "cv_q_percent": metrics_final.get("cv_q"),
        "classification": metrics_final["classification"],
        "wall_seconds": wall,
        "mass_balance_percent": mb_err,
        "mass_balance_note": (
            "(silica_total_final − silica_initial − ∫flux_in dt) / silica_initial"
        ),
        "bands_r_outer_in": metrics_final["r_outer_in"],
        "peak_band_count": peak_nb,
        "peak_band_count_time": pk_time,
        "final_band_count": final_nb,
        "classification_at_peak": mp.get("classification"),
        "classification_at_final": mf.get("classification"),
        "bands_persist": persist,
        "overshoot_fraction_final": ovs,
        "metrics_at_peak": mp,
        "metrics_at_final": mf,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    def _jab_lines(met: dict) -> str:
        ro = met.get("r_outer_in") or []
        pos = ", ".join(f"{x:.3f}" for x in ro) if ro else ""
        mq = met.get("mean_q", float("nan"))
        sq = met.get("std_q", float("nan"))
        cv = met.get("cv_q", float("nan"))
        cv_pct = (cv * 100) if cv == cv else float("nan")
        klass = met.get("classification", "")
        sp = f"  Spacings: {pos}" if pos else "  Spacings: (none)"
        jq = (
            f"  q mean={mq:.2f}  std={sq:.2f}  CV={cv_pct:.1f}%"
            if mq == mq
            else "  q: n/a"
        )
        kl = f"  → {klass}"
        return "\n".join([sp, jq, kl])

    if label == "BASELINE":
        blk_peak = _jab_lines(mp)
        blk_fin = _jab_lines(mf)
        print("")
        print(f"=== AGATE CH v1.5 — {label} ===")
        print(f"Grid: {g_}×{g_}  T: {T}  steps: {n_steps}  wall-clock: {fmt_hms(wall)}")
        print(f"Overshoot fraction (final): {ovs:.2f}%          [pass if <1%]")
        print(f"Mass balance error: {mb_err:.2f}%                  [pass if <2%]")
        print("")
        print("BAND EVOLUTION:")
        print(f"  Peak band count: {peak_nb} at t={pk_time}")
        print(f"  Final band count: {final_nb}")
        print(f"  Bands persist (final ≥ 0.5·peak): {persist}")
        print("")
        print("JABŁCZYŃSKI AT PEAK:")
        print(blk_peak)
        print("")
        print("JABŁCZYŃSKI AT FINAL:")
        print(blk_fin)
        print("")
        print(f"Outputs: {out_dir}/")

    return summary, rc, pt_r, ts["kymograph_t"], ts["kymograph_r"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Agate Cahn–Hilliard falsification runner")
    ap.add_argument("--config", type=str, default="configs/baseline_v15.yaml")
    ap.add_argument("--sweep", type=str, default="", help="Optional sweep YAML")
    ap.add_argument("--quick", action="store_true", help="small grid / short time")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm step bar (stdout stays clean for logs)",
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

    if args.sweep:
        sw_path = Path(args.sweep)
        if not sw_path.is_absolute():
            sw_path = root / sw_path
        sweep = load_yaml(sw_path)
        profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
        kymos: list[tuple[str, list[float], list[float]]] = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = root / "results" / f"sweep_{ts}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for item in sweep["runs"]:
            cid = item["id"]
            merged = merge_cfg(baseline, item.get("params", {}))
            out = sweep_dir / cid
            out.mkdir(parents=True, exist_ok=True)
            summ, rc, pt_r, kt, kr = run_one(merged, out, label=cid)
            lab = f"{cid}: {summ['classification_at_final']}"
            profiles.append((lab, rc, pt_r))
            kymos.append((cid, kt, kr))

        plot_sweep_compare(profiles, sweep_dir / "sweep_comparison.png")
        plot_sweep_kymographs(kymos, sweep_dir / "sweep_v15_comparison.png")
        print(f"Sweep outputs: {sweep_dir}/")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_one(baseline, out_dir, label="BASELINE")


if __name__ == "__main__":
    main()
