"""CLI: `python -m continuous_patterns.agate_ch.run --config configs/baseline.yaml`."""

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

from continuous_patterns.agate_ch.diagnostics import band_metrics, radial_profile
from continuous_patterns.agate_ch.plotting import (
    plot_fields_final,
    plot_jablczynski,
    plot_radial,
    plot_sweep_compare,
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
) -> tuple[dict, np.ndarray, np.ndarray]:
    chunk = int(cfg.get("chunk_size", 2000))
    t0 = time.perf_counter()
    c, pm, pc, meta, snaps = simulate_to_host(cfg, chunk_size=chunk)
    wall = time.perf_counter() - t0

    L, R = float(cfg["L"]), float(cfg["R"])
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

    metrics = band_metrics(rc, pt_r, R)

    plot_fields_final(c, pm, pc, L=L, R=R, path=out_dir / "fields_final.png")
    plot_radial(
        rc,
        {"phi_m": pm_r, "phi_c": pc_r, "phi_m+phi_c": pt_r},
        out_dir / "radial_profile.png",
    )
    plot_jablczynski(metrics, out_dir / "jablczynski.png", metrics["classification"])

    frames = [np.asarray(a + b) for _, _, a, b in snaps_full]
    mx = max(float(np.max(f)) for f in frames) if frames else 1.0
    frames_n = [np.clip(f / max(mx, 1e-9), 0.0, 1.0) for f in frames]
    write_animation(frames_n, out_dir / "evolution.mp4", fps=30.0)

    save_h5(out_dir / "snapshots.h5", snaps_full)

    dt = float(cfg["dt"])
    T = float(cfg["T"])
    n_steps = int(round(T / dt))
    mb_err = (
        abs(meta["mass_final"] - meta["mass_initial"])
        / max(abs(meta["mass_initial"]), 1e-12)
        * 100.0
    )

    cvq = metrics["cv_q"]
    cv_qpct = float(cvq) * 100 if cvq == cvq else None
    summary = {
        "label": label,
        "parameters": cfg,
        "N_b": metrics["N_b"],
        "mean_q": metrics["mean_q"],
        "std_q": metrics["std_q"],
        "cv_q_percent": cv_qpct,
        "classification": metrics["classification"],
        "wall_seconds": wall,
        "mass_balance_percent": mb_err,
        "bands_r_outer_in": metrics["r_outer_in"],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if label == "BASELINE":
        ro = metrics["r_outer_in"]
        pos = ", ".join(f"{x:.3f}" for x in ro) if ro else ""
        mq, sq, cv = metrics["mean_q"], metrics["std_q"], metrics["cv_q"]
        cv_pct = (cv * 100) if cv == cv else float("nan")
        print("")
        print("=== AGATE CH FALSIFICATION TEST — BASELINE ===")
        g_ = cfg["grid"]
        print(f"Grid: {g_}x{g_}  T: {T}  steps: {n_steps}  wall-clock: {fmt_hms(wall)}")
        print(f"Bands detected: {metrics['N_b']}")
        print(f"Band positions (outer→inner): {pos}")
        jab = (
            f"Jabłczyński ratio q = d_n/d_{{n-1}}: mean={mq:.2f}, std={sq:.2f}, "
            f"CV={cv_pct:.1f}%"
        )
        print(jab)
        print(f"Classification: {metrics['classification']}")
        print(f"Mass balance error: {mb_err:.3f}%")
        print(f"Outputs: {out_dir}/")

    return summary, rc, pt_r


def main() -> None:
    ap = argparse.ArgumentParser(description="Agate Cahn–Hilliard falsification runner")
    ap.add_argument("--config", type=str, default="configs/baseline.yaml")
    ap.add_argument("--sweep", type=str, default="", help="Optional sweep YAML")
    ap.add_argument("--quick", action="store_true", help="small grid / short time")
    args = ap.parse_args()
    root = _repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    baseline = load_yaml(cfg_path)
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = root / "results" / f"sweep_{ts}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for item in sweep["runs"]:
            cid = item["id"]
            merged = merge_cfg(baseline, item.get("params", {}))
            out = sweep_dir / cid
            out.mkdir(parents=True, exist_ok=True)
            summ, rc, pt_r = run_one(merged, out, label=cid)
            lab = f"{cid}: {summ['classification']}"
            profiles.append((lab, rc, pt_r))

        plot_sweep_compare(profiles, sweep_dir / "sweep_comparison.png")
        print(f"Sweep outputs: {sweep_dir}/")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "results" / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_one(baseline, out_dir, label="BASELINE")


if __name__ == "__main__":
    main()
