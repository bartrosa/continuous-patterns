#!/usr/bin/env python3
"""EXP-D5 analysis: reaction-limited Heaney moving-front diagnosis + Da(t) verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_patterns.core.banding_metrics import (  # noqa: E402
    compute_liesegang_laws,
    record_from_slab_twophase_result,
)
from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.experiments.checkpoint import load_sim_result_from_run_dir  # noqa: E402

_PREDICTION = (
    "At least one low-c0 / slow-D run (c0<=0.35, D<=0.03) sustains >=3 L<->H cycles "
    "while Da(t)>~1 before the front reaches the wall."
)
_DA_THRESHOLD = 1.0


def _load_manifest(sweep_dir: Path) -> dict[str, Any]:
    p = sweep_dir / "manifest_d5h.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_dir(sweep_dir: Path, entry: dict[str, Any]) -> Path:
    return (sweep_dir / str(entry["relative_path"])).resolve()


def _state_str(s: Any) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def _gate_c_dep(snap: dict[str, np.ndarray], *, label: str) -> None:
    cdep = np.asarray(snap.get("snapshots_c_dep", []), dtype=np.float64).ravel()
    if cdep.size == 0:
        raise RuntimeError(f"{label}: missing snapshots_c_dep")
    bad = ~np.isfinite(cdep) | (cdep < -1e-9) | (cdep > 1.0 + 1e-9)
    if np.any(bad):
        idx = int(np.where(bad)[0][0])
        raise RuntimeError(
            f"{label}: c_dep gate FAIL frame {idx} value={cdep[idx]!r} (must be in [0,1])"
        )


def _count_lh_cycles(
    states: list[str], c_dep: np.ndarray, *, c1: float, c2: float
) -> dict[str, Any]:
    if c_dep.size < 4 or len(states) < 4:
        return {"n_cycles": 0, "crossings_up": 0, "crossings_down": 0, "oscillates": False}
    up = down = 0
    for i in range(1, c_dep.size):
        if c_dep[i - 1] <= c1 < c_dep[i]:
            up += 1
        if c_dep[i - 1] >= c2 > c_dep[i]:
            down += 1
    n_cycles = min(up, down)
    return {
        "n_cycles": int(n_cycles),
        "crossings_up": int(up),
        "crossings_down": int(down),
        "oscillates": bool(n_cycles >= 3),
    }


def _resolve_baseline_physics(cfg: dict[str, Any]) -> dict[str, float]:
    base = load_run_config(_ROOT / "experiments/canonical/slab_twophase/baseline.yaml")
    base_phys = base["physics"]
    phys = cfg["physics"]
    D_base = float(base_phys["D_c"])
    k_rxn_base = float(base_phys["k_rxn"])
    k_L_scale = float(phys.get("k_L_scale", 1.0))
    kappa_rate = float(phys.get("kappa_rate", 5.0))
    k_L_abs = k_L_scale * k_rxn_base
    k_H_abs = kappa_rate * k_L_abs
    D_abs = float(phys["D_c"])
    return {
        "D_base": D_base,
        "k_rxn_base": k_rxn_base,
        "k_L_scale": k_L_scale,
        "kappa_rate": kappa_rate,
        "k_L_abs": k_L_abs,
        "k_H_abs": k_H_abs,
        "D_abs": D_abs,
        "c0": float(phys["c_0"]),
    }


def _compute_da_series(snap: dict[str, np.ndarray], *, D: float) -> np.ndarray:
    if "snapshots_Da" in snap:
        return np.asarray(snap["snapshots_Da"], dtype=np.float64).ravel()
    gap = np.asarray(snap.get("snapshots_gap_width", []), dtype=np.float64).ravel()
    k_eff = np.asarray(snap.get("snapshots_k_eff", []), dtype=np.float64).ravel()
    if gap.size and k_eff.size == gap.size:
        return k_eff * gap / max(D, 1e-30)
    raise RuntimeError("missing snapshots_Da and cannot reconstruct from k_eff/gap")


def _da_oscillation_report(
    *,
    states: list[str],
    c_dep: np.ndarray,
    da: np.ndarray,
    t: np.ndarray,
    c1: float,
    c2: float,
) -> dict[str, Any]:
    cycles = _count_lh_cycles(states, c_dep, c1=c1, c2=c2)
    st = np.asarray([1.0 if _state_str(s) == "H" else 0.0 for s in states])
    flips = np.where(np.diff(st) != 0)[0] + 1
    da_start = da_stop = None
    sustained = False
    dying = False
    if cycles["n_cycles"] >= 3:
        sustained = True
        if flips.size >= 2:
            da_start = float(da[int(flips[0])])
            da_stop = float(da[int(flips[-1])])
    elif cycles["n_cycles"] >= 1:
        dying = True
        if flips.size >= 1:
            da_start = float(da[int(flips[0])])
            da_stop = float(da[int(flips[-1])])
    below = np.where(da < _DA_THRESHOLD)[0]
    da_falls_below = float(da[int(below[0])]) if below.size else None
    t_below = float(t[int(below[0])]) if below.size and t.size == da.size else None
    frac_da_gt1 = float(np.mean(da > _DA_THRESHOLD)) if da.size else 0.0
    return {
        **cycles,
        "sustained": sustained,
        "dying": dying and not sustained,
        "da_start_osc": da_start,
        "da_stop_osc": da_stop,
        "da_at_first_below_1": da_falls_below,
        "t_at_first_da_below_1": t_below,
        "da_min": float(np.min(da)) if da.size else None,
        "da_max": float(np.max(da)) if da.size else None,
        "frac_time_da_gt_1": frac_da_gt1,
    }


def _band_metrics(run_dir: Path, label: str) -> dict[str, Any]:
    result = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    rec = record_from_slab_twophase_result(result, label)
    laws = compute_liesegang_laws(rec)
    d_sp = laws.get("d_spacings") or []
    mean_d = float(np.mean(np.asarray(d_sp, dtype=np.float64))) if d_sp else None
    snap = np.load(run_dir / "snapshots.npz")
    t = np.asarray(snap.get("snapshots_t", []), dtype=np.float64).ravel()
    xf = np.asarray(snap.get("snapshots_x_front", []), dtype=np.float64).ravel()
    v_open = None
    if t.size > 2 and xf.size == t.size:
        v_open = float(np.median(np.diff(xf) / np.diff(t)))
    tau = None
    states = snap.get("snapshots_S")
    if states is not None and t.size > 4:
        st = np.asarray([_state_str(s) for s in states])
        flips = np.where(np.diff(np.asarray([1 if s == "H" else 0 for s in st])) != 0)[0]
        if flips.size >= 2:
            periods = np.diff(t[flips])
            tau = float(np.median(periods)) if periods.size else None
    return {
        "lambda_band": mean_d,
        "tau_cycle": tau,
        "d_spacing": mean_d,
        "v_at_open": v_open,
        "n_bands": laws.get("n_bands"),
        "p_fit": laws.get("p_fit"),
    }


def _plot_d5h_diag(path: Path, runs: list[dict[str, Any]]) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * n), squeeze=False)
    for i, run in enumerate(runs):
        t = np.asarray(run["snap_t"], dtype=np.float64)
        ax0, ax1 = axes[i]
        cdep = np.asarray(run["snap_c_dep"], dtype=np.float64)
        ax0.plot(t, cdep, lw=0.9, color="0.15")
        ax0.axhline(run["c1"], color="#d62728", ls="--", lw=0.8, label="c1")
        ax0.axhline(run["c2"], color="#1f77b4", ls="--", lw=0.8, label="c2")
        osc = "OSC" if run["oscillates"] else "no osc"
        ax0.set_title(
            f"{run['run_label']} ({osc}, cycles={run['n_cycles']}, "
            f"c0={run['c0']}, D={run['D_abs']})"
        )
        ax0.set_ylabel(r"$c_{\mathrm{dep}}$")
        ax0.legend(fontsize=7, loc="upper right")
        if run.get("snap_S"):
            st = np.asarray([1.0 if _state_str(s) == "H" else 0.0 for s in run["snap_S"]])
            ax0b = ax0.inset_axes([0.0, 0.02, 1.0, 0.12])
            ax0b.fill_between(t, st, step="mid", alpha=0.35, color="#888888")
            ax0b.set_yticks([0, 1])
            ax0b.set_yticklabels(["L", "H"], fontsize=6)
            ax0b.set_xticks([])
        da = np.asarray(run["snap_Da"], dtype=np.float64)
        ax1.plot(t, da, lw=0.9, color="#9467bd")
        ax1.axhline(_DA_THRESHOLD, color="#888888", ls=":", lw=0.9, label="Da=1")
        ax1.set_ylabel("Da(t)")
        ax1.set_xlabel("time t")
        ax1.legend(fontsize=7)
        ax0.grid(True, alpha=0.25)
        ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D5 D5H analysis")
    parser.add_argument("d5h_sweep_dir", type=Path, help="D5H sweep timestamp directory")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    sweep_dir = args.d5h_sweep_dir.resolve()
    out_dir = (args.out_dir or sweep_dir).resolve()
    manifest = _load_manifest(sweep_dir)

    per_run: list[dict[str, Any]] = []
    plot_runs: list[dict[str, Any]] = []

    for entry in tqdm(manifest.get("entries", []), desc="D5H analyze", unit="run"):
        label = str(entry["run_label"])
        run_dir = _run_dir(sweep_dir, entry)
        cfg = load_run_config(run_dir / "config.yaml")
        phys = cfg["physics"]
        c1, c2, c_sat = float(phys["c1"]), float(phys["c2"]), float(phys["c_sat"])
        baseline = _resolve_baseline_physics(cfg)
        snap = {k: np.asarray(v) for k, v in np.load(run_dir / "snapshots.npz").items()}
        _gate_c_dep(snap, label=label)
        cdep = np.asarray(snap["snapshots_c_dep"], dtype=np.float64).ravel()
        t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
        states = [_state_str(s) for s in snap.get("snapshots_S", np.array([], dtype=object))]
        da = _compute_da_series(snap, D=baseline["D_abs"])
        if da.size != cdep.size:
            raise RuntimeError(f"{label}: Da series length {da.size} != c_dep {cdep.size}")
        da_report = _da_oscillation_report(states=states, c_dep=cdep, da=da, t=t, c1=c1, c2=c2)
        summary = json.loads((run_dir / "summary.json").read_text())
        mass = summary.get("mass_residual_pct")
        if mass is None or float(mass) >= 1e-4:
            raise RuntimeError(f"{label}: mass_residual_pct={mass} fails S1 gate")
        row: dict[str, Any] = {
            "run_label": label,
            "c1": c1,
            "c2": c2,
            "c_sat": c_sat,
            "mass_residual_pct": float(mass),
            **baseline,
            **da_report,
        }
        if da_report["oscillates"]:
            row["band_metrics"] = _band_metrics(run_dir, label)
        per_run.append(row)
        plot_runs.append(
            {
                **row,
                "snap_t": t,
                "snap_c_dep": cdep,
                "snap_S": states,
                "snap_Da": da,
            }
        )

    oscillators = [r for r in per_run if r["oscillates"]]
    if oscillators:
        master_run = max(oscillators, key=lambda r: r["n_cycles"])["run_label"]
        any_osc = True
    else:
        master_run = max(per_run, key=lambda r: r["crossings_up"] + r["crossings_down"])[
            "run_label"
        ]
        any_osc = False

    low_c0_slow_d = [r for r in per_run if r["c0"] <= 0.35 + 1e-9 and r["D_abs"] <= 0.03 + 1e-9]
    prediction_score = {
        "predicted": _PREDICTION,
        "observed": any(r["oscillates"] for r in low_c0_slow_d),
        "pass": any(r["oscillates"] for r in low_c0_slow_d),
        "candidate_runs": [r["run_label"] for r in low_c0_slow_d],
    }

    verdict = {
        "master_comparison_run": master_run,
        "any_oscillates": any_osc,
        "per_run": per_run,
        "prediction_score": prediction_score,
        "baseline_physics": {
            "source": "experiments/canonical/slab_twophase/baseline.yaml",
            "D_base": per_run[0]["D_base"] if per_run else None,
            "k_rxn_base": per_run[0]["k_rxn_base"] if per_run else None,
            "k_L_scale": 3.0,
            "kappa_rate": 5.0,
            "k_L_abs": per_run[0]["k_L_abs"] if per_run else None,
            "k_H_abs": per_run[0]["k_H_abs"] if per_run else None,
        },
        "root_cause_verdict": (
            "reaction-limited regime sustains Heaney clock on moving front"
            if any_osc
            else "reaction-limited sweep exhausted — no >=3-cycle oscillation; "
            "Da(t) traces show supply/consumption balance"
        ),
        "d5h_sweep_dir": str(sweep_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d5_part_a_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _plot_d5h_diag(out_dir / "figure_d5h_diag.png", plot_runs)

    link = _ROOT / "results/slab_twophase/d5h_heaney_reaction_limited/d5_part_a_verdict.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_file() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(out_dir / "d5_part_a_verdict.json")
    except OSError:
        import shutil

        shutil.copy2(out_dir / "d5_part_a_verdict.json", link)

    print(f"D5 analysis complete: {out_dir}")
    print(f"Master D5H row: {master_run} (oscillates={any_osc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
