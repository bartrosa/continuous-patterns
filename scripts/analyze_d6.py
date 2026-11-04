#!/usr/bin/env python3
"""EXP-D6 PART C analysis: bug-vs-physics gate, swing-vs-band verdict, D6H figure."""

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

_DA_THRESHOLD = 1.0
_PREDICTION = (
    "Oscillation iff c_dep swing (P95-P5) exceeds band width (c1-c2); "
    "narrow-band runs should oscillate."
)


def _load_manifest(sweep_dir: Path) -> dict[str, Any]:
    p = sweep_dir / "manifest_d6h.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_dir(sweep_dir: Path, entry: dict[str, Any]) -> Path:
    return (sweep_dir / str(entry["relative_path"])).resolve()


def _state_str(s: Any) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def _gate_c_dep(cdep: np.ndarray, *, label: str) -> None:
    if cdep.size == 0:
        raise RuntimeError(f"{label}: missing snapshots_c_dep")
    bad = ~np.isfinite(cdep) | (cdep < -1e-9) | (cdep > 1.0 + 1e-9)
    if np.any(bad):
        idx = int(np.where(bad)[0][0])
        raise RuntimeError(f"{label}: c_dep gate FAIL frame {idx} value={cdep[idx]!r}")


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


def _swing_amplitude(cdep: np.ndarray, *, skip_frac: float = 0.1) -> float:
    n = cdep.size
    if n < 4:
        return float(np.max(cdep) - np.min(cdep)) if n else 0.0
    i0 = int(n * skip_frac)
    i1 = max(i0 + 1, int(n * (1.0 - skip_frac)))
    window = cdep[i0:i1]
    return float(np.percentile(window, 95) - np.percentile(window, 5))


def _crosses_c2(cdep: np.ndarray, c2: float) -> bool:
    return bool(np.any(cdep >= c2))


def _crosses_c1(cdep: np.ndarray, c1: float) -> bool:
    return bool(np.any(cdep <= c1)) and bool(np.any(cdep >= c1))


def _c1_crossing_up(cdep: np.ndarray, c1: float) -> bool:
    for i in range(1, cdep.size):
        if cdep[i - 1] <= c1 < cdep[i]:
            return True
    return False


def _c2_crossing_down(cdep: np.ndarray, c2: float) -> bool:
    for i in range(1, cdep.size):
        if cdep[i - 1] >= c2 > cdep[i]:
            return True
    return False


def _state_update_bug_report(
    states: list[str], cdep: np.ndarray, *, c1: float, c2: float
) -> dict[str, Any]:
    """C1: c_dep crosses threshold but S does not flip -> state-update bug."""
    bugs: list[str] = []
    for i in range(1, min(len(states), cdep.size)):
        st = _state_str(states[i])
        prev = _state_str(states[i - 1])
        if cdep[i - 1] <= c1 < cdep[i] and st != "H" and prev != "H":
            bugs.append(f"frame {i}: c_dep up-cross c1 but S={st!r}")
        if cdep[i - 1] >= c2 > cdep[i] and st != "L" and prev != "L":
            bugs.append(f"frame {i}: c_dep down-cross c2 but S={st!r}")
    return {
        "state_update_bug": bool(bugs),
        "state_update_bug_notes": bugs[:5],
        "crosses_c2": _crosses_c2(cdep, c2),
        "crosses_c1": _crosses_c1(cdep, c1),
        "c1_crossing_up": _c1_crossing_up(cdep, c1),
        "c2_crossing_down": _c2_crossing_down(cdep, c2),
    }


def _resolve_baseline_physics(cfg: dict[str, Any]) -> dict[str, float]:
    base = load_run_config(_ROOT / "experiments/canonical/slab_twophase/baseline.yaml")
    bp = base["physics"]
    phys = cfg["physics"]
    k_rxn_base = float(bp["k_rxn"])
    D_base = float(bp["D_c"])
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


def _compute_da(snap: dict[str, np.ndarray], *, D: float) -> np.ndarray:
    if "snapshots_Da" in snap:
        return np.asarray(snap["snapshots_Da"], dtype=np.float64).ravel()
    gap = np.asarray(snap.get("snapshots_gap_width", []), dtype=np.float64).ravel()
    k_eff = np.asarray(snap.get("snapshots_k_eff", []), dtype=np.float64).ravel()
    if gap.size and k_eff.size == gap.size:
        return k_eff * gap / max(D, 1e-30)
    raise RuntimeError("missing snapshots_Da and cannot reconstruct")


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


def _frame_traces(snap: dict[str, np.ndarray]) -> dict[str, Any]:
    t = np.asarray(snap.get("snapshots_t", []), dtype=np.float64).ravel()
    cdep = np.asarray(snap.get("snapshots_c_dep", []), dtype=np.float64).ravel()
    states = [_state_str(s) for s in snap.get("snapshots_S", np.array([], dtype=object))]
    j_consume = np.asarray(snap.get("snapshots_consume_front", []), dtype=np.float64).ravel()
    j_supply = np.asarray(snap.get("snapshots_supply_front", []), dtype=np.float64).ravel()
    l_gap = np.asarray(snap.get("snapshots_gap_width", []), dtype=np.float64).ravel()
    da = np.asarray(snap.get("snapshots_Da", []), dtype=np.float64).ravel()
    xf = np.asarray(snap.get("snapshots_x_front", []), dtype=np.float64).ravel()
    advance = np.full_like(t, np.nan)
    if t.size > 1 and xf.size == t.size:
        advance[1:] = np.diff(xf) / np.diff(t)
    return {
        "t": t.tolist(),
        "c_dep": cdep.tolist(),
        "S": states,
        "J_consume": j_consume.tolist(),
        "J_supply": j_supply.tolist(),
        "L_gap": l_gap.tolist(),
        "Da": da.tolist(),
        "front_advance_rate": advance.tolist(),
    }


def _write_table_d6h(path: Path, per_run: list[dict[str, Any]], *, master_run: str) -> None:
    lines = [
        "# D6H Heaney moving-front sweep",
        "",
        "| run | c1 | c2 | band width | k_L | swing (P95-P5) | predicted | n_cycles | oscillates | master |",  # noqa: E501
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in per_run:
        lines.append(
            f"| {r['run_label']} | {r['c1']:.3g} | {r['c2']:.3g} | {r['band_width']:.3g} | "
            f"{r['k_L_abs']:.3g} | {r['swing_amplitude']:.4g} | "
            f"{'yes' if r['predicted_oscillates'] else 'no'} | {r['n_cycles']} | "
            f"{'yes' if r['oscillates'] else 'no'} | "
            f"{'**winner**' if r['run_label'] == master_run else ''} |"
        )
    lines += [
        "",
        "Prediction: oscillation iff swing > band width (c1-c2).",
        "",
        "## Da diagnostics",
        "",
        "| run | Da median | Da max | crosses Da=1 |",
        "| --- | --- | --- | --- |",
    ]
    for r in per_run:
        da = np.asarray(r.get("Da_trace", []), dtype=np.float64)
        med = float(np.median(da)) if da.size else float("nan")
        mx = float(np.max(da)) if da.size else float("nan")
        crosses = bool(np.any(da >= _DA_THRESHOLD)) if da.size else False
        lines.append(f"| {r['run_label']} | {med:.3g} | {mx:.3g} | {'yes' if crosses else 'no'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_d6h_diag(path: Path, runs: list[dict[str, Any]]) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.9 * n), squeeze=False)
    for i, run in enumerate(runs):
        t = np.asarray(run["snap_t"], dtype=np.float64)
        ax0, ax1 = axes[i]
        cdep = np.asarray(run["snap_c_dep"], dtype=np.float64)
        c1, c2 = run["c1"], run["c2"]
        bw = run["band_width"]
        swing = run["swing_amplitude"]
        ax0.plot(t, cdep, lw=0.9, color="0.15")
        ax0.axhline(c1, color="#d62728", ls="--", lw=0.8)
        ax0.axhline(c2, color="#1f77b4", ls="--", lw=0.8)
        mid = 0.5 * (c1 + c2)
        ax0.axhspan(
            mid - 0.5 * swing, mid + 0.5 * swing, alpha=0.12, color="#9467bd", label="swing"
        )
        ax0.axhspan(c2, c1, alpha=0.08, color="#2ca02c", label="band width")
        osc = "OSC" if run["oscillates"] else "no osc"
        ax0.set_title(
            f"{run['run_label']} ({osc}, swing={swing:.3f}, width={bw:.3f}, cycles={run['n_cycles']})"  # noqa: E501
        )
        ax0.legend(fontsize=6, loc="upper right")
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
    parser = argparse.ArgumentParser(description="EXP-D6 D6H analysis")
    parser.add_argument("d6h_sweep_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    sweep_dir = args.d6h_sweep_dir.resolve()
    out_dir = (args.out_dir or sweep_dir).resolve()
    manifest = _load_manifest(sweep_dir)

    per_run: list[dict[str, Any]] = []
    plot_runs: list[dict[str, Any]] = []
    any_state_bug = False

    for entry in tqdm(manifest.get("entries", []), desc="D6H analyze", unit="run"):
        label = str(entry["run_label"])
        run_dir = _run_dir(sweep_dir, entry)
        cfg = load_run_config(run_dir / "config.yaml")
        phys = cfg["physics"]
        c1, c2 = float(phys["c1"]), float(phys["c2"])
        band_width = c1 - c2
        baseline = _resolve_baseline_physics(cfg)
        snap = {k: np.asarray(v) for k, v in np.load(run_dir / "snapshots.npz").items()}
        cdep = np.asarray(snap["snapshots_c_dep"], dtype=np.float64).ravel()
        _gate_c_dep(cdep, label=label)
        t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
        states = [_state_str(s) for s in snap.get("snapshots_S", np.array([], dtype=object))]
        da = _compute_da(snap, D=baseline["D_abs"])
        bug = _state_update_bug_report(states, cdep, c1=c1, c2=c2)
        if bug["state_update_bug"]:
            any_state_bug = True
        swing = _swing_amplitude(cdep)
        cycles = _count_lh_cycles(states, cdep, c1=c1, c2=c2)
        predicted_osc = swing > band_width
        summary = json.loads((run_dir / "summary.json").read_text())
        mass = summary.get("mass_residual_pct")
        if mass is None or float(mass) >= 1e-4:
            raise RuntimeError(f"{label}: mass_residual_pct={mass} fails S1 gate")
        xf = np.asarray(snap.get("snapshots_x_front", []), dtype=np.float64).ravel()
        v_front = None
        if t.size > 2 and xf.size == t.size:
            v_front = float(np.median(np.diff(xf) / np.diff(t)))
        row: dict[str, Any] = {
            "run_label": label,
            "c1": c1,
            "c2": c2,
            "band_width": band_width,
            "swing_amplitude": swing,
            "predicted_oscillates": predicted_osc,
            "mass_residual_pct": float(mass),
            "v_front_median": v_front,
            **baseline,
            **bug,
            **cycles,
        }
        traces = _frame_traces(snap)
        row["Da_trace"] = traces["Da"]
        diag_path = out_dir / f"d6h_{label}_frame_diagnostics.json"
        diag_path.write_text(json.dumps({"run_label": label, **traces}, indent=2), encoding="utf-8")
        row["frame_diagnostics_path"] = str(diag_path)
        if cycles["oscillates"]:
            row["band_metrics"] = _band_metrics(run_dir, label)
        per_run.append(row)
        plot_runs.append({**row, "snap_t": t, "snap_c_dep": cdep, "snap_S": states, "snap_Da": da})

    c1_decision = (
        "state_update_bug detected — hysteresis state machine must be fixed before physics verdict"
        if any_state_bug
        else "state machine consistent — negative (if any) is physics/regime"
    )

    oscillators = [r for r in per_run if r["oscillates"]]
    if oscillators:
        master_run = max(oscillators, key=lambda r: r["n_cycles"])["run_label"]
        any_osc = True
    else:
        master_run = max(per_run, key=lambda r: r["swing_amplitude"])["run_label"]
        any_osc = False

    prediction_score = {
        "predicted": _PREDICTION,
        "per_run_pass": [
            {
                "run_label": r["run_label"],
                "predicted": r["predicted_oscillates"],
                "observed": r["oscillates"],
                "pass": r["predicted_oscillates"] == r["oscillates"],
            }
            for r in per_run
        ],
        "n_correct": sum(1 for r in per_run if r["predicted_oscillates"] == r["oscillates"]),
    }

    verdict = {
        "master_comparison_run": master_run,
        "any_oscillates": any_osc,
        "c1_decision": c1_decision,
        "any_state_update_bug": any_state_bug,
        "per_run": per_run,
        "prediction_score": prediction_score,
        "baseline_physics": {
            "source": "experiments/canonical/slab_twophase/baseline.yaml",
            "D_base": per_run[0]["D_base"] if per_run else None,
            "k_rxn_base": per_run[0]["k_rxn_base"] if per_run else None,
            "D_abs": 0.03 * float(per_run[0]["D_base"]) if per_run else None,
            "k_L_abs_values": {r["run_label"]: r["k_L_abs"] for r in per_run},
        },
        "root_cause_verdict": (
            "swing exceeds band width — Heaney clock sustains on moving front"
            if any_osc
            else "finite grid exhausted — swing vs band width table shows insufficient span"
        ),
        "d6h_sweep_dir": str(sweep_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d6_part_c_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _write_table_d6h(out_dir / "table_d6h.md", per_run, master_run=master_run)
    _plot_d6h_diag(out_dir / "figure_d6h_diag.png", plot_runs)

    print(f"D6H analysis complete: {out_dir}")
    print(f"C1 decision: {c1_decision}")
    print(f"Master D6H row: {master_run} (oscillates={any_osc})")
    if any_state_bug:
        raise RuntimeError(
            "C1 state-update bug detected — fix hysteresis kernel before campaign continues"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
