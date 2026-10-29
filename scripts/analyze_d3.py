#!/usr/bin/env python3
"""EXP-D3 analysis: validity re-eval notes, Heaney moving-front diagnosis, report."""

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

_ROOT_CAUSE_HYPOTHESIS = (
    "Heaney clock needs c_sat < c2 < c1 so c_dep can complete the hysteresis downstroke "
    "while depositing; D2B used c_sat=0.2 > c2=0.10, flooring c_dep before H→L."
)
_PREDICTION = "D3H_fix1 oscillates (c_sat=0.05 < c2=0.10, poster c1/c2)."


def _load_manifest(sweep_dir: Path) -> dict[str, Any]:
    p = sweep_dir / "manifest_d3h.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_dir(sweep_dir: Path, entry: dict[str, Any]) -> Path:
    return (sweep_dir / str(entry["relative_path"])).resolve()


def _state_str(s: Any) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def _gate_c_dep(snap: dict[str, np.ndarray], *, label: str) -> dict[str, Any]:
    cdep = np.asarray(snap.get("snapshots_c_dep", []), dtype=np.float64).ravel()
    if cdep.size == 0:
        raise RuntimeError(f"{label}: missing snapshots_c_dep")
    bad = ~np.isfinite(cdep) | (cdep < -1e-9) | (cdep > 1.0 + 1e-9)
    if np.any(bad):
        idx = int(np.where(bad)[0][0])
        raise RuntimeError(
            f"{label}: c_dep gate FAIL frame {idx} value={cdep[idx]!r} (must be in [0,1])"
        )
    return {
        "n_frames": int(cdep.size),
        "c_dep_min": float(np.min(cdep)),
        "c_dep_max": float(np.max(cdep)),
    }


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


def _oscillation_persistence(
    gap: np.ndarray, cdep: np.ndarray, states: list[str], *, c2: float, c1: float
) -> dict[str, Any]:
    """Score whether oscillation survives as diffusion gap shrinks."""
    if gap.size < 4 or cdep.size < 4:
        return {"persists_at_small_gap": None, "last_cycle_gap": None}
    st = np.asarray([1.0 if _state_str(s) == "H" else 0.0 for s in states])
    crossings = [i for i in range(1, min(gap.size, st.size)) if st[i - 1] >= 0.5 and st[i] < 0.5]
    if not crossings:
        return {"persists_at_small_gap": False, "last_cycle_gap": float(gap[-1])}
    last_hl = int(crossings[-1])
    small_gap = float(gap[last_hl]) < 0.25 * float(gap[0]) if gap[0] > 0 else False
    i0 = max(0, last_hl // 2)
    late_osc = _count_lh_cycles(states[i0:], cdep[i0:], c1=c1, c2=c2)
    return {
        "persists_at_small_gap": bool(late_osc["n_cycles"] >= 1 and small_gap),
        "last_cycle_gap": float(gap[last_hl]),
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


def _plot_d3h_diag(path: Path, runs: list[dict[str, Any]]) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 3, figsize=(14, 2.6 * n), squeeze=False)
    for i, run in enumerate(runs):
        t = np.asarray(run["snap_t"], dtype=np.float64)
        ax0, ax1, ax2 = axes[i]
        cdep = np.asarray(run["snap_c_dep"], dtype=np.float64)
        ax0.plot(t, cdep, lw=0.9, color="0.15")
        ax0.axhline(run["c1"], color="#d62728", ls="--", lw=0.8)
        ax0.axhline(run["c2"], color="#1f77b4", ls="--", lw=0.8)
        ax0.axhline(run["c_sat"], color="#888888", ls=":", lw=0.8)
        osc = "OSC" if run["oscillates"] else "no osc"
        ax0.set_title(f"{run['run_label']} ({osc}, cycles={run['n_cycles']})")
        ax0.set_ylabel(r"$c_{\mathrm{dep}}$")
        if run.get("snap_S"):
            st = np.asarray([1.0 if _state_str(s) == "H" else 0.0 for s in run["snap_S"]])
            ax0b = ax0.inset_axes([0.0, 0.02, 1.0, 0.12])
            ax0b.fill_between(t, st, step="mid", alpha=0.35, color="#888888")
            ax0b.set_yticks([0, 1])
            ax0b.set_yticklabels(["L", "H"], fontsize=6)
            ax0b.set_xticks([])
        rate = run.get("snap_rate")
        supply = run.get("snap_supply")
        if rate is not None and supply is not None and t.size:
            ax1.plot(t, rate, label=r"$\sum R$", lw=0.8)
            ax1.plot(t, supply, label="front supply", lw=0.8, alpha=0.8)
            ax1.legend(fontsize=7)
            ax1.set_ylabel("rate")
        gap = run.get("snap_gap")
        if gap is not None and t.size:
            ax2.plot(t, gap, color="#2ca02c", lw=0.9)
            ax2.set_ylabel("gap width")
        ax2.set_xlabel("time t")
        ax0.grid(True, alpha=0.25)
        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    per_run: list[dict[str, Any]],
    master_run: str,
    any_osc: bool,
    prediction_score: dict[str, Any],
    validity_notes: dict[str, str],
) -> None:
    lines = [
        "# REPORT_D3 — validity fix + Heaney moving-front diagnosis",
        "",
        "## PART A — validity flag re-base",
        "- **VALID** iff `mass_residual_pct` < 1e-3 AND `min_phi_w` ≥ -1e-6.",
        "- `clipped_mass_frac` removed from hard INVALID; replaced by `clip_rate` WARN "
        f"(threshold={validity_notes['clip_rate_warn_threshold']}).",
        "- **E7 reconciliation:** baseline-Ostwald band-count expectation dropped; "
        "4 bands is acceptable (weak Ostwald gate still yields modest demixing).",
        "",
        "## PART B — root-cause hypothesis",
        f"- **Hypothesis:** {_ROOT_CAUSE_HYPOTHESIS}",
        f"- **Prediction:** {_PREDICTION}",
        f"- **Prediction score:** {prediction_score}",
        "",
        "### Per-run verdict",
    ]
    for r in per_run:
        verdict = "OSCILLATES" if r["oscillates"] else "no sustained oscillation"
        lines.append(
            f"- **{r['run_label']}**: {verdict} (cycles={r['n_cycles']}, "
            f"c_sat={r['c_sat']}, c2={r['c2']}, c_sat<c2={r['c_sat_lt_c2']}) "
            f"<!-- d3h.{r['run_label']}.verdict -->"
        )
        if r.get("band_metrics"):
            bm = r["band_metrics"]
            lines.append(
                f"  - band metrics: lambda={bm.get('lambda_band')}, tau={bm.get('tau_cycle')}, "
                f"d={bm.get('d_spacing')}, v_open={bm.get('v_at_open')} "
                f"<!-- d3h.{r['run_label']}.bands -->"
            )
        if r.get("persistence"):
            lines.append(
                f"  - gap persistence: {r['persistence']} <!-- d3h.{r['run_label']}.gap -->"
            )
    lines += [
        "",
        f"**Master comparison row:** `{master_run}` <!-- d3h.master_comparison_run -->",
        f"**Bracket outcome:** {'oscillation found' if any_osc else 'honest negative'}",
        "",
        "## PART C — Heaney terminology",
        "- Kinetic clock rows: `Heaney kinetic clock` (`slab_kinetic`)",
        "- Moving-front row: `Heaney moving front` (`slab_twophase` hysteresis)",
        "- Run labels: `E1_Heaney_open`, `E2_Heaney_control`, `E3_Heaney_closed`, `E9_Heaney_movingfront`",  # noqa: E501
        "",
        "## Master figure row index",
        "- See `experiments/comparisons/d3_master_comparison.yaml` display labels",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D3 analysis")
    parser.add_argument(
        "d3h_sweep_dir",
        type=Path,
        help="D3H sweep timestamp directory",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    sweep_dir = args.d3h_sweep_dir.resolve()
    out_dir = (args.out_dir or sweep_dir).resolve()
    manifest = _load_manifest(sweep_dir)

    per_run: list[dict[str, Any]] = []
    plot_runs: list[dict[str, Any]] = []

    for entry in tqdm(manifest.get("entries", []), desc="D3H analyze", unit="run"):
        label = str(entry["run_label"])
        run_dir = _run_dir(sweep_dir, entry)
        cfg = load_run_config(run_dir / "config.yaml")
        phys = cfg["physics"]
        c1, c2, c_sat = float(phys["c1"]), float(phys["c2"]), float(phys["c_sat"])
        snap = {k: np.asarray(v) for k, v in np.load(run_dir / "snapshots.npz").items()}
        _gate_c_dep(snap, label=label)
        cdep = np.asarray(snap["snapshots_c_dep"], dtype=np.float64).ravel()
        t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
        states = [_state_str(s) for s in snap.get("snapshots_S", np.array([], dtype=object))]
        cycles = _count_lh_cycles(states, cdep, c1=c1, c2=c2)
        gap = np.asarray(snap.get("snapshots_gap_width", []), dtype=np.float64).ravel()
        persistence = _oscillation_persistence(gap, cdep, states, c2=c2, c1=c1)
        summary = json.loads((run_dir / "summary.json").read_text())
        mass = summary.get("mass_residual_pct")
        if mass is None or float(mass) >= 1e-4:
            raise RuntimeError(f"{label}: mass_residual_pct={mass} fails S1 gate")
        row: dict[str, Any] = {
            "run_label": label,
            "c1": c1,
            "c2": c2,
            "c_sat": c_sat,
            "c_sat_lt_c2": bool(c_sat < c2),
            "mass_residual_pct": float(mass),
            **cycles,
            "persistence": persistence,
        }
        if cycles["oscillates"]:
            row["band_metrics"] = _band_metrics(run_dir, label)
        per_run.append(row)
        plot_runs.append(
            {
                **row,
                "snap_t": t,
                "snap_c_dep": cdep,
                "snap_S": states,
                "snap_rate": snap.get("snapshots_rate_total"),
                "snap_supply": snap.get("snapshots_supply_front"),
                "snap_gap": gap if gap.size else None,
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

    fix1 = next(r for r in per_run if r["run_label"] == "D3H_fix1")
    prediction_score = {
        "predicted": "D3H_fix1 oscillates",
        "observed": bool(fix1["oscillates"]),
        "pass": bool(fix1["oscillates"]),
    }

    verdict = {
        "master_comparison_run": master_run,
        "any_oscillates": any_osc,
        "per_run": per_run,
        "hypothesis": _ROOT_CAUSE_HYPOTHESIS,
        "prediction_score": prediction_score,
        "root_cause_verdict": (
            "c_sat>c2 blocked H→L in D2B; lowering c_sat restores cycles"
            if any_osc
            else "c_sat<c2 alone insufficient — supply/gap geometry may pin clock"
        ),
        "d3h_sweep_dir": str(sweep_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d3_part_b_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out_dir / "summary_d3.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _plot_d3h_diag(out_dir / "figure_d3h_diag.png", plot_runs)
    _write_report(
        out_dir / "REPORT_D3.md",
        per_run=per_run,
        master_run=master_run,
        any_osc=any_osc,
        prediction_score=prediction_score,
        validity_notes={"clip_rate_warn_threshold": "4e-5"},
    )

    link = _ROOT / "results/slab_twophase/d3h_heaney_diag/d3_part_b_verdict.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_file() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(out_dir / "d3_part_b_verdict.json")
    except OSError:
        import shutil

        shutil.copy2(out_dir / "d3_part_b_verdict.json", link)

    print(f"D3 analysis complete: {out_dir}")
    print(f"Master D3H row: {master_run} (oscillates={any_osc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
