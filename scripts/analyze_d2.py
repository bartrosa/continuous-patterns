#!/usr/bin/env python3
"""EXP-D2 PART A analysis: c_dep readout gate, oscillation verdict, sawtooth figure, report."""

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

from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.experiments.checkpoint import is_run_complete  # noqa: E402

_POSTER_MAP = {
    "poster_Da": 1.0,
    "poster_kappa_rate": 3.0,
    "poster_c1": 0.49,
    "poster_c2": 0.10,
    "slab_k_rxn_base": 0.5,
    "mapping": (
        "UW poster reference: Da=1, kappa=3, c1=0.49, c2=0.10 map to "
        "slab_twophase hysteresis with D_c=1 (S1 transport), k_rxn=0.5 (S1 base), "
        "kappa_rate=3 (= poster kappa = k_H/k_L), k_L = k_L_scale × k_rxn, "
        "c1/c2 direct, gamma=0 M0=0 (psi recorder)."
    ),
}


def _load_manifest(sweep_dir: Path) -> dict[str, Any]:
    p = sweep_dir / "manifest_d2h.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_dir(sweep_dir: Path, entry: dict[str, Any]) -> Path:
    return (sweep_dir / str(entry["relative_path"])).resolve()


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
        "pass": True,
    }


def _state_str(s: Any) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def _count_lh_cycles(
    states: list[str], c_dep: np.ndarray, *, c1: float, c2: float
) -> dict[str, Any]:
    """Count clean L↔H cycles: L→H when c_dep crosses above c1, H→L when below c2."""
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


def _verify_e1_sawtooth(e1_run_dir: Path | None) -> dict[str, Any]:
    """Optional E1 check: c(x=0) in [0,1] and known sawtooth transitions."""
    if e1_run_dir is None or not is_run_complete(e1_run_dir):
        return {"status": "skipped", "reason": "E1 run dir not provided"}
    hist = e1_run_dir / "history.npz"
    if not hist.is_file():
        return {"status": "skipped", "reason": "no history.npz"}
    z = np.load(hist)
    c0 = np.asarray(z["c_at_x0_history"], dtype=np.float64)
    st = np.asarray(z["state_history"])
    if np.any(~np.isfinite(c0)) or np.any(c0 < -1e-9) or np.any(c0 > 1.0 + 1e-9):
        raise RuntimeError(f"E1 c(x=0) outside [0,1]: min={np.min(c0)} max={np.max(c0)}")
    n_tr = int(np.sum(np.diff(st.astype(int)) != 0))
    return {
        "status": "pass",
        "c0_min": float(np.min(c0)),
        "c0_max": float(np.max(c0)),
        "n_transitions": n_tr,
        "sawtooth_expected": n_tr >= 50,
    }


def _plot_d2h_sawtooth(
    path: Path,
    runs: list[dict[str, Any]],
) -> None:
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=False, squeeze=False)
    for i, run in enumerate(runs):
        ax_c = axes[i, 0]
        t = np.asarray(run["snap_t"], dtype=np.float64)
        cdep = np.asarray(run["snap_c_dep"], dtype=np.float64)
        ax_c.plot(t, cdep, lw=0.9, color="0.15")
        ax_c.axhline(run["c1"], color="#d62728", ls="--", lw=0.8, label=f"c1={run['c1']}")
        ax_c.axhline(run["c2"], color="#1f77b4", ls="--", lw=0.8, label=f"c2={run['c2']}")
        ax_c.set_ylabel(r"$c_{\mathrm{dep}}$")
        osc = "OSC" if run["oscillates"] else "no osc"
        ax_c.set_title(f"{run['run_label']}  ({osc}, cycles={run['n_cycles']})")
        ax_c.legend(fontsize=7, loc="upper right")
        ax_c.grid(True, alpha=0.25)
        if run.get("snap_S"):
            ax_s = ax_c.inset_axes([0.0, 0.02, 1.0, 0.12])
            st = np.asarray([1.0 if _state_str(s) == "H" else 0.0 for s in run["snap_S"]])
            ax_s.fill_between(t, st, step="mid", alpha=0.35, color="#888888")
            ax_s.set_yticks([0, 1])
            ax_s.set_yticklabels(["L", "H"], fontsize=6)
            ax_s.set_xticks([])
    axes[-1, 0].set_xlabel("time t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    per_run: list[dict[str, Any]],
    e1_check: dict[str, Any],
    master_run: str,
    any_osc: bool,
    master_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# REPORT_D2 — Heaney moving-front readout + master comparison index",
        "",
        "## A1 — c_dep readout fix",
        "- **Definition:** `c_dep` = silica concentration at the active front: last cell with "
        "`phi_w > 0.05`, with `c = u/phi_w` guarded against `phi_w → 0` in deposited cells.",
        "- **Implementation:** `twophase_kernel_d1.c_dep_at_active_front` / `c_from_u_guarded`.",
        f"- **E1 verification:** {e1_check}",
        "",
        "## A2 — Poster unit mapping",
        f"- {_POSTER_MAP['mapping']} <!-- poster.mapping -->",
        f"- poster c1={_POSTER_MAP['poster_c1']}, c2={_POSTER_MAP['poster_c2']}, "
        f"kappa_rate={_POSTER_MAP['poster_kappa_rate']} <!-- poster.params -->",
        "",
        "## A4 — PART A verdict (pre-declared bracket)",
    ]
    for r in per_run:
        verdict = "OSCILLATES" if r["oscillates"] else "no sustained oscillation"
        lines.append(
            f"- **{r['run_label']}**: {verdict} "
            f"(cycles={r['n_cycles']}, c_dep∈[{r['c_dep_min']:.4g},{r['c_dep_max']:.4g}]) "
            f"<!-- d2h.{r['run_label']}.verdict -->"
        )
    lines += [
        "",
        f"**Master row selection:** `{master_run}` <!-- d2h.master_comparison_run -->",
        f"**Bracket outcome:** {'at least one run oscillates' if any_osc else 'honest negative — no run reached 3 L↔H cycles'}",  # noqa: E501
        "",
        "## PART B — master figure row index",
    ]
    for row in master_rows:
        lines.append(
            f"- **{row['run_label']}** ({row['display']}): {row['class']} — {row['key']} "
            f"<!-- master.{row['run_label']} -->"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D2 PART A analysis")
    parser.add_argument(
        "d2h_sweep_dir",
        type=Path,
        help="D2H sweep timestamp directory (contains manifest_d2h.json)",
    )
    parser.add_argument(
        "--e1-run-dir",
        type=Path,
        default=None,
        help="Optional E1 run dir for sawtooth verification (history.npz)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write outputs here (default: d2h_sweep_dir)",
    )
    args = parser.parse_args(argv)

    sweep_dir = args.d2h_sweep_dir.resolve()
    out_dir = (args.out_dir or sweep_dir).resolve()
    manifest = _load_manifest(sweep_dir)

    per_run: list[dict[str, Any]] = []
    plot_runs: list[dict[str, Any]] = []

    for entry in tqdm(manifest.get("entries", []), desc="D2H analyze", unit="run"):
        label = str(entry["run_label"])
        run_dir = _run_dir(sweep_dir, entry)
        cfg = load_run_config(run_dir / "config.yaml")
        phys = cfg["physics"]
        c1, c2 = float(phys["c1"]), float(phys["c2"])
        snap = {k: np.asarray(v) for k, v in np.load(run_dir / "snapshots.npz").items()}
        gate = _gate_c_dep(snap, label=label)
        cdep = np.asarray(snap["snapshots_c_dep"], dtype=np.float64).ravel()
        t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
        states = [_state_str(s) for s in snap.get("snapshots_S", np.array([], dtype=object))]
        cycles = _count_lh_cycles(states, cdep, c1=c1, c2=c2)
        summary = json.loads((run_dir / "summary.json").read_text())
        mass = summary.get("mass_residual_pct")
        if mass is None or float(mass) >= 1e-4:
            raise RuntimeError(f"{label}: mass_residual_pct={mass} fails S1 gate")
        row = {
            "run_label": label,
            "c1": c1,
            "c2": c2,
            "k_L_scale": float(phys.get("k_L_scale", 1.0)),
            "kappa_rate": float(phys.get("kappa_rate", 1.0)),
            **gate,
            **cycles,
            "mass_residual_pct": float(mass),
        }
        per_run.append(row)
        plot_runs.append({**row, "snap_t": t, "snap_c_dep": cdep, "snap_S": states})

    oscillators = [r for r in per_run if r["oscillates"]]
    if oscillators:
        master_run = max(oscillators, key=lambda r: r["n_cycles"])["run_label"]
        any_osc = True
    else:
        master_run = max(per_run, key=lambda r: r["crossings_up"] + r["crossings_down"])[
            "run_label"
        ]
        any_osc = False

    e1_check = _verify_e1_sawtooth(args.e1_run_dir.resolve() if args.e1_run_dir else None)

    verdict = {
        "master_comparison_run": master_run,
        "any_oscillates": any_osc,
        "per_run": per_run,
        "e1_verification": e1_check,
        "poster_mapping": _POSTER_MAP,
        "d2h_sweep_dir": str(sweep_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d2_part_a_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out_dir / "summary_d2.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _plot_d2h_sawtooth(out_dir / "figure_d2h_sawtooth.png", plot_runs)

    master_rows = [
        {
            "run_label": "E1_Heaney_open",
            "display": "E1",
            "class": "Heaney kinetic",
            "key": "open sawtooth reference",
        },
        {
            "run_label": "E2_Heaney_control",
            "display": "E2",
            "class": "Heaney kinetic",
            "key": "poster control (stuck)",
        },
        {
            "run_label": "E3_Heaney_closed",
            "display": "E3",
            "class": "Heaney kinetic",
            "key": "closed fixed-gap",
        },
        {
            "run_label": "E9_Heaney_movingfront",
            "display": master_run,
            "class": "Heaney moving front",
            "key": "oscillates" if any_osc else "no sustained oscillation",
        },
        {
            "run_label": "E10–E13",
            "display": "D1A v-sweep",
            "class": "CH driven",
            "key": "lambda vs v (ingested)",
        },
        {
            "run_label": "E14–E15",
            "display": "D1A f15/f50",
            "class": "CH driven",
            "key": "duty vs f (ingested)",
        },
        {
            "run_label": "E7/E8",
            "display": "E7/E8",
            "class": "CH gated Ostwald",
            "key": "S1 mechanism",
        },
        {
            "run_label": "E16/E17",
            "display": "D1C g4/g8",
            "class": "CH closed",
            "key": "rind→core (ingested)",
        },
    ]
    _write_report(
        out_dir / "REPORT_D2.md",
        per_run=per_run,
        e1_check=e1_check,
        master_run=master_run,
        any_osc=any_osc,
        master_rows=master_rows,
    )

    # symlink verdict for master yaml default path
    verdict_link = _ROOT / "results/slab_twophase/d2h_heaney_movingfront/d2_part_a_verdict.json"
    verdict_link.parent.mkdir(parents=True, exist_ok=True)
    if verdict_link.is_file() or verdict_link.is_symlink():
        verdict_link.unlink()
    try:
        verdict_link.symlink_to(out_dir / "d2_part_a_verdict.json")
    except OSError:
        import shutil

        shutil.copy2(out_dir / "d2_part_a_verdict.json", verdict_link)

    print(f"D2 analysis complete: {out_dir}")
    print(f"Master D2H row: {master_run} (oscillates={any_osc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
