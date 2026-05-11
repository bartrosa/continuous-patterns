"""Post-process a slab_kinetic parametric sweep directory.

Reads ``manifest_axes.json``, writes ``table_all_runs.md``,
``figure_sweep_summary.png``, and ``figure_szymczak_replication.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

PAIR_LIST: list[tuple[float, float]] = [
    (0.49, 0.10),
    (0.40, 0.26),
    (0.25, 0.10),
    (0.50, 0.15),
    (0.33, 0.20),
    (0.40, 0.30),
    (0.45, 0.30),
    (0.45, 0.35),
    (0.50, 0.40),
    (0.30, 0.05),
    (0.40, 0.15),
]

DA_AXIS = [0.5, 1.0, 2.0, 5.0]
KAPPA_AXIS = [1.5, 3.0, 5.0, 10.0]

POSTER5 = PAIR_LIST[:5]


def _pair_index(c1: float, c2: float) -> int | None:
    for i, (a, b) in enumerate(PAIR_LIST):
        if abs(float(c1) - a) < 1e-6 and abs(float(c2) - b) < 1e-6:
            return i
    return None


def oscillation_window(Da: float, kappa: float, c1: float, c2: float) -> bool:
    lo = 1.0 / (1.0 + float(kappa) * float(Da))
    hi = 1.0 / (1.0 + float(Da))
    return lo < float(c2) < float(c1) < hi


def _fmt_bool_pl(yes: bool) -> str:
    return "TAK" if yes else "NIE"


def _collect_entries(manifest: dict) -> list[dict]:
    return [e for e in manifest.get("entries", []) if e.get("status") == "success"]


def write_table_all_runs(sweep_dir: Path, manifest: dict) -> None:
    rows: list[dict[str, object]] = []
    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        Da = float(e.get("Da", float("nan")))
        ka = float(e.get("kappa", float("nan")))
        c1 = float(e.get("c1", float("nan")))
        c2 = float(e.get("c2", float("nan")))
        T = float(e.get("T", float("nan")))
        n_tr = int(e.get("n_transitions", 0))
        osc_p = bool(e.get("osc_predicted", oscillation_window(Da, ka, c1, c2)))
        osc_o = bool(e.get("osc_observed", n_tr >= 2))
        rows.append(
            {
                "run_label": e.get("run_label", ""),
                "Da": Da,
                "kappa": ka,
                "c1": c1,
                "c2": c2,
                "T": T,
                "n_transitions": n_tr,
                "dwell_L_mean": e.get("dwell_L_mean"),
                "dwell_H_mean": e.get("dwell_H_mean"),
                "peak_period": e.get("peak_period"),
                "thickness_final": e.get("thickness_final"),
                "wall_time_s": e.get("wall_time_s"),
                "osc_predicted": osc_p,
                "osc_observed": osc_o,
                "osc_match": bool(e.get("osc_match", osc_p == osc_o)),
            }
        )
    rows.sort(key=lambda r: (r["Da"], r["kappa"], r["c1"], r["c2"]))

    lines = [
        "| run_label | Da | κ | c1 | c2 | T | n_transitions | dwell_L_mean | dwell_H_mean | "
        "peak_period | thickness_final | wall_time_s | osc_predicted | osc_observed | osc_match |",
        "|-----------|----|---|----|----|---|---------------|--------------|--------------|"
        "-------------|-----------------|-------------|---------------|--------------|-----------|",
    ]
    for r in rows:
        dlm = float(r["dwell_L_mean"]) if r["dwell_L_mean"] is not None else float("nan")
        dhm = float(r["dwell_H_mean"]) if r["dwell_H_mean"] is not None else float("nan")
        pp = float(r["peak_period"]) if r["peak_period"] is not None else float("nan")
        tf = float(r["thickness_final"]) if r["thickness_final"] is not None else float("nan")
        wt = float(r["wall_time_s"]) if r["wall_time_s"] is not None else float("nan")
        lines.append(
            f"| {r['run_label']} | {r['Da']} | {r['kappa']} | {r['c1']} | {r['c2']} | {r['T']} | "
            f"{r['n_transitions']} | {dlm:.6g} | {dhm:.6g} | {pp:.6g} | {tf:.6g} | {wt:.4g} | "
            f"{_fmt_bool_pl(bool(r['osc_predicted']))} | {_fmt_bool_pl(bool(r['osc_observed']))} | "
            f"{_fmt_bool_pl(bool(r['osc_match']))} |"
        )
    (sweep_dir / "table_all_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pair_matrices(entries: list[dict], field: str) -> np.ndarray:
    """Shape (4, 4, 3, 4): Da idx, κ idx, row, col in pair mini-grid."""
    mat = np.full((4, 4, 3, 4), np.nan, dtype=np.float64)
    for e in entries:
        Da = float(e.get("Da", float("nan")))
        ka = float(e.get("kappa", float("nan")))
        c1 = float(e.get("c1", float("nan")))
        c2 = float(e.get("c2", float("nan")))
        i_d = next((i for i, v in enumerate(DA_AXIS) if abs(v - Da) < 1e-6), None)
        i_k = next((i for i, v in enumerate(KAPPA_AXIS) if abs(v - ka) < 1e-6), None)
        pi = _pair_index(c1, c2)
        if i_d is None or i_k is None or pi is None:
            continue
        r, c = divmod(pi, 4)
        val = e.get(field)
        n_tr = int(e.get("n_transitions", 0))
        if field == "peak_period":
            if n_tr < 2:
                mat[i_d, i_k, r, c] = np.nan
            elif val is not None:
                mat[i_d, i_k, r, c] = float(val)
            continue
        if val is None:
            continue
        mat[i_d, i_k, r, c] = float(val)
    return mat


def plot_sweep_summary(sweep_dir: Path, manifest: dict) -> None:
    entries = _collect_entries(manifest)
    nt = build_pair_matrices(entries, "n_transitions")
    pp_raw = build_pair_matrices(entries, "peak_period")

    nt_max = float(np.nanmax(nt))
    if not np.isfinite(nt_max) or nt_max <= 0.0:
        nt_max = 1.0

    pp_plot = np.where(nt < 2, np.nan, pp_raw)
    pp_max = float(np.nanmax(pp_plot))
    if not np.isfinite(pp_max) or pp_max <= 0.0:
        pp_max = 1.0

    cmap_nt = plt.get_cmap("viridis").copy()
    cmap_nt.set_bad("white")
    cmap_pp = plt.get_cmap("plasma").copy()
    cmap_pp.set_bad("lightgray")

    # Tick labels: middle-row c2 per column, first-column c1 per row (pair mini-grid)
    xtick_labels = [f"{PAIR_LIST[4 + c][1]:.2f}" for c in range(4)]
    ytick_labels = [f"{PAIR_LIST[r * 4][0]:.2f}" for r in range(3)]

    fig = plt.figure(figsize=(18.0, 22.0))
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.28)

    for row, use_pp in enumerate((False, True)):
        gs_blk = gridspec.GridSpecFromSubplotSpec(
            4, 4, subplot_spec=gs_outer[row], wspace=0.38, hspace=0.48
        )
        ims: list = []
        axes_blk: list = []
        block_title = "peak_period" if use_pp else r"$n_{\mathrm{transitions}}$"
        vmax = pp_max if use_pp else nt_max
        cmap = cmap_pp if use_pp else cmap_nt
        for i in range(4):
            for j in range(4):
                ax_ij = fig.add_subplot(gs_blk[i, j])
                axes_blk.append(ax_ij)
                slot_nt = np.asarray(nt[i, j], dtype=np.float64)
                if use_pp:
                    slot = np.where(slot_nt < 2, np.nan, np.asarray(pp_raw[i, j], dtype=np.float64))
                    plot_arr = np.ma.masked_invalid(slot)
                else:
                    plot_arr = np.ma.masked_invalid(slot_nt)
                im = ax_ij.imshow(
                    plot_arr,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ims.append(im)
                ax_ij.set_title(f"Da={DA_AXIS[i]}, κ={KAPPA_AXIS[j]}", fontsize=9)
                ax_ij.set_xticks(np.arange(4))
                ax_ij.set_xticklabels(xtick_labels, fontsize=7, rotation=35, ha="right")
                ax_ij.set_yticks(np.arange(3))
                ax_ij.set_yticklabels(ytick_labels, fontsize=7)
                ax_ij.set_xlabel(r"$c_2$", fontsize=8)
                ax_ij.set_ylabel(r"$c_1$", fontsize=8)
        fig.colorbar(ims[-1], ax=axes_blk, shrink=0.65, label=block_title)

    fig.suptitle(
        "slab_kinetic parametric sweep — 3×4 pair grid (11 pairs + pad)",
        fontsize=13,
    )
    out = sweep_dir / "figure_sweep_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_szymczak_replication(sweep_dir: Path, manifest: dict) -> None:
    entries = _collect_entries(manifest)
    fig, axes = plt.subplots(2, 5, figsize=(22.0, 9.0), sharey=True)

    for Da in (1.0, 2.0):
        row = 0 if Da == 1.0 else 1
        for col, (tc1, tc2) in enumerate(POSTER5):
            ax = axes[row, col]
            hit = None
            for e in entries:
                if abs(float(e.get("Da", -1)) - Da) > 1e-6:
                    continue
                if abs(float(e.get("kappa", -1)) - 3.0) > 1e-6:
                    continue
                if abs(float(e.get("c1", -1)) - tc1) > 1e-6:
                    continue
                if abs(float(e.get("c2", -1)) - tc2) > 1e-6:
                    continue
                hit = e
                break
            if hit is None:
                ax.text(0.5, 0.5, "missing run", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            rel = hit.get("relative_path")
            if not isinstance(rel, str):
                ax.text(0.5, 0.5, "no path", ha="center", transform=ax.transAxes)
                continue
            run_dir = sweep_dir / rel
            hist_p = run_dir / "history.npz"
            summ_p = run_dir / "summary.json"
            if not hist_p.is_file():
                ax.text(0.5, 0.5, "no history", ha="center", transform=ax.transAxes)
                continue
            H = np.load(hist_p)
            tt = np.asarray(H["t_history"], dtype=np.float64).ravel()
            c0 = np.asarray(H["c_at_x0_history"], dtype=np.float64).ravel()
            st = np.asarray(H["state_history"], dtype=np.int32).ravel()
            summ_j = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.is_file() else {}
            tr_raw = summ_j.get("kinetic_transitions", [])
            transitions: list[tuple[float, str]] = []
            if isinstance(tr_raw, list):
                for item in tr_raw:
                    if isinstance(item, dict):
                        transitions.append((float(item["t"]), str(item["kind"])))
            dt_med = float(np.median(np.diff(tt))) if tt.size > 1 else 1.0
            i = 0
            while i < tt.size:
                j = i
                while j + 1 < tt.size and int(st[j + 1]) == int(st[i]):
                    j += 1
                t_lo = float(tt[i]) - 0.5 * dt_med
                t_hi = float(tt[j]) + 0.5 * dt_med
                col_b = "#cfe8ff" if int(st[i]) == 0 else "#ffd6d6"
                ax.axvspan(t_lo, t_hi, facecolor=col_b, alpha=0.55, zorder=0)
                i = j + 1
            ax.plot(tt, c0, "k-", lw=1.0, zorder=2)
            ax.axhline(float(tc1), color="tab:red", ls="--", lw=1.0)
            ax.axhline(float(tc2), color="tab:blue", ls="--", lw=1.0)
            for tv, _k in transitions:
                ix = int(np.argmin(np.abs(tt - float(tv))))
                ax.plot(float(tt[ix]), float(c0[ix]), "o", ms=3, zorder=3)
            n_tr = int(summ_j.get("n_transitions", 0))
            ax.set_title(f"Da={Da:g}, c1={tc1:g}, c2={tc2:g}, n_trans={n_tr}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 1:
                ax.set_xlabel(r"$t$")
            if col == 0:
                ax.set_ylabel(r"$c(0,t)$")

    fig.suptitle("Szymczak poster pairs × Da ∈ {1, 2}, κ = 3", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(sweep_dir / "figure_szymczak_replication.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "Usage: uv run python scripts/generate_slab_kinetic_summary_figures.py "
            "<sweep_results_dir>",
            file=sys.stderr,
        )
        return 2
    sweep_dir = Path(args[0]).resolve()
    man_path = sweep_dir / "manifest_axes.json"
    if not man_path.is_file():
        print(f"Missing {man_path}", file=sys.stderr)
        return 1
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    write_table_all_runs(sweep_dir, manifest)
    plot_sweep_summary(sweep_dir, manifest)
    plot_szymczak_replication(sweep_dir, manifest)

    ok = [e for e in manifest.get("entries", []) if e.get("status") == "success"]
    matches = [e for e in ok if e.get("osc_match") is True]
    rate = 100.0 * len(matches) / max(len(ok), 1)
    stats_line = (
        f"osc_match_rate_pct={rate:.2f} (n_success={len(ok)}, "
        f"n_match={len(matches)}) total_wall_s={manifest.get('total_wall_time_s')}"
    )
    (sweep_dir / "sweep_stats.txt").write_text(stats_line + "\n", encoding="utf-8")
    print(stats_line)
    print(f"Wrote table and figures under {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
