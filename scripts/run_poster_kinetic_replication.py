#!/usr/bin/env python3
"""Fresh UW poster kinetic replication (12 slab_kinetic runs + analytical panels)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

_REPO = Path(__file__).resolve().parents[1]

from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.core.poster_kinetic_analytical import (  # noqa: E402
    POSTER_C_PAIRS,
    AnalyticalGrid,
    compute_T_grid,
    grid_metadata,
    poster_formula_notes,
    simulate_semiinfinite_green,
)
from continuous_patterns.experiments.run import run_one  # noqa: E402

# 12 fresh runs (see task spec).
RUN_SPECS: list[dict[str, Any]] = [
    {"name": "Da1_c1_0p49_c2_0p10", "Da": 1.0, "kappa": 3.0, "c1": 0.49, "c2": 0.10},
    {"name": "Da1_c1_0p40_c2_0p26", "Da": 1.0, "kappa": 3.0, "c1": 0.40, "c2": 0.26},
    {"name": "Da1_c1_0p25_c2_0p10", "Da": 1.0, "kappa": 3.0, "c1": 0.25, "c2": 0.10},
    {"name": "Da1_c1_0p50_c2_0p15", "Da": 1.0, "kappa": 3.0, "c1": 0.50, "c2": 0.15},
    {"name": "Da1_c1_0p33_c2_0p20", "Da": 1.0, "kappa": 3.0, "c1": 0.33, "c2": 0.20},
    {"name": "Da1_c1_0p49_c2_0p26", "Da": 1.0, "kappa": 3.0, "c1": 0.49, "c2": 0.26},
    {"name": "Da2_c1_0p49_c2_0p10", "Da": 2.0, "kappa": 3.0, "c1": 0.49, "c2": 0.10},
    {"name": "Da2_c1_0p40_c2_0p26", "Da": 2.0, "kappa": 3.0, "c1": 0.40, "c2": 0.26},
    {"name": "Da2_c1_0p25_c2_0p10", "Da": 2.0, "kappa": 3.0, "c1": 0.25, "c2": 0.10},
    {"name": "Da2_c1_0p50_c2_0p15", "Da": 2.0, "kappa": 3.0, "c1": 0.50, "c2": 0.15},
    {"name": "Da2_c1_0p33_c2_0p20", "Da": 2.0, "kappa": 3.0, "c1": 0.33, "c2": 0.20},
    {"name": "Da1_c1_0p49_c2_0p35", "Da": 1.0, "kappa": 3.0, "c1": 0.49, "c2": 0.35},
]

PANEL_A = [(0.49, 0.10, "tab:red"), (0.40, 0.26, "tab:blue")]
PANEL_B = [
    (0.49, 0.10, "tab:red"),
    (0.40, 0.26, "tab:blue"),
    (0.25, 0.10, "tab:orange"),
    (0.50, 0.15, "tab:green"),
    (0.33, 0.20, "tab:purple"),
]
F2_X_START = (0.0, 0.25, 0.5, 0.75, 1.0)


def _run_dir(root: Path, spec: dict[str, Any]) -> Path:
    rel = spec.get("relative_path")
    if isinstance(rel, str) and rel:
        path = root / rel
        if path.is_dir():
            return path
    base = root / "numerical_runs" / str(spec["name"])
    if base.is_dir():
        subs = sorted(p for p in base.iterdir() if p.is_dir())
        if subs:
            inner = subs[-1]
            if inner.is_dir():
                ts_dirs = sorted(p for p in inner.iterdir() if p.is_dir())
                if ts_dirs:
                    return ts_dirs[-1]
            return inner
    raise FileNotFoundError(f"missing run folder for {spec.get('name')}")


def _load_history(run_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    hist = np.load(run_path / "history.npz")
    summ = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    return (
        np.asarray(hist["t_history"], dtype=np.float64).ravel(),
        np.asarray(hist["c_at_x0_history"], dtype=np.float64).ravel(),
        summ,
    )


def _load_snapshots(run_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sn = np.load(run_path / "snapshots.npz")
    return (
        np.asarray(sn["snapshots_t"], dtype=np.float64).ravel(),
        np.asarray(sn["snapshots_c"], dtype=np.float64),
    )


def _lookup_spec(root: Path, Da: float, c1: float, c2: float) -> dict[str, Any]:
    for spec in RUN_SPECS:
        if (
            abs(spec["Da"] - Da) < 1e-9
            and abs(spec["c1"] - c1) < 1e-9
            and abs(spec["c2"] - c2) < 1e-9
        ):
            out = dict(spec)
            man_path = root / "manifest.json"
            if man_path.is_file():
                man = json.loads(man_path.read_text(encoding="utf-8"))
                for row in man.get("runs", []):
                    if row.get("name") == spec["name"]:
                        out["relative_path"] = row.get("relative_path")
                        break
            return out
    raise KeyError(f"no run spec for Da={Da}, c1={c1}, c2={c2}")


def _plot_c0_panel(
    ax: plt.Axes,
    root: Path,
    curves: list[tuple[float, float, str, float]],
    *,
    Da: float,
    t_max: float = 5.0,
) -> None:
    for c1, c2, color, _da in curves:
        spec = _lookup_spec(root, Da, c1, c2)
        rdir = _run_dir(root, spec)
        tt, c0, summ = _load_history(rdir)
        mask = tt <= t_max + 1e-12
        ax.plot(tt[mask], c0[mask], color=color, lw=1.2, label=f"c1={c1:g}, c2={c2:g}")
        ax.axhline(c1, color=color, ls="--", lw=0.9, alpha=0.7)
        ax.axhline(c2, color=color, ls=":", lw=0.9, alpha=0.7)
        tr = summ.get("kinetic_transitions") or []
        if isinstance(tr, list):
            for item in tr:
                if isinstance(item, dict):
                    tv = float(item["t"])
                    if tv <= t_max:
                        ix = int(np.argmin(np.abs(tt - tv)))
                        ax.plot(tt[ix], c0[ix], "o", ms=3, color=color, zorder=3)
    ax.set_xlim(0.0, t_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$c(x=0,\,t)$")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)


def _plot_heatmap_panel(
    ax: plt.Axes,
    root: Path,
    spec_name: str,
    title: str,
) -> None:
    spec = next(r for r in RUN_SPECS if r["name"] == spec_name)
    man_path = root / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for row in man.get("runs", []):
            if row.get("name") == spec_name:
                spec = {**spec, "relative_path": row.get("relative_path")}
                break
    rdir = _run_dir(root, spec)
    sn_t, sn_c = _load_snapshots(rdir)
    im = ax.imshow(
        sn_c,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 1.0, float(sn_t[0]), float(sn_t[-1])),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _build_analytical(root: Path) -> AnalyticalGrid:
    ana_dir = root / "analytical"
    ana_dir.mkdir(parents=True, exist_ok=True)
    Da_vals = np.linspace(0.5, 5.0, 30)
    kappa_vals = np.linspace(1.5, 10.0, 30)
    grid = compute_T_grid(Da_vals, kappa_vals, POSTER_C_PAIRS)
    np.savez_compressed(
        ana_dir / "T1_T2_grid.npz",
        Da=Da_vals,
        kappa=kappa_vals,
        pairs=np.asarray(POSTER_C_PAIRS, dtype=np.float64),
        T1=grid.T1,
        T2=grid.T2,
        period=grid.period(),
        formula_notes=poster_formula_notes(),
    )
    (ana_dir / "formula_notes.txt").write_text(poster_formula_notes(), encoding="utf-8")
    return grid


def _build_green_F1(root: Path) -> None:
    tt, c0, _xg = simulate_semiinfinite_green(
        Da=1.0,
        kappa=3.0,
        c1=0.49,
        c2=0.35,
        T_final=2.0,
        dt_out=0.002,
    )
    np.savez_compressed(root / "analytical" / "green_function_F1.npz", t=tt, c_at_0=c0)


def _run_all_numerical(root: Path) -> tuple[list[dict[str, Any]], float]:
    num_root = root / "numerical_runs"
    num_root.mkdir(parents=True, exist_ok=True)
    base = load_run_config(_REPO / "experiments/canonical/slab_kinetic/baseline.yaml")
    rows: list[dict[str, Any]] = []
    t0_all = time.perf_counter()
    for spec in RUN_SPECS:
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = str(spec["name"])
        cfg.setdefault("physics", {})
        cfg["physics"]["Da"] = float(spec["Da"])
        cfg["physics"]["kappa"] = float(spec["kappa"])
        cfg["physics"]["c1"] = float(spec["c1"])
        cfg["physics"]["c2"] = float(spec["c2"])
        cfg.setdefault("time", {})
        cfg["time"]["T"] = 5.0
        cfg["time"]["dt_safety"] = 0.4
        cfg["time"]["snapshot_every"] = 400
        cfg.setdefault("output", {})
        cfg["output"]["include_params_panel"] = False
        cfg["output"]["log_level"] = "WARNING"

        t0 = time.perf_counter()
        result = run_one(
            cfg,
            results_root=num_root,
            chunk_size=2000,
            write_artifacts=True,
            show_progress=False,
            log_level="WARNING",
        )
        wall = time.perf_counter() - t0
        assert result.paths is not None
        diag = result.diagnostics
        rows.append(
            {
                "name": spec["name"],
                "Da": spec["Da"],
                "kappa": spec["kappa"],
                "c1": spec["c1"],
                "c2": spec["c2"],
                "relative_path": result.paths.root.relative_to(root).as_posix(),
                "n_transitions": int(diag.get("n_transitions", 0)),
                "mean_dwell_period": diag.get("mean_dwell_period"),
                "peak_period": diag.get("peak_period"),
                "thickness_final": diag.get("thickness_final"),
                "wall_time_s": wall,
            }
        )
    total_wall = time.perf_counter() - t0_all
    return rows, total_wall


def _panel_E_heatmaps(root: Path, grid: AnalyticalGrid) -> None:
    panels = root / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    Da = grid.Da
    ka = grid.kappa
    extent = [float(Da[0]), float(Da[-1]), float(ka[0]), float(ka[-1])]

    for row_name, arr in ("T1", grid.T1), ("T2", grid.T2):
        fig, axes = plt.subplots(1, len(POSTER_C_PAIRS), figsize=(14, 2.8), sharey=True)
        if len(POSTER_C_PAIRS) == 1:
            axes = [axes]
        for ip, (c1, c2) in enumerate(POSTER_C_PAIRS):
            ax = axes[ip]
            data = arr[ip]
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
            )
            ax.set_title(f"c1={c1:g}, c2={c2:g}", fontsize=8)
            ax.set_xlabel(r"$Da$")
            if ip == 0:
                ax.set_ylabel(r"$\kappa$")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(f"Analytical {row_name}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out = panels / f"panel_E_{row_name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)


def _panel_E_scatter(root: Path, grid: AnalyticalGrid) -> None:
    panels = root / "panels"
    c1_ref, c2_ref = 0.40, 0.26
    ip = POSTER_C_PAIRS.index((c1_ref, c2_ref))
    period_grid = grid.period()[ip]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ik = int(np.argmin(np.abs(grid.kappa - 3.0)))
    for ida, da in enumerate(grid.Da):
        p = period_grid[ik, ida]
        if np.isfinite(p):
            ax.scatter(float(da), float(p), s=18, c="C0", alpha=0.35, label="_ana grid")

    num_x: list[float] = []
    num_y: list[float] = []
    for spec in RUN_SPECS:
        if abs(spec["c1"] - c1_ref) > 1e-9 or abs(spec["c2"] - c2_ref) > 1e-9:
            continue
        rdir = _run_dir(root, spec)
        summ = json.loads((rdir / "summary.json").read_text(encoding="utf-8"))
        md = summ.get("mean_dwell_period")
        if md is None or not np.isfinite(float(md)):
            continue
        ida = int(np.argmin(np.abs(grid.Da - float(spec["Da"]))))
        p_ana = period_grid[ik, ida]
        if np.isfinite(p_ana):
            num_x.append(float(md))
            num_y.append(float(p_ana))
            ax.scatter(float(md), float(p_ana), s=80, c="C3", edgecolors="k", zorder=5)

    lims = [0.0, max(ax.get_xlim()[1], ax.get_ylim()[1], 1.0)]
    ax.plot(lims, lims, "k--", lw=1.0, alpha=0.6, label=r"$y=x$")
    ax.set_xlabel("Numerical mean_dwell_period")
    ax.set_ylabel(r"Analytical $T_1+T_2$")
    ax.set_title(f"Numerical vs analytical period for c1={c1_ref:g}, c2={c2_ref:g}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(panels / "panel_E_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _panel_F1(root: Path) -> None:
    panels = root / "panels"
    spec = next(r for r in RUN_SPECS if r["name"] == "Da1_c1_0p49_c2_0p35")
    rdir = _run_dir(root, spec)
    tt, c0, _ = _load_history(rdir)
    mask = tt <= 2.0 + 1e-12
    gf = np.load(root / "analytical" / "green_function_F1.npz")
    tg = np.asarray(gf["t"], dtype=np.float64).ravel()
    cg = np.asarray(gf["c_at_0"], dtype=np.float64).ravel()
    mg = tg <= 2.0 + 1e-12

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(tt[mask], c0[mask], "k-", lw=1.2, label="slab x∈[0,1]")
    ax.plot(tg[mg], cg[mg], "r--", lw=1.2, label="semi-infinite Green")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$c(0,t)$")
    ax.set_title(r"BC in $x=1$, not $\infty$ — $Da=1$, $\kappa=3$, $c_1=0.49$, $c_2=0.35$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(panels / "panel_F1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _panel_F2(root: Path) -> None:
    panels = root / "panels"
    spec = next(r for r in RUN_SPECS if r["name"] == "Da1_c1_0p49_c2_0p35")
    rdir = _run_dir(root, spec)
    sn_t, sn_c = _load_snapshots(rdir)
    n = sn_c.shape[1]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(F2_X_START)))
    for xs, col in zip(F2_X_START, colors, strict=True):
        ix = int(round(xs * (n - 1)))
        ax.plot(sn_t, sn_c[:, ix], color=col, lw=1.0, label=f"x={xs:g}")
        ax.axhline(xs, color=col, ls="--", lw=0.8, alpha=0.5)
    ax.set_xlim(0.0, min(5.0, float(sn_t[-1])))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$c$")
    ax.set_title(r"$c(x_{\mathrm{start}},t)$ for various $x_{\mathrm{start}}$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(panels / "panel_F2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _assemble_figure(root: Path) -> None:
    panels = root / "panels"
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.4, 0.55],
        hspace=0.28,
        wspace=0.22,
    )
    fig.suptitle("UW poster kinetic full replication (slab_kinetic)", fontsize=14, y=0.995)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_c0_panel(ax_a, root, [(c1, c2, col, 1.0) for c1, c2, col in PANEL_A], Da=1.0)
    ax_a.set_title(r"Panel A: $Da=1$, $\kappa=3$")

    ax_c = fig.add_subplot(gs[0, 1])
    _plot_heatmap_panel(
        ax_c,
        root,
        "Da1_c1_0p49_c2_0p26",
        r"$c(x,t)$ for $Da=1$, $c_1=0.49$, $c_2=0.26$",
    )

    ax_d = fig.add_subplot(gs[0, 2])
    _plot_heatmap_panel(
        ax_d,
        root,
        "Da2_c1_0p33_c2_0p20",
        r"$c(x,t)$ for $Da=2$, $c_1=0.33$, $c_2=0.20$",
    )

    ax_b = fig.add_subplot(gs[1, 0])
    _plot_c0_panel(
        ax_b,
        root,
        [(c1, c2, col, 2.0) for c1, c2, col in PANEL_B],
        Da=2.0,
    )
    ax_b.set_title(r"Panel B: $Da=2$, $\kappa=3$")

    ax_f1 = fig.add_subplot(gs[1, 1:])
    img = plt.imread(panels / "panel_F1.png")
    ax_f1.imshow(img)
    ax_f1.axis("off")

    ax_f2 = fig.add_subplot(gs[2, 0])
    img2 = plt.imread(panels / "panel_F2.png")
    ax_f2.imshow(img2)
    ax_f2.axis("off")

    ax_e1 = fig.add_subplot(gs[2, 1:])
    img3 = plt.imread(panels / "panel_E_T1.png")
    ax_e1.imshow(img3)
    ax_e1.axis("off")

    ax_e2 = fig.add_subplot(gs[3, 0:2])
    img4 = plt.imread(panels / "panel_E_T2.png")
    ax_e2.imshow(img4)
    ax_e2.axis("off")

    ax_sc = fig.add_subplot(gs[3, 2])
    img5 = plt.imread(panels / "panel_E_scatter.png")
    ax_sc.imshow(img5)
    ax_sc.axis("off")

    out = root / "figure_poster_kinetic_replication.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_panels_and_figure(root: Path) -> None:
    grid = _build_analytical(root)
    if not (root / "analytical" / "green_function_F1.npz").is_file():
        _build_green_F1(root)
    panels = root / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    print("=== panels A–D ===", flush=True)
    fig_a, ax = plt.subplots(figsize=(5, 3.5))
    _plot_c0_panel(ax, root, [(c1, c2, col, 1.0) for c1, c2, col in PANEL_A], Da=1.0)
    ax.set_title(r"$c(x=0,t)$, $Da=1$, $\kappa=3$")
    fig_a.tight_layout()
    fig_a.savefig(panels / "panel_A.png", dpi=130, bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax = plt.subplots(figsize=(5, 3.5))
    _plot_c0_panel(ax, root, [(c1, c2, col, 2.0) for c1, c2, col in PANEL_B], Da=2.0)
    ax.set_title(r"$c(x=0,t)$, $Da=2$, $\kappa=3$")
    fig_b.tight_layout()
    fig_b.savefig(panels / "panel_B.png", dpi=130, bbox_inches="tight")
    plt.close(fig_b)

    fig_c, ax = plt.subplots(figsize=(4.5, 3.8))
    _plot_heatmap_panel(
        ax,
        root,
        "Da1_c1_0p49_c2_0p26",
        r"$c(x,t)$ for $Da=1$, $c_1=0.49$, $c_2=0.26$",
    )
    fig_c.tight_layout()
    fig_c.savefig(panels / "panel_C.png", dpi=130, bbox_inches="tight")
    plt.close(fig_c)

    fig_d, ax = plt.subplots(figsize=(4.5, 3.8))
    _plot_heatmap_panel(
        ax,
        root,
        "Da2_c1_0p33_c2_0p20",
        r"$c(x,t)$ for $Da=2$, $c_1=0.33$, $c_2=0.20$",
    )
    fig_d.tight_layout()
    fig_d.savefig(panels / "panel_D.png", dpi=130, bbox_inches="tight")
    plt.close(fig_d)

    _panel_E_heatmaps(root, grid)
    _panel_E_scatter(root, grid)
    _panel_F1(root)
    _panel_F2(root)
    _assemble_figure(root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Existing replication output dir (for --panels-only)",
    )
    ap.add_argument(
        "--panels-only",
        action="store_true",
        help="Skip simulations; build panels from existing manifest",
    )
    args = ap.parse_args()

    if args.panels_only:
        if args.root is None:
            print("--root required with --panels-only", file=sys.stderr)
            return 2
        root = args.root.resolve()
        if not (root / "manifest.json").is_file():
            print(f"Missing manifest under {root}", file=sys.stderr)
            return 1
        _build_panels_and_figure(root)
        print(f"Panels written under {root}")
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = (
        args.root.resolve()
        if args.root
        else (_REPO / "results/slab_kinetic/poster_kinetic_replication" / stamp)
    )
    root.mkdir(parents=True, exist_ok=True)

    print("=== 12 fresh slab_kinetic runs ===", flush=True)
    rows, total_wall = _run_all_numerical(root)

    manifest = {
        "timestamp": root.name,
        "formula_notes": poster_formula_notes(),
        "grid_meta": grid_metadata(),
        "runs": rows,
        "total_wall_time_s": total_wall,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("=== analytical + panels ===", flush=True)
    _build_panels_and_figure(root)

    print(f"Done. Output: {root}", flush=True)
    print(f"Total numerical wall: {total_wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
