"""Tables and heatmaps for ``slab_kinetic_oscillation_window`` sweep results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Must match experiments/sweeps/slab_kinetic_oscillation_window.yaml
C1_VALUES = [0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.47, 0.49, 0.50]
C2_VALUES = [0.25, 0.27, 0.29, 0.31, 0.34, 0.37, 0.40, 0.43, 0.46, 0.49]

FIXED_DA = 1.0
FIXED_KAPPA = 3.0


def analytical_bounds(Da: float, kappa: float) -> tuple[float, float]:
    da = float(Da)
    ka = float(kappa)
    lo = 1.0 / (1.0 + ka * da)
    hi = 1.0 / (1.0 + da)
    return lo, hi


def write_oscillation_window_artifacts(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    sweep_dir = sweep_dir.resolve()
    write_table_all_runs(sweep_dir, manifest)
    plot_window_figures(sweep_dir, manifest)


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TAK" if v else "NIE"
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def write_table_all_runs(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for e in manifest.get("entries", []):
        c1 = float(e.get("c1", float("nan")))
        c2 = float(e.get("c2", float("nan")))
        n_tr = e.get("n_transitions")
        rows.append(
            {
                "run_label": e.get("run_label", ""),
                "c1": c1,
                "c2": c2,
                "status": e.get("status", ""),
                "n_transitions": n_tr if n_tr is not None else "",
                "dwell_L_mean": e.get("dwell_L_mean"),
                "dwell_H_mean": e.get("dwell_H_mean"),
                "peak_period": e.get("peak_period"),
                "thickness_final": e.get("thickness_final"),
                "wall_time_s": e.get("wall_time_s"),
                "osc_predicted": e.get("osc_predicted"),
                "osc_observed": e.get("osc_observed"),
            }
        )
    rows.sort(key=lambda r: (float(r["c1"]), float(r["c2"])))

    header = (
        "| run_label | c1 | c2 | status | n_transitions | dwell_L_mean | dwell_H_mean | "
        "peak_period | thickness_final | wall_time_s | osc_predicted | osc_observed |\n"
        "|-----------|----|----|--------|---------------|--------------|--------------|"
        "-------------|-----------------|-------------|---------------|--------------|\n"
    )
    lines = [header.rstrip("\n")]
    for r in rows:
        lines.append(
            f"| {r['run_label']} | {r['c1']} | {r['c2']} | {r['status']} | "
            f"{_fmt_cell(r['n_transitions'])} | {_fmt_cell(r['dwell_L_mean'])} | "
            f"{_fmt_cell(r['dwell_H_mean'])} | {_fmt_cell(r['peak_period'])} | "
            f"{_fmt_cell(r['thickness_final'])} | {_fmt_cell(r['wall_time_s'])} | "
            f"{_fmt_cell(r['osc_predicted'])} | {_fmt_cell(r['osc_observed'])} |"
        )
    out = sweep_dir / "table_all_runs.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _idx(vals: list[float], x: float) -> int:
    for i, v in enumerate(vals):
        if abs(float(v) - float(x)) < 1e-8:
            return i
    raise ValueError(f"value {x} not in axis list")


def plot_window_figures(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    lo, hi = analytical_bounds(FIXED_DA, FIXED_KAPPA)

    n_mat = np.full((10, 10), np.nan, dtype=np.float64)
    pp_mat = np.full((10, 10), np.nan, dtype=np.float64)

    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        c1 = float(e["c1"])
        c2 = float(e["c2"])
        i = _idx(C1_VALUES, c1)
        j = _idx(C2_VALUES, c2)
        n_tr = int(e.get("n_transitions", 0))
        n_mat[i, j] = float(n_tr)
        if n_tr >= 2:
            pp = e.get("mean_dwell_period")
            if pp is None:
                pp = e.get("peak_period")
            if pp is not None:
                pp_mat[i, j] = float(pp)

    # --- n_transitions ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_nt = plt.cm.viridis.copy()
    cmap_nt.set_bad("white")
    masked_nt = np.ma.masked_invalid(n_mat)
    vmax_nt = float(np.nanmax(n_mat)) if np.isfinite(np.nanmax(n_mat)) else 1.0
    vmax_nt = max(vmax_nt, 1.0)
    im = ax.imshow(
        masked_nt,
        origin="lower",
        cmap=cmap_nt,
        aspect="auto",
        vmin=0.0,
        vmax=vmax_nt,
        extent=(
            C2_VALUES[0],
            C2_VALUES[-1],
            C1_VALUES[0],
            C1_VALUES[-1],
        ),
        interpolation="nearest",
    )
    ax.set_xticks(C2_VALUES)
    ax.set_yticks(C1_VALUES)
    ax.set_xlabel(r"$c_2$")
    ax.set_ylabel(r"$c_1$")
    ax.set_title("slab_kinetic — n_transitions w oknie oscylacji (Da=1, κ=3, T=20)")
    ax.axvline(lo, color="red", linestyle="--", label=rf"$c_2=1/(1+\kappa Da)={lo:.2f}$")
    ax.axhline(hi, color="red", linestyle="--", label=rf"$c_1=1/(1+Da)={hi:.2f}$")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$n_{\mathrm{transitions}}$")
    fig.savefig(sweep_dir / "figure_window_n_transitions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- mean dwell period (fallback: FFT peak_period if missing) ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cmap_pp = plt.cm.plasma.copy()
    cmap_pp.set_bad("lightgray")
    plot_pp = np.array(pp_mat, dtype=np.float64)
    masked_pp = np.ma.masked_invalid(plot_pp)
    finite_pp = plot_pp[np.isfinite(plot_pp)]
    if finite_pp.size > 0:
        vmin_pp = float(np.nanmin(finite_pp))
        vmax_pp = float(np.nanmax(finite_pp))
        if not np.isfinite(vmin_pp):
            vmin_pp, vmax_pp = 0.0, 1.0
        if vmax_pp <= vmin_pp:
            vmax_pp = vmin_pp + 1e-12
    else:
        vmin_pp, vmax_pp = 0.0, 1.0

    im2 = ax2.imshow(
        masked_pp,
        origin="lower",
        cmap=cmap_pp,
        aspect="auto",
        vmin=vmin_pp,
        vmax=vmax_pp,
        extent=(
            C2_VALUES[0],
            C2_VALUES[-1],
            C1_VALUES[0],
            C1_VALUES[-1],
        ),
        interpolation="nearest",
    )
    ax2.set_xticks(C2_VALUES)
    ax2.set_yticks(C1_VALUES)
    ax2.set_xlabel(r"$c_2$")
    ax2.set_ylabel(r"$c_1$")
    ax2.set_title(
        "slab_kinetic — mean dwell period (= dwell_L + dwell_H) w oknie oscylacji (Da=1, κ=3, T=20)"
    )
    ax2.axvline(lo, color="red", linestyle="--", label=rf"$c_2=1/(1+\kappa Da)={lo:.2f}$")
    ax2.axhline(hi, color="red", linestyle="--", label=rf"$c_1=1/(1+Da)={hi:.2f}$")
    ax2.legend(loc="upper right", fontsize=8)
    fig2.colorbar(im2, ax=ax2, label="mean_dwell_period")
    fig2.savefig(sweep_dir / "figure_window_peak_period.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)


def load_manifest(sweep_dir: Path) -> dict[str, Any]:
    return json.loads((sweep_dir / "manifest_axes.json").read_text(encoding="utf-8"))


# Thin hysteresis cells re-simulated with smaller dt (must match fine sweep YAML / script)
REFINED_PAIRS: list[tuple[float, float]] = [
    (0.30, 0.27),
    (0.30, 0.29),
    (0.33, 0.29),
    (0.33, 0.31),
    (0.36, 0.34),
    (0.39, 0.37),
    (0.42, 0.40),
    (0.45, 0.43),
    (0.47, 0.46),
    (0.49, 0.46),
]


def axis_edges(vals: list[float]) -> np.ndarray:
    """Bin edges for heatmap cells centered on ``vals`` (midpoint boundaries)."""
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return np.asarray([])
    if v.size == 1:
        d = 0.01
        return np.asarray([float(v[0]) - d, float(v[0]) + d])
    e = np.empty(v.size + 1, dtype=np.float64)
    e[0] = v[0] - (v[1] - v[0]) / 2.0
    for k in range(1, v.size):
        e[k] = (v[k - 1] + v[k]) / 2.0
    e[-1] = v[-1] + (v[-1] - v[-2]) / 2.0
    return e


def _fill_matrices_from_manifest(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n_mat = np.full((10, 10), np.nan, dtype=np.float64)
    m_mat = np.full((10, 10), np.nan, dtype=np.float64)
    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        c1 = float(e["c1"])
        c2 = float(e["c2"])
        i = _idx(C1_VALUES, c1)
        j = _idx(C2_VALUES, c2)
        n_tr = int(e.get("n_transitions", 0))
        n_mat[i, j] = float(n_tr)
        if n_tr >= 2:
            md = e.get("mean_dwell_period")
            if md is None:
                md = e.get("peak_period")
            if md is not None:
                m_mat[i, j] = float(md)
    return n_mat, m_mat


def write_final_combined_artifacts(
    main_sweep_dir: Path,
    fine_sweep_dir: Path,
    out_dir: Path,
) -> None:
    """Combined heatmaps: main sweep values with refined cells overwritten from fine sweep."""
    from matplotlib.patches import Rectangle

    main_sweep_dir = main_sweep_dir.resolve()
    fine_sweep_dir = fine_sweep_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_man = load_manifest(main_sweep_dir)
    fine_man = load_manifest(fine_sweep_dir)

    n_mat, m_mat = _fill_matrices_from_manifest(main_man)
    n_fine, m_fine = _fill_matrices_from_manifest(fine_man)

    refined = set(REFINED_PAIRS)
    for c1, c2 in refined:
        i = _idx(C1_VALUES, c1)
        j = _idx(C2_VALUES, c2)
        if np.isfinite(n_fine[i, j]):
            n_mat[i, j] = n_fine[i, j]
        if np.isfinite(m_fine[i, j]):
            m_mat[i, j] = m_fine[i, j]

    lo, hi = analytical_bounds(FIXED_DA, FIXED_KAPPA)
    xe = axis_edges(C2_VALUES)
    ye = axis_edges(C1_VALUES)

    title_nt = (
        "slab_kinetic — n_transitions [combined: main + 10 refined cells, dt_safety=0.1] "
        "(Da=1, κ=3, T=20)"
    )
    title_md = (
        "slab_kinetic — mean dwell period [combined: dt_safety=0.4 + 10 refined cells with "
        "dt_safety=0.1] (Da=1, κ=3, T=20)"
    )

    def _plot_matrix(
        data: np.ndarray,
        cmap_name: str,
        bad: str,
        vmin: float | None,
        vmax: float | None,
        cbar_label: str,
        fname: str,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(bad)
        plot_arr = np.ma.masked_invalid(data)
        im = ax.imshow(
            plot_arr,
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=(
                C2_VALUES[0],
                C2_VALUES[-1],
                C1_VALUES[0],
                C1_VALUES[-1],
            ),
            interpolation="nearest",
        )
        ax.set_xticks(C2_VALUES)
        ax.set_yticks(C1_VALUES)
        ax.set_xlabel(r"$c_2$")
        ax.set_ylabel(r"$c_1$")
        ax.set_title(title)
        ax.axvline(lo, color="red", linestyle="--", label=rf"$c_2=1/(1+\kappa Da)={lo:.2f}$")
        ax.axhline(hi, color="red", linestyle="--", label=rf"$c_1=1/(1+Da)={hi:.2f}$")
        for c1, c2 in REFINED_PAIRS:
            i = _idx(C1_VALUES, c1)
            j = _idx(C2_VALUES, c2)
            x0, x1 = float(xe[j]), float(xe[j + 1])
            y0, y1 = float(ye[i]), float(ye[i + 1])
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                )
            )
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)

    vmax_nt = float(np.nanmax(n_mat)) if np.isfinite(np.nanmax(n_mat)) else 1.0
    vmax_nt = max(vmax_nt, 1.0)
    _plot_matrix(
        n_mat,
        "viridis",
        "white",
        0.0,
        vmax_nt,
        r"$n_{\mathrm{transitions}}$",
        "figure_window_n_transitions_final.png",
        title_nt,
    )

    finite_m = m_mat[np.isfinite(m_mat)]
    if finite_m.size > 0:
        vmin_m = float(np.nanmin(finite_m))
        vmax_m = float(np.nanmax(finite_m))
        if not np.isfinite(vmin_m):
            vmin_m, vmax_m = 0.0, 1.0
        if vmax_m <= vmin_m:
            vmax_m = vmin_m + 1e-12
    else:
        vmin_m, vmax_m = 0.0, 1.0

    _plot_matrix(
        m_mat,
        "plasma",
        "lightgray",
        vmin_m,
        vmax_m,
        "mean_dwell_period",
        "figure_window_peak_period_final.png",
        title_md,
    )

    # Combined markdown pointer (manifest paths only; full merge left to notebooks)
    lines = [
        "# Combined oscillation-window tables",
        "",
        f"- Main sweep: `{main_sweep_dir}` (`table_all_runs.md`)",
        f"- Fine sweep: `{fine_sweep_dir}` (`table_all_runs.md`)",
        "",
        "Refined pairs (black outline on final figures) overwrite main-grid cells:",
        "",
        "| c1 | c2 |",
        "|----|-----|",
    ]
    for c1, c2 in REFINED_PAIRS:
        lines.append(f"| {c1} | {c2} |")
    (out_dir / "table_all_runs_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
