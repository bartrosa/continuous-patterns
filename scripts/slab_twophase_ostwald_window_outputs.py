"""Tables and summary figure for the S1 Ostwald-window sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.experiments.s1_ostwald_window import check_s1_anchor

_RUN_COLORS = {
    "S1_cost0p22_g4": "#1f77b4",
    "S1_cost0p25_g4": "#ff7f0e",
    "S1_cost0p30_g4": "#2ca02c",
    "S1_cost0p40_g4": "#d62728",
    "S1_cost0p25_g2": "#9467bd",
    "S1_cost0p25_g8": "#8c564b",
}

_PANEL_D_NOTE = (
    "Panel (d): one universal $c_{\\mathrm{front}}(t)$ trajectory "
    "(identical for all $\\gamma=4$ runs — composition has no feedback)."
)


def write_s1_artifacts(sweep_dir: Path, manifest: dict[str, Any]) -> str:
    sweep_dir = sweep_dir.resolve()
    write_table_s1(sweep_dir, manifest)
    plot_figure_s1_window(sweep_dir, manifest)
    anchor_ok, anchor_msg = check_s1_anchor(_anchor_entry(manifest))
    print(f"[S1 anchor] {anchor_msg}", flush=True)
    return anchor_msg if anchor_ok or "FAIL" in anchor_msg else anchor_msg


def _anchor_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    for entry in manifest.get("entries", []):
        if str(entry.get("run_label")) == "S1_cost0p25_g4":
            return dict(entry)
    return {}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.4g}"
    return str(v)


def _spacing_robust(r: dict[str, Any]) -> str:
    med = r.get("median_spacing_zone")
    excl = r.get("mean_spacing_zone_excl_last")
    if (
        med is not None
        and excl is not None
        and np.isfinite(float(med))
        and np.isfinite(float(excl))
    ):
        return f"{float(med):.4g} ({float(excl):.4g})"
    if med is not None and np.isfinite(float(med)):
        return f"{float(med):.4g} (—)"
    return "—"


def write_table_s1(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        rows.append(
            {
                "run_label": entry.get("run_label", ""),
                "c_ostwald": entry.get("c_ostwald"),
                "gamma": entry.get("gamma"),
                "n_bands_zone": entry.get("n_bands_zone"),
                "band_zone_extent": entry.get("band_zone_extent"),
                "mean_spacing_zone": entry.get("mean_spacing_zone"),
                "median_spacing_zone": entry.get("median_spacing_zone"),
                "mean_spacing_zone_excl_last": entry.get("mean_spacing_zone_excl_last"),
                "std_spacing_zone": entry.get("std_spacing_zone"),
                "row_validity": entry.get("row_validity", entry.get("status")),
            }
        )
    rows.sort(key=lambda r: (float(r.get("c_ostwald", 0)), float(r.get("gamma", 0))))

    header = (
        "| run_label | c_ostwald | gamma | n_bands_zone | band_zone_extent | "
        "mean_spacing ± std | median (excl-last mean) | validity |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = [header.rstrip("\n")]
    for r in rows:
        mean = r.get("mean_spacing_zone")
        std = r.get("std_spacing_zone")
        if mean is not None and std is not None and np.isfinite(float(mean)):
            sp = f"{float(mean):.4g} ± {float(std):.4g}"
        else:
            sp = "—"
        lines.append(
            f"| {r['run_label']} | {_fmt(r.get('c_ostwald'))} | {_fmt(r.get('gamma'))} | "
            f"{_fmt(r.get('n_bands_zone'))} | {_fmt(r.get('band_zone_extent'))} | {sp} | "
            f"{_spacing_robust(r)} | {r.get('row_validity', '—')} |"
        )
    (sweep_dir / "table_s1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure_s1_window(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    entries = [e for e in manifest.get("entries", []) if e.get("status") == "success"]
    if not entries:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    # (a) window map — gamma=4
    ax_a = axes[0, 0]
    g4 = sorted(
        [e for e in entries if float(e.get("gamma", -1)) == 4.0],
        key=lambda e: float(e.get("c_ostwald", 0)),
    )
    if g4:
        cost = np.array([float(e["c_ostwald"]) for e in g4], dtype=np.float64)
        extent = np.array([float(e.get("band_zone_extent", np.nan)) for e in g4])
        nb = np.array([float(e.get("n_bands_zone", np.nan)) for e in g4])
        ax_a.plot(cost, extent, "o-", color="#1f77b4", lw=1.2, label="band_zone_extent")
        ax_b = ax_a.twinx()
        ax_b.plot(cost, nb, "s--", color="#ff7f0e", lw=1.0, label="n_bands_zone")
        ax_a.set_xlabel(r"$c_{\mathrm{ostwald}}$")
        ax_a.set_ylabel("band zone extent", color="#1f77b4")
        ax_b.set_ylabel(r"$n_{\mathrm{bands,zone}}$", color="#ff7f0e")
        ax_a.set_title("(a) Ostwald window map ($\\gamma=4$)")
        lines1, lab1 = ax_a.get_legend_handles_labels()
        lines2, lab2 = ax_b.get_legend_handles_labels()
        ax_a.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=8)
    else:
        ax_a.set_title("(a) Ostwald window map (no data)")

    # (b) spacing vs gamma at c_ost=0.25
    ax_b2 = axes[0, 1]
    gscan = sorted(
        [e for e in entries if abs(float(e.get("c_ostwald", 0)) - 0.25) < 1e-9],
        key=lambda e: float(e.get("gamma", 0)),
    )
    if gscan:
        gam = np.array([float(e["gamma"]) for e in gscan], dtype=np.float64)
        mean_d = np.array([float(e.get("mean_spacing_zone", np.nan)) for e in gscan])
        std_d = np.array([float(e.get("std_spacing_zone", 0.0)) for e in gscan])
        mask = np.isfinite(mean_d) & (mean_d > 0) & (gam > 0)
        if np.any(mask):
            for i in np.where(mask)[0]:
                entry = gscan[i]
                invalid = str(entry.get("row_validity", "OK")) != "OK"
                ax_b2.errorbar(
                    gam[i],
                    mean_d[i],
                    yerr=std_d[i],
                    fmt="o",
                    capsize=3,
                    color="#2ca02c",
                    markerfacecolor="none" if invalid else "#2ca02c",
                    markeredgecolor="#2ca02c",
                    markeredgewidth=1.2,
                )
            ref_i = int(np.argmin(np.abs(gam[mask] - 4.0)))
            g0 = float(gam[mask][ref_i])
            d0 = float(mean_d[mask][ref_i])
            g_line = np.linspace(gam[mask].min(), gam[mask].max(), 64)
            d_line = d0 * np.sqrt(g0 / g_line)
            ax_b2.plot(g_line, d_line, "--", color="0.4", lw=1.0, label=r"slope $-1/2$ anchor")
            ax_b2.set_xscale("log")
            ax_b2.set_yscale("log")
        ax_b2.set_xlabel(r"$\gamma$")
        ax_b2.set_ylabel(r"mean spacing $d$")
        ax_b2.set_title(r"(b) Spacing vs $\gamma$ ($c_{\mathrm{ostwald}}=0.25$)")
        ax_b2.legend(fontsize=8)
    else:
        ax_b2.set_title("(b) spacing vs gamma (no data)")

    # (c) spacing vs front velocity
    ax_c = axes[1, 0]
    seen_labels: set[str] = set()
    for entry in entries:
        label = str(entry.get("run_label", ""))
        col = _RUN_COLORS.get(label, "#333333")
        pts = entry.get("spacing_pacing") or []
        d_vals = [float(p["d_n"]) for p in pts if np.isfinite(float(p.get("d_n", float("nan"))))]
        v_vals = [
            float(p["v_front"]) for p in pts if np.isfinite(float(p.get("v_front", float("nan"))))
        ]
        if not d_vals:
            continue
        show_label = label not in seen_labels
        ax_c.scatter(
            v_vals,
            d_vals,
            s=18,
            alpha=0.75,
            color=col,
            label=label if show_label else None,
        )
        seen_labels.add(label)
    ax_c.set_xlabel(r"front velocity $v_{\mathrm{front}}$ at band formation")
    ax_c.set_ylabel(r"spacing $d_n$")
    ax_c.set_title("(c) Spacing vs front velocity (all runs)")
    handles, labels = ax_c.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles, strict=False))
        ax_c.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")

    # (d) universal c_front(t) with Ostwald windows — gamma=4 runs
    ax_d = axes[1, 1]
    cft: np.ndarray | None = None
    for entry in g4:
        label = str(entry.get("run_label", ""))
        col = _RUN_COLORS.get(label, "#333333")
        rel = entry.get("relative_path")
        if not rel:
            continue
        summary_path = sweep_dir / str(rel) / "summary.json"
        if not summary_path.is_file():
            continue
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        cft_row = np.asarray(summ.get("c_front_t", []), dtype=np.float64)
        if cft_row.ndim != 2 or cft_row.shape[1] < 2 or cft_row.size == 0:
            continue
        if cft is None:
            cft = cft_row
        c_ost = float(entry.get("c_ostwald", 0.25))
        w = float(entry.get("w_ostwald", 0.05))
        ax_d.fill_between(
            cft_row[:, 0],
            c_ost - w,
            c_ost + w,
            color=col,
            alpha=0.12,
            linewidth=0,
            label=f"$c_\\mathrm{{ost}}={c_ost:.2g}$ window",
        )
    if cft is not None:
        ax_d.plot(
            cft[:, 0],
            cft[:, 1],
            lw=1.4,
            color="#333333",
            label="universal $c_{\\mathrm{front}}(t)$ (identical for all $\\gamma=4$ runs — composition has no feedback)",  # noqa: E501
        )
    ax_d.set_xlabel("time t")
    ax_d.set_ylabel(r"$c_{\mathrm{front}}(t)$")
    ax_d.set_title(r"(d) Front concentration vs Ostwald window ($\gamma=4$)")
    ax_d.legend(fontsize=6.5, loc="best")

    fig.text(0.5, 0.01, _PANEL_D_NOTE, ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    fig.savefig(sweep_dir / "figure_s1_window.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
