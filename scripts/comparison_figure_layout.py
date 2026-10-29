"""Dedicated inch-stacked banner layout for master comparison figures (EXP-D6 / D6.1)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.text import Text

_FONT_CAPTION = 8
_FONT_TICK = 8
_FONT_TITLE = 11

_TIER1_HEANEY = (
    "HEANEY MODEL — kinetic hysteresis clock | liquid concentration oscillates at the "
    "reaction surface; sets the PERIOD"
)
_TIER1_CH = (
    "CAHN-HILLIARD MODEL — phase-field demixing | composition separates in the solid "
    "behind the front; sets the WAVELENGTH"
)
_HEANEY_TOKEN = "Heaney"

# D4/D5 overlay reference: fig_h = 4.6 * n_rows + banner_pad; height_ratios [1, 0.12]/row.
FIG_WIDTH_IN = 22.0
PLOT_BLOCK_IN = 4.6
PANEL_ROW_IN = PLOT_BLOCK_IN * (1.0 / (1.0 + 0.12))
CAPTION_ROW_IN = PLOT_BLOCK_IN * (0.12 / (1.0 + 0.12))
H_SUBTITLE_IN = 0.35
H_TIER1_IN = 0.45
H_TIER2_IN = 0.30
H_LEGEND_IN = 0.14
H_FOOTER_IN = 0.10
ROW_GAP_IN = 0.08
FIG_DPI = 160
MIN_PANEL_ROW_PX_AT_DPI = PANEL_ROW_IN * FIG_DPI * 0.90

_INCHES = {
    "subtitle": H_SUBTITLE_IN,
    "tier1": H_TIER1_IN,
    "tier2": H_TIER2_IN,
    "panel": PANEL_ROW_IN,
    "caption": CAPTION_ROW_IN,
}

LAST_LAYOUT_REPORT: dict[str, Any] | None = None


def _layout_report(
    *,
    fig: plt.Figure,
    n_plot_rows: int,
    n_tier1: int,
    n_tier2: int,
    measured_panel_row_px_min: float,
) -> dict[str, Any]:
    return {
        "fig_width_in": FIG_WIDTH_IN,
        "fig_height_in": float(fig.get_figheight()),
        "panel_row_in": PANEL_ROW_IN,
        "h_tier1_in": H_TIER1_IN,
        "h_tier2_in": H_TIER2_IN,
        "n_plot_rows": n_plot_rows,
        "n_tier1": n_tier1,
        "n_tier2": n_tier2,
        "min_panel_row_px": MIN_PANEL_ROW_PX_AT_DPI,
        "measured_panel_row_px_min": measured_panel_row_px_min,
        "bbox_gate_pass": True,
    }


def _figure_row_plan(experiments: list[dict[str, Any]], *, subtitle: str) -> list[tuple[str, Any]]:
    plan: list[tuple[str, Any]] = []
    if subtitle:
        plan.append(("subtitle", subtitle))
    heaney_t1 = ch_t1 = False
    for i, entry in enumerate(experiments):
        mech = str(entry.get("mechanism_class", ""))
        is_heaney = _HEANEY_TOKEN in mech
        if is_heaney and not heaney_t1:
            plan.append(("tier1", _TIER1_HEANEY))
            heaney_t1 = True
        if not is_heaney and not ch_t1:
            plan.append(("tier1", _TIER1_CH))
            ch_t1 = True
        vb = entry.get("variant_banner")
        if vb:
            plan.append(("tier2", str(vb)))
        plan.append(("panel", i))
        plan.append(("caption", i))
    return plan


def _bbox_intersects(a: Any, b: Any, *, tol_px: float = 2.0) -> tuple[float, float] | None:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    w = x1 - x0
    h = y1 - y0
    if w > tol_px and h > tol_px:
        return w, h
    return None


def assert_d6_layout_gates(
    fig: plt.Figure,
    *,
    banner_axes: list[Any],
    banner_texts: list[Text],
    plot_axes: list[Any],
    panel_row_groups: list[list[Any]],
    tol_px: float = 2.0,
) -> dict[str, Any]:
    """Hard-fail if banners touch plots, plot axes overlap, or panel rows shrank."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for bi, ba in enumerate(banner_axes):
        bb_b = ba.get_window_extent(renderer=renderer)
        for pi, pa in enumerate(plot_axes):
            bb_p = pa.get_window_extent(renderer=renderer)
            hit = _bbox_intersects(bb_b, bb_p, tol_px=tol_px)
            if hit:
                raise RuntimeError(
                    f"banner/plot overlap: banner_ax[{bi}] vs plot[{pi}] "
                    f"intersection={hit[0]:.1f}x{hit[1]:.1f}px"
                )

    for ti, txt in enumerate(banner_texts):
        bb_t = txt.get_window_extent(renderer=renderer)
        for pi, pa in enumerate(plot_axes):
            bb_p = pa.get_window_extent(renderer=renderer)
            hit = _bbox_intersects(bb_t, bb_p, tol_px=tol_px)
            if hit:
                raise RuntimeError(
                    f"banner text/plot overlap: banner_text[{ti}] vs plot[{pi}] "
                    f"intersection={hit[0]:.1f}x{hit[1]:.1f}px"
                )

    for i in range(len(plot_axes)):
        for j in range(i + 1, len(plot_axes)):
            bb_i = plot_axes[i].get_window_extent(renderer=renderer)
            bb_j = plot_axes[j].get_window_extent(renderer=renderer)
            hit = _bbox_intersects(bb_i, bb_j, tol_px=tol_px)
            if hit:
                raise RuntimeError(
                    f"plot/plot overlap: plot[{i}] vs plot[{j}] "
                    f"intersection={hit[0]:.1f}x{hit[1]:.1f}px"
                )

    min_panel_px = PANEL_ROW_IN * float(fig.dpi) * 0.90
    row_heights_px: list[float] = []
    for gi, group in enumerate(panel_row_groups):
        bbs = [ax.get_window_extent(renderer=renderer) for ax in group]
        y0 = min(bb.y0 for bb in bbs)
        y1 = max(bb.y1 for bb in bbs)
        h_px = y1 - y0
        row_heights_px.append(h_px)
        if h_px < min_panel_px:
            raise RuntimeError(
                f"panel row {gi} height {h_px:.1f}px @ dpi={fig.dpi:.0f} "
                f"< reference {min_panel_px:.1f}px "
                f"(panel_row_in={PANEL_ROW_IN})"
            )

    return _layout_report(
        fig=fig,
        n_plot_rows=len(panel_row_groups),
        n_tier1=0,
        n_tier2=0,
        measured_panel_row_px_min=min(row_heights_px) if row_heights_px else 0.0,
    )


def assert_banner_plot_disjoint(
    fig: plt.Figure, banner_axes: list[Any], plot_axes: list[Any], *, tol_px: float = 2.0
) -> None:
    """Backward-compatible wrapper."""
    assert_d6_layout_gates(
        fig,
        banner_axes=banner_axes,
        banner_texts=[],
        plot_axes=plot_axes,
        panel_row_groups=[plot_axes[i : i + 4] for i in range(0, len(plot_axes), 4)],
        tol_px=tol_px,
    )


_PANEL_COLS = (0.025, 0.225, 0.015)  # x0, width, gap


def _place_panel_row(fig: plt.Figure, y_bottom: float, h_frac: float) -> list[Any]:
    x0, col_w, col_gap = _PANEL_COLS
    axes: list[Any] = []
    for col in range(4):
        xf = x0 + col * (col_w + col_gap)
        axes.append(fig.add_axes([xf, y_bottom, col_w, h_frac * 0.98]))
    return axes


def write_comparison_figure_dedicated(
    path: Path,
    *,
    records: list[Any],
    metrics_list: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    front_origins: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    invocation_id: str,
    footer_note: str,
    figure_subtitle: str,
    label_legend: list[tuple[str, str]],
    panel_titles: tuple[str, str, str, str],
    plot_wall_dynamics: Callable[..., None],
    plot_kymograph: Callable[..., None],
    plot_band_strip: Callable[..., None],
    plot_spacing_laws: Callable[..., list[Any]],
    load_run_snapshots: Callable[..., Any],
    load_run_history: Callable[..., Any],
    supply_regime: Callable[[dict[str, Any]], str],
    fmt_law: Callable[..., str],
    status_draws_overlay: Callable[[str], bool],
) -> dict[str, Any]:
    global LAST_LAYOUT_REPORT

    plan = _figure_row_plan(experiments, subtitle=figure_subtitle)
    heights_in = [_INCHES[kind] for kind, _ in plan]
    n_gs = len(heights_in)
    n_tier1 = sum(1 for k, _ in plan if k == "tier1")
    n_tier2 = sum(1 for k, _ in plan if k == "tier2")
    fig_h = sum(heights_in) + ROW_GAP_IN * max(0, n_gs - 1) + H_LEGEND_IN + H_FOOTER_IN + 0.15
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=FIG_DPI)

    banner_axes: list[Any] = []
    banner_texts: list[Text] = []
    plot_axes: list[Any] = []
    panel_row_groups: list[list[Any]] = []
    first_panel_titles = True
    y = 1.0 - 0.006

    for i, (kind, payload) in enumerate(plan):
        h_frac = heights_in[i] / fig_h
        if i > 0:
            y -= ROW_GAP_IN / fig_h
        y -= h_frac
        y_bottom = y

        if kind == "subtitle":
            ax = fig.add_axes([0.025, y_bottom, 0.95, h_frac * 0.98])
            ax.axis("off")
            banner_axes.append(ax)
            banner_texts.append(
                ax.text(
                    0.5,
                    0.5,
                    str(payload),
                    ha="center",
                    va="center",
                    fontsize=13,
                    transform=ax.transAxes,
                )
            )
        elif kind == "tier1":
            ax = fig.add_axes([0.025, y_bottom, 0.95, h_frac * 0.98])
            ax.axis("off")
            banner_axes.append(ax)
            banner_texts.append(
                ax.text(
                    0.5,
                    0.62,
                    str(payload),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
            )
            ax.axhline(0.18, color="0.65", lw=0.9, xmin=0.05, xmax=0.95)
        elif kind == "tier2":
            ax = fig.add_axes([0.025, y_bottom, 0.95, h_frac * 0.98])
            ax.axis("off")
            banner_axes.append(ax)
            banner_texts.append(
                ax.text(
                    0.5,
                    0.62,
                    str(payload),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="semibold",
                    transform=ax.transAxes,
                )
            )
            ax.axhline(0.15, color="0.75", lw=0.6, xmin=0.08, xmax=0.92)
        elif kind == "panel":
            row_idx = int(payload)
            rec = records[row_idx]
            metrics = metrics_list[row_idx]
            cfg = configs[row_idx]
            fo = front_origins[row_idx]
            trow = table_rows[row_idx]
            entry = experiments[row_idx]
            row_status = str(trow.get("row_status", "OK"))
            axes = _place_panel_row(fig, y_bottom, h_frac)
            if first_panel_titles:
                for ax, title in zip(axes, panel_titles, strict=True):
                    ax.set_title(title, fontsize=_FONT_TITLE)
                first_panel_titles = False

            c1 = c2 = None
            if rec.model == "slab_kinetic":
                phys = cfg.get("physics", {})
                c1 = float(phys.get("c1", float("nan")))
                c2 = float(phys.get("c2", float("nan")))
            elif rec.model == "slab_twophase":
                phys = cfg.get("physics", {})
                if str(phys.get("precipitation", "")).lower() == "hysteresis":
                    c1 = float(phys.get("c1", float("nan")))
                    c2 = float(phys.get("c2", float("nan")))

            run_dir = metrics.get("run_dir")
            run_path = Path(run_dir) if run_dir else None
            snapshots = load_run_snapshots(run_path)
            history = load_run_history(run_path)

            plot_wall_dynamics(
                axes[0],
                rec,
                c1=c1,
                c2=c2,
                history=history,
                snapshots=snapshots,
                front_origin=fo,
                cfg=cfg,
            )
            plot_kymograph(axes[1], rec, snapshots)
            plot_band_strip(
                axes[2],
                rec,
                row_status=row_status,
                invalid_reasons=list(trow.get("invalid_reasons") or []),
            )
            law_axes = plot_spacing_laws(
                axes[3],
                metrics,
                row_status=row_status,
                run_label=str(entry.get("run_label", "")),
            )
            row_plot_axes = [*axes[:3], *law_axes]
            plot_axes.extend(row_plot_axes)
            panel_row_groups.append(row_plot_axes)
        elif kind == "caption":
            row_idx = int(payload)
            metrics = metrics_list[row_idx]
            cfg = configs[row_idx]
            fo = front_origins[row_idx]
            reg = regressions[row_idx]
            trow = table_rows[row_idx]
            entry = experiments[row_idx]
            rec = records[row_idx]
            row_status = str(trow.get("row_status", "OK"))
            ax = fig.add_axes([0.025, y_bottom, 0.95, h_frac * 0.98])
            ax.axis("off")
            supply = supply_regime(cfg)
            mass_pct = metrics.get("mass_residual_pct")
            mass_str = "n/a" if mass_pct is None else f"{float(mass_pct):.3g}%"
            if row_status not in ("OK",):
                laws_str = f"FLAG ({row_status})"
            elif str(metrics.get("laws_reason") or "").startswith("integrity_"):
                laws_str = f"p={fmt_law(metrics.get('p_fit'), metrics)} (integrity flag)"
            elif metrics.get("laws_reason") == "insufficient_bands":
                laws_str = "laws: n/a (n<6)"
            elif supply == "closed":
                laws_str = (
                    f"p={fmt_law(metrics.get('p_fit'), metrics)} / "
                    f"excl_last={fmt_law(metrics.get('p_fit_excl_last'), metrics)}"
                )
            else:
                laws_str = f"p={fmt_law(metrics.get('p_fit'), metrics)}"
            min_pw = metrics.get("min_phi_w")
            min_pw_str = "n/a" if min_pw is None else f"{float(min_pw):.3g}"
            clip_rate = metrics.get("clip_rate")
            clip_str = "n/a" if clip_rate is None else f"{float(clip_rate):.3e}"
            flux = metrics.get("max_abs_right_flux")
            flux_str = "n/a" if flux is None else f"{float(flux):.3g}"
            e3_note = metrics.get("e3_pair_summary")
            display = str(entry.get("display_label", rec.run_label))
            mech = str(entry.get("mechanism_class", ""))
            ingest_note = metrics.get("ingest_source")
            caption = (
                f"{display} | {rec.model} | {supply} | status={row_status} | "
                f"n_bands={metrics.get('n_bands')} | {laws_str} | "
                f"front_origin={fo.get('origin', 'NONE')} | mass_res={mass_str} | "
                f"min_phi_w={min_pw_str} | clip_rate={clip_str} | flux_R={flux_str} | "
                f"regression={reg.get('status', 'SKIP')}"
            )
            if mech:
                caption = f"{caption} | class={mech}"
            if ingest_note:
                caption = f"{caption} | ingest={ingest_note}"
            if e3_note:
                caption = f"{caption} | {e3_note}"
            ax.text(
                0.0,
                0.5,
                caption,
                fontsize=_FONT_CAPTION - 1,
                family="monospace",
                va="center",
                ha="left",
                transform=ax.transAxes,
                color="0.35",
            )

    legend_h = H_LEGEND_IN / fig_h
    legend_ax = fig.add_axes([0.08, 0.018, 0.84, legend_h * 0.85])
    legend_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="k", linewidth=0.3)
        for _lbl, col in label_legend
    ]
    legend_ax.legend(
        handles,
        [lbl for lbl, _ in label_legend],
        loc="center",
        ncol=3,
        fontsize=_FONT_TICK,
        frameon=False,
    )

    footer = f"invocation_id={invocation_id}  (fresh compute — all rows from this timestamp)"
    if footer_note:
        footer = f"{footer}  |  {footer_note}"
    fig.text(
        0.5, 0.004, footer, ha="center", va="bottom", fontsize=_FONT_CAPTION, family="monospace"
    )

    report = assert_d6_layout_gates(
        fig,
        banner_axes=banner_axes,
        banner_texts=banner_texts,
        plot_axes=plot_axes,
        panel_row_groups=panel_row_groups,
    )
    report["n_tier1"] = n_tier1
    report["n_tier2"] = n_tier2
    LAST_LAYOUT_REPORT = report

    fig.savefig(path, dpi=FIG_DPI, pad_inches=0.08)
    plt.close(fig)
    return report
