"""Static figure helpers (NumPy + Matplotlib).

Field PNGs and publication-style panels. Must not import ``io``, ``models``,
or ``experiments`` (``docs/ARCHITECTURE.md`` §3.7).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle


def _format_config_text(params: dict[str, Any]) -> str:
    """Plain-text CONFIG column for the params panel (monospace)."""
    exp = params.get("experiment", {}) or {}
    geo = params.get("geometry", {}) or {}
    phys = params.get("physics", {}) or {}
    stress = params.get("stress", {}) or {}
    time = params.get("time", {}) or {}
    kx = phys.get("kappa_x", phys.get("kappa", "?"))
    ky = phys.get("kappa_y", phys.get("kappa", "?"))

    def _fmt(x: Any) -> str:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, int | float):
            return f"{float(x):g}"
        return str(x)

    lines = [
        "CONFIG",
        "",
        f"  model:         {exp.get('model', '?')}",
        f"  stress.mode:   {stress.get('mode', 'none')}",
        f"  sigma_0:       {_fmt(stress.get('sigma_0', 0.0))}",
        f"  B:             {_fmt(stress.get('stress_coupling_B', 0.0))}",
        f"  T:             {_fmt(time.get('T', '?'))}",
        f"  dt:            {_fmt(time.get('dt', '?'))}",
        f"  n:             {geo.get('n', '?')}",
        f"  L:             {_fmt(geo.get('L', '?'))}",
        f"  R:             {_fmt(geo.get('R', '?'))}",
        f"  gamma:         {_fmt(phys.get('gamma', '?'))}",
        f"  kappa_x,y:     {kx!s}, {ky!s}",
        f"  ratchet:       {'on' if phys.get('use_ratchet', False) else 'off'}",
    ]
    return "\n".join(lines)


def _format_diagnostics_text(diagnostics: dict[str, Any] | None) -> str:
    """Plain-text DIAGNOSTICS column for the params panel."""
    lines = ["DIAGNOSTICS", ""]
    if diagnostics is None:
        lines.append("  (not provided)")
        return "\n".join(lines)

    if "option_b_leak_pct" in diagnostics:
        v = float(diagnostics["option_b_leak_pct"])
        lines.append(f"  option_b_leak_pct:   {v:.3f}%")

    jab = diagnostics.get("jab_canonical")
    if isinstance(jab, dict):
        if "n_bands" in jab:
            lines.append(f"  n_bands:             {jab['n_bands']}")
        if "q_cv" in jab:
            lines.append(f"  q_cv:                {float(jab['q_cv']):.3f}")

    aniso = diagnostics.get("psi_fft_anisotropy")
    if isinstance(aniso, dict) and "psi_fft_anisotropy_ratio" in aniso:
        lines.append(f"  psi_fft_aniso:       {float(aniso['psi_fft_anisotropy_ratio']):.3f}")

    bm = diagnostics.get("bands_multislice")
    if isinstance(bm, dict) and "median_peak_count_multislice" in bm:
        lines.append(f"  bands_median:        {float(bm['median_peak_count_multislice']):.3f}")

    if "chi_weighted_silica_final" in diagnostics:
        lines.append(
            f"  chi_w_silica:        {float(diagnostics['chi_weighted_silica_final']):.4g}"
        )

    if "max_phi_sum" in diagnostics:
        lines.append(f"  max_phi_sum:         {float(diagnostics['max_phi_sum']):.4g}")

    cm = diagnostics.get("coarsening_metrics")
    if isinstance(cm, dict):
        stats = cm.get("stats", {})
        if isinstance(stats, dict) and "var_psi" in stats:
            lines.append(f"  var(psi):            {float(stats['var_psi']):.4f}")

    if "wall_time_s" in diagnostics:
        lines.append(f"  wall_time:           {float(diagnostics['wall_time_s']):.1f}s")

    if len(lines) == 2:
        lines.append("  (no scalar metrics)")
    return "\n".join(lines)


def _plot_field_panels(
    fig: plt.Figure,
    axes_grid: tuple[Any, Any, Any, Any],
    *,
    pm: np.ndarray,
    pc: np.ndarray,
    cc: np.ndarray,
    chi: np.ndarray | None,
    L: float,
    R: float,
    extent: tuple[float, float, float, float],
) -> None:
    """Draw the four imshow panels on given axes (2×2 order)."""
    ax_pm, ax_pc, ax_c, ax_chi = axes_grid
    xc = 0.5 * L
    yc = 0.5 * L
    panels: list[tuple[Any, np.ndarray, str]] = [
        (ax_pm, pm, r"$\phi_m$"),
        (ax_pc, pc, r"$\phi_c$"),
        (ax_c, cc, r"$c$"),
    ]
    if chi is not None:
        panels.append((ax_chi, chi, r"$\chi$"))
    else:
        ax_chi.axis("off")

    for ax, arr, panel_title in panels:
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(panel_title)
        fig.colorbar(im, ax=ax, shrink=0.7)
        if R > 1e-9:
            ax.add_patch(
                Circle(
                    (xc, yc),
                    float(R),
                    fill=False,
                    edgecolor="white",
                    linewidth=0.9,
                    linestyle="--",
                )
            )


def plot_fields_final(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    c: np.ndarray,
    *,
    L: float,
    R: float,
    path: Path | str,
    chi: np.ndarray | None = None,
    dpi: int = 120,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    include_params_panel: bool = True,
) -> Path:
    """Save a 2×2 field panel (and optional suptitle + monospace params row) to PNG.

    If ``path`` ends with ``.png``, it is the output file. Otherwise ``path`` is
    treated as a directory and the file is ``path / "figures_final.png"``.

    When ``include_params_panel`` is true and ``params`` is not ``None``, the figure
    uses a third row for CONFIG / DIAGNOSTICS text (``params["_diagnostics"]``).
    """
    out = Path(path)
    if out.suffix.lower() != ".png":
        out = out / "figures_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    pm = np.asarray(phi_m, dtype=np.float64)
    pc = np.asarray(phi_c, dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64)
    ch = np.asarray(chi, dtype=np.float64) if chi is not None else None
    extent = (0.0, float(L), 0.0, float(L))

    use_panel = bool(include_params_panel and params is not None)

    if use_panel:
        fig = plt.figure(figsize=(9.0, 11.0), constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[4.0, 4.0, 1.6])
        ax_pm = fig.add_subplot(gs[0, 0])
        ax_pc = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_chi = fig.add_subplot(gs[1, 1])
        ax_params = fig.add_subplot(gs[2, :])
        ax_params.axis("off")
        _plot_field_panels(
            fig, (ax_pm, ax_pc, ax_c, ax_chi), pm=pm, pc=pc, cc=cc, chi=ch, L=L, R=R, extent=extent
        )
        if title:
            fig.suptitle(title, fontsize=12)
        diag_raw = params.get("_diagnostics")
        diag = diag_raw if isinstance(diag_raw, dict) else None
        config_text = _format_config_text(params)
        diag_text = _format_diagnostics_text(diag)
        ax_params.text(
            0.02,
            0.98,
            config_text,
            transform=ax_params.transAxes,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
        )
        ax_params.text(
            0.52,
            0.98,
            diag_text,
            transform=ax_params.transAxes,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.5), constrained_layout=True)
        _plot_field_panels(
            fig,
            (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]),
            pm=pm,
            pc=pc,
            cc=cc,
            chi=ch,
            L=L,
            R=R,
            extent=extent,
        )
        if title:
            fig.suptitle(title, fontsize=12)

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def parse_run_stamp_utc(stamp: str) -> str | None:
    """Parse ``YYYYMMDDTHHMMSSZ`` run directory name to ``YYYY-MM-DD HH:MM UTC``."""
    try:
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return None
