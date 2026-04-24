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
    """Plain-text DIAGNOSTICS column: mass balance block + morphology block."""
    lines = ["MASS BALANCE:"]
    if diagnostics is None:
        lines.append("  (not provided)")
        return "\n".join(lines)

    sd = diagnostics.get("spectral_mass_drift", {})
    if isinstance(sd, dict) and sd.get("leak_pct") is not None:
        lines.append(f"  spectral_drift:      {float(sd['leak_pct']):.2e}%")
    else:
        lines.append("  spectral_drift:      (not recorded)")

    dmb = diagnostics.get("dirichlet_mass_balance", {})
    if isinstance(dmb, dict) and dmb.get("residual_pct") is not None:
        lines.append(f"  dirichlet_residual:  {float(dmb['residual_pct']):.3f}%")
        rto = dmb.get("ratio")
        if rto is not None and rto == rto:
            lines.append(f"  dirichlet_ratio:     {float(rto):.3f}")
        else:
            lines.append("  dirichlet_ratio:     (n/a)")
    else:
        lines.append("  dirichlet_residual:  (not recorded)")

    sfb = diagnostics.get("surface_flux_balance", {})
    if isinstance(sfb, dict) and sfb.get("leak_pct") is not None:
        lp = float(sfb["leak_pct"])
        ns = int(sfb.get("n_samples", 0))
        ft = sfb.get("front_arrival_t", float("nan"))
        lines.append(f"  surface_flux_leak:   {lp:+.3f}%")
        lines.append(f"  surface_n_samples:   {ns:d}")
        try:
            ft_f = float(ft)
            ft_ok = ft_f == ft_f
        except (TypeError, ValueError):
            ft_ok = False
        if ft_ok:
            lines.append(f"  surface_front_t:     {ft_f:.1f}")
        else:
            lines.append("  surface_front_t:     (not reached)")
    else:
        lines.append("  surface_flux_leak:   (not recorded)")

    lines.append("")
    lines.append("MORPHOLOGY:")

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

    return "\n".join(lines)


def _plot_field_panels(
    fig: plt.Figure,
    axes_grid: tuple[Any, Any, Any, Any],
    *,
    pm: np.ndarray,
    pc: np.ndarray,
    cc: np.ndarray,
    L: float,
    R: float,
    extent: tuple[float, float, float, float],
) -> None:
    """Draw four imshow panels: φ_m, φ_c, φ_m+φ_c, c (2×2 row-major)."""
    ax_pm, ax_pc, ax_sum, ax_c = axes_grid
    xc = 0.5 * L
    yc = 0.5 * L
    phi_sum = pm + pc
    panels: list[tuple[Any, np.ndarray, str]] = [
        (ax_pm, pm, r"$\phi_m$"),
        (ax_pc, pc, r"$\phi_c$"),
        (ax_sum, phi_sum, r"$\phi_m + \phi_c$"),
        (ax_c, cc, r"$c$"),
    ]

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
    dpi: int = 120,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    include_params_panel: bool = True,
) -> Path:
    """Save a 2×2 field panel (and optional suptitle + monospace params row) to PNG.

    Panels are ``φ_m``, ``φ_c``, ``φ_m + φ_c`` (total crystalline fraction), and ``c``.

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
    extent = (0.0, float(L), 0.0, float(L))

    use_panel = bool(include_params_panel and params is not None)

    if use_panel:
        fig = plt.figure(figsize=(9.0, 12.5), constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[4.0, 4.0, 2.5])
        ax_pm = fig.add_subplot(gs[0, 0])
        ax_pc = fig.add_subplot(gs[0, 1])
        ax_phi_sum = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[1, 1])
        ax_params = fig.add_subplot(gs[2, :])
        ax_params.axis("off")
        _plot_field_panels(
            fig,
            (ax_pm, ax_pc, ax_phi_sum, ax_c),
            pm=pm,
            pc=pc,
            cc=cc,
            L=L,
            R=R,
            extent=extent,
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
            fontsize=8,
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
            L=L,
            R=R,
            extent=extent,
        )
        if title:
            fig.suptitle(title, fontsize=12)

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def write_evolution_gif(
    snapshots: list[tuple[float, np.ndarray]],
    path: Path | str,
    *,
    L: float,
    R: float,
    fps: int = 10,
    field_name: str = "phi_m",
) -> Path | None:
    """Write an animated GIF of scalar field evolution (``phi_m`` slices).

    Parameters
    ----------
    snapshots
        ``(t, field)`` pairs with ``field`` shape ``(n, n)`` on the physical grid.
    path
        Output ``.gif`` path.
    L, R
        Domain size and cavity radius (circle overlay skipped when ``R <= 0``).
    fps
        Frames per second for :class:`matplotlib.animation.PillowWriter`.
    field_name
        Label used in the frame title.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    out = Path(path)
    if not snapshots:
        return None

    stack = np.stack([np.asarray(f, dtype=np.float64) for _, f in snapshots], axis=0)
    vmin = float(np.min(stack))
    vmax = float(np.max(stack))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    first = np.asarray(snapshots[0][1], dtype=np.float64)
    im = ax.imshow(
        first.T,
        origin="lower",
        extent=(0.0, L, 0.0, L),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        aspect="equal",
    )
    if R > 0.0:
        circ = Circle(
            (0.5 * L, 0.5 * L),
            R,
            fill=False,
            linestyle="--",
            edgecolor="w",
            linewidth=1.0,
        )
        ax.add_patch(circ)
    title = ax.set_title(f"{field_name}, t={snapshots[0][0]:.3g}")

    def _update(frame_idx: int):
        t_i, arr = snapshots[frame_idx]
        im.set_data(np.asarray(arr, dtype=np.float64).T)
        title.set_text(f"{field_name}, t={float(t_i):.3g}")
        return (im, title)

    anim = FuncAnimation(fig, _update, frames=len(snapshots), blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def parse_run_stamp_utc(stamp: str) -> str | None:
    """Parse ``YYYYMMDDTHHMMSSZ`` run directory name to ``YYYY-MM-DD HH:MM UTC``."""
    try:
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return None
