"""Compare a new ``final_state.npz`` against an archived run (Pearson on cavity-masked Δφ)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _cavity_chi_grid(*, L: float, R: float, n: int, eps_scale: float = 2.0) -> np.ndarray:
    """Numpy replica of ``circular_cavity_masks`` χ (1 inside cavity, 0 outside)."""
    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    ii = np.arange(n, dtype=np.float64)[:, None]
    jj = np.arange(n, dtype=np.float64)[None, :]
    x_cent = (ii + 0.5) * dx
    y_cent = (jj + 0.5) * dx
    rv = np.sqrt((x_cent - xc) ** 2 + (y_cent - yc) ** 2)
    eps_chi = max(float(eps_scale) * dx, dx)
    return 0.5 * (1.0 - np.tanh((rv - R) / eps_chi))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_run_with_archive(
    new_final_state: Path,
    archive_final_state: Path,
    *,
    L: float,
    R: float,
    output_dir: Path,
    cavity_chi_threshold: float = 0.5,
) -> dict[str, Any]:
    """Load both NPZs, compute Pearson correlation of ``phi_m - phi_c`` on the cavity mask.

    Uses ``chi`` from the new file when present; otherwise rebuilds χ from ``L``, ``R``, ``n``.
    The archive NPZ is expected to use the same array keys (``phi_m``, ``phi_c``) and shape.

    Parameters
    ----------
    new_final_state, archive_final_state
        Paths to ``final_state.npz``.
    L, R
        Domain side and cavity radius (for χ if missing from NPZ).
    output_dir
        Directory for ``comparison_delta_phi.png``.
    cavity_chi_threshold
        Pixels with ``chi > threshold`` are treated as cavity interior.

    Returns
    -------
    dict
        ``correlation``, ``n_pixels_masked``, ``delta_range`` (vmin/vmax for plot),
        ``morphology_notes`` (short string), and ``output_figure`` path.

    Raises
    ------
    ValueError
        If shapes disagree or required arrays are missing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    new_z = np.load(new_final_state)
    arch_z = np.load(archive_final_state)
    for key in ("phi_m", "phi_c"):
        if key not in new_z or key not in arch_z:
            raise ValueError(f"Both NPZs must contain {key!r}")

    phi_m_n = np.asarray(new_z["phi_m"], dtype=np.float64)
    phi_c_n = np.asarray(new_z["phi_c"], dtype=np.float64)
    phi_m_a = np.asarray(arch_z["phi_m"], dtype=np.float64)
    phi_c_a = np.asarray(arch_z["phi_c"], dtype=np.float64)

    if phi_m_n.shape != phi_m_a.shape or phi_m_n.shape != phi_c_n.shape:
        raise ValueError(f"Shape mismatch: new {phi_m_n.shape}, archive {phi_m_a.shape}")

    n = int(phi_m_n.shape[0])
    if phi_m_n.shape[1] != n:
        raise ValueError(f"Expected square grid, got {phi_m_n.shape}")

    if "chi" in new_z.files:
        chi = np.asarray(new_z["chi"], dtype=np.float64)
    else:
        chi = _cavity_chi_grid(L=L, R=R, n=n)
    if chi.shape != phi_m_n.shape:
        raise ValueError(f"chi shape {chi.shape} != field shape {phi_m_n.shape}")

    mask = chi > float(cavity_chi_threshold)
    d_n = (phi_m_n - phi_c_n)[mask]
    d_a = (phi_m_a - phi_c_a)[mask]
    corr = _pearson(d_n, d_a)

    vmin = float(min(d_n.min(), d_a.min()))
    vmax = float(max(d_n.max(), d_a.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axes[0].imshow(phi_m_n - phi_c_n, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("new: φ_m − φ_c")
    im1 = axes[1].imshow(phi_m_a - phi_c_a, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("archive: φ_m − φ_c")
    fig.colorbar(im1, ax=axes, shrink=0.8, label="Δφ")
    for ax in axes:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    fig.suptitle(f"Pearson(masked Δφ) = {corr:.4f}")
    fig.tight_layout()
    fig_path = output_dir / "comparison_delta_phi.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    notes = (
        f"masked pixels={int(mask.sum())}, Δφ new [{d_n.min():.3f},{d_n.max():.3f}], "
        f"archive [{d_a.min():.3f},{d_a.max():.3f}]"
    )
    return {
        "correlation": corr,
        "n_pixels_masked": int(mask.sum()),
        "delta_range": {"vmin": vmin, "vmax": vmax},
        "morphology_notes": notes,
        "output_figure": str(fig_path),
    }
