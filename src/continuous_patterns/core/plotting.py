"""Static figure helpers (NumPy + Matplotlib).

Field PNGs and publication-style panels. Must not import ``io``, ``models``,
or ``experiments`` (``docs/ARCHITECTURE.md`` §3.7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


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
) -> Path:
    """Save a 2×2 panel (``φ_m``, ``φ_c``, ``c``, optional ``χ``) to ``figures_final.png``.

    If ``path`` ends with ``.png``, it is the output file. Otherwise ``path`` is
    treated as a directory and the file is ``path / "figures_final.png"``.
    """
    out = Path(path)
    if out.suffix.lower() != ".png":
        out = out / "figures_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    pm = np.asarray(phi_m, dtype=np.float64)
    pc = np.asarray(phi_c, dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64)
    extent = (0.0, float(L), 0.0, float(L))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.5), constrained_layout=True)
    panels: list[tuple[Any, np.ndarray, str]] = [
        (axes[0, 0], pm, r"$\phi_m$"),
        (axes[0, 1], pc, r"$\phi_c$"),
        (axes[1, 0], cc, r"$c$"),
    ]
    if chi is not None:
        ch = np.asarray(chi, dtype=np.float64)
        panels.append((axes[1, 1], ch, r"$\chi$"))
    else:
        axes[1, 1].axis("off")

    xc = 0.5 * L
    yc = 0.5 * L
    for ax, arr, title in panels:
        im = ax.imshow(
            arr.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title)
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

    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
