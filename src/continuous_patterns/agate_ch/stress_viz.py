"""Static stress-tensor maps and principal-direction overlays (Experiment 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.agate_ch.solver import build_geometry_from_cfg


def save_stress_mode_diagnostic(cfg: dict[str, Any], out_path: Path, *, dpi: int = 130) -> None:
    """Write ``σ_xx−σ_yy`` heatmap + quiver of major principal stress direction.

    Parameters
    ----------
    cfg
        Flat agate_ch config (as in ``summary.json`` ``parameters``).
    out_path
        e.g. ``.../stress_diagnostic_pure_shear.png``
    """
    L = float(cfg["L"])
    n = int(cfg["grid"])
    geom = build_geometry_from_cfg(cfg)
    sxx = np.asarray(jax.device_get(geom.sigma_xx), dtype=np.float64)
    syy = np.asarray(jax.device_get(geom.sigma_yy), dtype=np.float64)
    sxy = np.asarray(jax.device_get(geom.sigma_xy), dtype=np.float64)
    dsg = sxx - syy

    # Major principal direction: angle φ with tan(2φ) = 2τ / (σ_xx - σ_yy)
    ang = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    u = np.cos(ang)
    v = np.sin(ang)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.5))
    vmax = float(np.percentile(np.abs(dsg), 99.5)) or 1.0
    im = ax.imshow(
        dsg.T,
        origin="lower",
        extent=[0, L, 0, L],
        aspect="equal",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\sigma_{xx}-\sigma_{yy}$")

    step = max(n // 32, 1)
    xs = (np.arange(n) + 0.5) * (L / n)
    ys = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    qv = ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="k",
        alpha=0.45,
        scale=25.0,
        width=0.0018,
    )
    ax.quiverkey(qv, 0.9, 1.02, 0.4, "major principal", labelpos="E", coordinates="axes")
    sm = str(cfg.get("stress_mode", "none"))
    ax.set_title(f"Stress tensor  {sm}  (σ₀={cfg.get('sigma_0', 0)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
