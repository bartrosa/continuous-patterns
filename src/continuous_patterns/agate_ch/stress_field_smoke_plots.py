"""Save standalone diagnostic PNGs for the three analytic stress tensors (no simulation).

Output: ``results/agate_ch/smoke_stress_fields/`` — one panel per mode
(σ_xx−σ_yy, σ_xy, optional cross-sections).

Example::

    uv run python -m continuous_patterns.agate_ch.stress_field_smoke_plots
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continuous_patterns.agate_ch.stress_fields import (
    kirsch_field,
    pressure_gradient_field,
    pure_shear_field,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = _repo_root()
    out_dir = root / "results" / "agate_ch" / "smoke_stress_fields"
    out_dir.mkdir(parents=True, exist_ok=True)

    L, n = 200.0, 256
    R = 80.0
    sigma_0 = 1.0

    def plot_row(
        label: str,
        sxx: jnp.ndarray,
        syy: jnp.ndarray,
        sxy: jnp.ndarray,
    ) -> None:
        sxx_ = np.asarray(jax.device_get(sxx))
        syy_ = np.asarray(jax.device_get(syy))
        sxy_ = np.asarray(jax.device_get(sxy))
        d = sxx_ - syy_
        fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.6))
        titles = (r"$\sigma_{xx}-\sigma_{yy}$", r"$\sigma_{xy}$", r"$\sigma_{xx}$")
        cmaps = ("coolwarm", "coolwarm", "viridis")
        for a, dat, ttl, cmap in zip(ax, (d, sxy_, sxx_), titles, cmaps, strict=True):
            vmax = float(np.percentile(np.abs(dat), 99.5)) or 1.0
            if ttl == r"$\sigma_{xx}$":
                im = a.imshow(dat.T, origin="lower", extent=[0, L, 0, L], cmap=cmap)
            else:
                vm = vmax
                im = a.imshow(
                    dat.T,
                    origin="lower",
                    extent=[0, L, 0, L],
                    cmap=cmap,
                    vmin=-vm,
                    vmax=vm,
                )
            plt.colorbar(im, ax=a, fraction=0.046)
            a.set_title(ttl)
            a.set_aspect("equal")
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(out_dir / f"{label.replace(' ', '_')}.png", dpi=140)
        plt.close(fig)

    ps = pure_shear_field(L, n, sigma_0)
    plot_row("pure_shear", *ps)

    pg = pressure_gradient_field(L, n, sigma_0)
    plot_row("pressure_gradient", *pg)

    kf = kirsch_field(L, R, n, sigma_0)
    plot_row("kirsch", *kf)

    print(f"Smoke stress field PNGs written under {out_dir}")


if __name__ == "__main__":
    main()
