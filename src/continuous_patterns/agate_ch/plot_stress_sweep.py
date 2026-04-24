"""Experiment 6 figures: stress overview (3×N) and Flamant diagnostics (1×3).

Reads ``manifest.json`` from :mod:`run_stress_sweep` or explicit ``--dirs``, rebuilds
σ fields from each run's ``summary.json`` parameters, and loads final ``phi_m``,
``phi_c`` from ``snapshots.h5``.

Writes (default under ``results/agate_ch/``):

- ``stress_overview.png`` — rows: ``σ_xx−σ_yy``, ``φ_m−φ_c``, overlay contours
- ``stress_field_diagnostics.png`` — centerline profiles + principal stress (uses
  the strongest-σ run: ``flamant_B_2_0`` when present).

Example::

    uv run python -m continuous_patterns.agate_ch.plot_stress_sweep \\
        --manifest results/agate_ch/stress_sweep_<ts>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.agate_ch.solver import build_geometry_from_cfg
from continuous_patterns.agate_ch.stress_fields import principal_sigma_max


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_final_diff(h5_path: Path) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(str(x).split("_")[1]))
        g = h5[keys[-1]]
        pm = np.asarray(g["phi_m"], dtype=np.float64)
        pc = np.asarray(g["phi_c"], dtype=np.float64)
    return pm - pc


def _sigma_maps_from_params(prm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``Δσ``, ``σ_xx``, ``σ_yy``, ``σ_xy`` on cell-centred grid."""
    geom = build_geometry_from_cfg(prm)
    sxx = np.asarray(jax.device_get(geom.sigma_xx), dtype=np.float64)
    syy = np.asarray(jax.device_get(geom.sigma_yy), dtype=np.float64)
    sxy = np.asarray(jax.device_get(geom.sigma_xy), dtype=np.float64)
    return sxx - syy, sxx, syy, sxy


def _center_index(L: float, n: int, target: float) -> int:
    xs = (np.arange(n, dtype=np.float64) + 0.5) * (L / n)
    return int(np.argmin(np.abs(xs - target)))


def plot_overview(
    run_dirs: list[Path],
    labels: list[str],
    prms: list[dict],
    out_path: Path,
    *,
    dpi: int = 140,
) -> None:
    ncols = len(run_dirs)
    L = float(prms[0]["L"])
    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10.5))
    if ncols == 1:
        axes = axes.reshape(3, 1)

    diffs: list[np.ndarray] = []
    d_sigmas: list[np.ndarray] = []

    for j in range(ncols):
        dsig, _, _, _ = _sigma_maps_from_params(prms[j])
        d_sigmas.append(dsig)
        diffs.append(_load_final_diff(run_dirs[j] / "snapshots.h5"))

    vmax_s = max(float(np.percentile(np.abs(d), 99.0)) for d in d_sigmas if np.any(d != 0)) or 1.0

    for j in range(ncols):
        dsig = d_sigmas[j]
        diff = diffs[j]

        ax0 = axes[0, j]
        if np.max(np.abs(dsig)) < 1e-14:
            ax0.imshow(
                np.zeros_like(dsig).T,
                origin="lower",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                extent=[0, L, 0, L],
                aspect="equal",
            )
            ax0.set_title(labels[j] + r" ($\Delta\sigma\approx 0$)")
        else:
            im0 = ax0.imshow(
                dsig.T,
                origin="lower",
                cmap="coolwarm",
                vmin=-vmax_s,
                vmax=vmax_s,
                extent=[0, L, 0, L],
                aspect="equal",
            )
            plt.colorbar(im0, ax=ax0, fraction=0.046)
            ax0.set_title(labels[j] + r"  $\sigma_{xx}-\sigma_{yy}$")
        ax0.set_xlabel("x")
        if j == 0:
            ax0.set_ylabel("y")

        ax1 = axes[1, j]
        im1 = ax1.imshow(
            diff.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            extent=[0, L, 0, L],
            aspect="equal",
        )
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        ax1.set_title(r"$\phi_m-\phi_c$")
        ax1.set_xlabel("x")
        if j == 0:
            ax1.set_ylabel("y")

        ax2 = axes[2, j]
        bg = dsig if np.max(np.abs(dsig)) > 1e-14 else np.zeros_like(diff)
        vbg = max(float(np.percentile(np.abs(bg), 99.0)), 1e-9)
        im2 = ax2.imshow(
            bg.T,
            origin="lower",
            cmap="viridis",
            vmin=-vbg,
            vmax=vbg,
            extent=[0, L, 0, L],
            aspect="equal",
            alpha=0.85,
        )
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        cs = ax2.contour(
            diff.T,
            levels=np.linspace(-0.9, 0.9, 10),
            colors="white",
            linewidths=0.6,
            extent=[0, L, 0, L],
            origin="lower",
        )
        ax2.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
        ax2.set_title("overlay: Δσ background + φ contours")
        ax2.set_xlabel("x")
        if j == 0:
            ax2.set_ylabel("y")

    fig.suptitle("Experiment 6 — Flamant stress vs antiphase fields")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_diagnostics(prm: dict, out_path: Path, *, dpi: int = 140) -> None:
    """Horizontal / vertical σ profiles and principal stress map for one parameter set."""
    L = float(prm["L"])
    n = int(prm["grid"])
    dsig, sxx, syy, sxy = _sigma_maps_from_params(prm)
    p1 = np.asarray(
        jax.device_get(
            principal_sigma_max(
                jnp.asarray(sxx, dtype=jnp.float32),
                jnp.asarray(syy, dtype=jnp.float32),
                jnp.asarray(sxy, dtype=jnp.float32),
            )
        ),
        dtype=np.float64,
    )

    ix = _center_index(L, n, L / 2)
    iy = _center_index(L, n, L / 2)
    x_line = (np.arange(n) + 0.5) * (L / n)
    y_line = (np.arange(n) + 0.5) * (L / n)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    axes[0].plot(x_line, sxx[:, iy], color="tab:blue", lw=1.2, label=r"$\sigma_{xx}$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\sigma_{xx}$ at $y=L/2$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_line, syy[ix, :], color="tab:orange", lw=1.2, label=r"$\sigma_{yy}$")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(r"$\sigma_{yy}$ at $x=L/2$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    vmax = float(np.percentile(np.abs(p1), 99.5))
    im = axes[2].imshow(
        p1.T,
        origin="lower",
        cmap="magma",
        vmin=-vmax,
        vmax=vmax,
        extent=[0, L, 0, L],
        aspect="equal",
    )
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[2].set_title(r"Larger principal stress $\sigma_1$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.suptitle(r"Experiment 6 — stress field diagnostics (Flamant two-point)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=str, default="", help="Sweep manifest JSON")
    ap.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help="Run directories in order: stress_off, flamant_B_0_5, …",
    )
    ap.add_argument(
        "--output-overview",
        type=str,
        default="",
        help="Path for 3-row overview PNG",
    )
    ap.add_argument(
        "--output-diagnostics",
        type=str,
        default="",
        help="Path for 1×3 diagnostics PNG",
    )
    args = ap.parse_args(argv)

    run_dirs: list[Path] = []
    labels: list[str] = []
    prms: list[dict] = []

    if args.dirs:
        for d in args.dirs:
            p = Path(d) if Path(d).is_absolute() else root / d
            run_dirs.append(p)
            summ = json.loads((p / "summary.json").read_text())
            prms.append(summ["parameters"])
            labels.append(p.name)
    elif args.manifest:
        mp = Path(args.manifest)
        if not mp.is_absolute():
            mp = root / mp
        man = json.loads(mp.read_text())
        for item in man["runs"]:
            p = root / item["out_dir"]
            run_dirs.append(p)
            summ = json.loads((p / "summary.json").read_text())
            prms.append(summ["parameters"])
            labels.append(item.get("id", p.name))
    else:
        print("Provide --manifest or --dirs", file=sys.stderr)
        sys.exit(2)

    out1 = (
        Path(args.output_overview)
        if args.output_overview
        else root / "results" / "agate_ch" / "stress_overview.png"
    )
    if not out1.is_absolute():
        out1 = root / out1
    out2 = (
        Path(args.output_diagnostics)
        if args.output_diagnostics
        else root / "results" / "agate_ch" / "stress_field_diagnostics.png"
    )
    if not out2.is_absolute():
        out2 = root / out2

    plot_overview(run_dirs, labels, prms, out1)

    diag_prm = next(
        (prm for prm, d in zip(prms, run_dirs, strict=True) if d.name == "flamant_B_2_0"),
        prms[-1],
    )
    plot_diagnostics(diag_prm, out2)


if __name__ == "__main__":
    main()
