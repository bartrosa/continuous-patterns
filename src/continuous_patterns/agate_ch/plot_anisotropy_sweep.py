"""Comparison figure for Experiment 5a (anisotropic ``kappa_x``, ``kappa_y``).

Loads the final snapshot from each run's ``snapshots.h5`` and writes:

- ``anisotropy_comparison.png`` — 2×N: antiphase field + log₁₀(|FFT(φ_m−φ_c)|²) (shifted).

Example::

    uv run python -m continuous_patterns.agate_ch.plot_anisotropy_sweep \\
        --manifest results/agate_ch/anisotropy_sweep_YYYYMMDD_HHMMSS/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


_LABEL_BY_DIR = {
    "iso": r"isotropic $\kappa_x=\kappa_y$",
    "aniso_2x": r"$\kappa_x/\kappa_y=2$",
    "aniso_5x": r"$\kappa_x/\kappa_y=5$",
    "aniso_10x": r"$\kappa_x/\kappa_y=10$",
}


def _labels_for_dirs(run_dirs: list[Path]) -> list[str]:
    return [_LABEL_BY_DIR.get(p.name, p.name) for p in run_dirs]


def plot_sweep(
    run_dirs: list[Path],
    labels: list[str],
    out_path: Path,
    *,
    dpi: int = 140,
    eps_fft: float = 1e-18,
) -> None:
    if len(run_dirs) != len(labels):
        raise ValueError("run_dirs and labels length mismatch")
    ncols = len(run_dirs)
    if ncols == 0:
        raise ValueError("no runs")

    diffs: list[np.ndarray] = []
    ffts: list[np.ndarray] = []

    for rdir in run_dirs:
        h5_path = rdir / "snapshots.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        diff = _load_final_diff(h5_path)
        diffs.append(diff)
        ft = np.fft.fft2(diff)
        pow2 = np.abs(ft) ** 2
        ffts.append(np.fft.fftshift(np.log10(pow2 + eps_fft)))

    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 7.2))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for j in range(ncols):
        ax0 = axes[0, j]
        im0 = ax0.imshow(
            diffs[j].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            aspect="equal",
        )
        ax0.set_title(labels[j])
        ax0.set_xticks([])
        ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        ax1 = axes[1, j]
        f1 = ffts[j]
        vmax = float(np.percentile(f1, 99.5))
        vmin = float(np.percentile(f1, 0.5))
        im1 = ax1.imshow(f1.T, origin="lower", cmap="magma", aspect="equal", vmin=vmin, vmax=vmax)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(r"$\log_{10}$ $|\hat\phi|^2$")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

    fig.suptitle(
        r"Experiment 5a — anisotropic gradient energy: antiphase ($\phi_m-\phi_c$) "
        r"and spectral power (FFT magnitude squared, log scale)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="JSON manifest from run_anisotropy_sweep",
    )
    ap.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help="Run directories (repo-relative or ascending order: iso, aniso_2x, …)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: results/agate_ch/anisotropy_comparison.png)",
    )
    args = ap.parse_args(argv)

    if args.dirs:
        dirs = [Path(d) if Path(d).is_absolute() else root / d for d in args.dirs]
        labels = _labels_for_dirs(dirs)
    elif args.manifest:
        mp = Path(args.manifest)
        if not mp.is_absolute():
            mp = root / mp
        man = json.loads(mp.read_text())
        dirs = [root / item["out_dir"] for item in man["runs"]]
        labels = _labels_for_dirs(dirs)
    else:
        print("Provide --manifest or --dirs", file=sys.stderr)
        sys.exit(2)

    outp = (
        Path(args.output)
        if args.output
        else root / "results" / "agate_ch" / "anisotropy_comparison.png"
    )
    if not outp.is_absolute():
        outp = root / outp
    plot_sweep(dirs, labels, outp)


if __name__ == "__main__":
    main()
