"""Comparison figures for Experiment 4 (``c0_alpha`` gravity sweep).

Reads run directories (from ``manifest.json`` or explicit ``--dirs``), loads the
final snapshot from each ``snapshots.h5``, and writes:

- ``gravity_comparison.png`` — 3×N: antiphase field, horizontal slice at
  ``y=L/2``, vertical slice at ``x=L/2``
- ``gravity_vertical_profiles.png`` — 1×N: vertical profiles only at ``x = L/2``

Example (manifest from a sweep)::

    uv run python -m continuous_patterns.agate_ch.plot_gravity_sweep \\
        --manifest results/agate_ch/gravity_sweep_YYYYMMDD_HHMMSS/manifest.json

Combined six-way figure after a partial sweep — pass directories in ascending
α order::

    uv run python -m continuous_patterns.agate_ch.plot_gravity_sweep --dirs \\
      results/agate_ch/gravity_sweep_OLD/alpha_0_00 \\
      ... \\
      results/agate_ch/gravity_sweep_NEW/alpha_0_80
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


def _load_final_diff(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(str(x).split("_")[1]))
        g = h5[keys[-1]]
        pm = np.asarray(g["phi_m"], dtype=np.float64)
        pc = np.asarray(g["phi_c"], dtype=np.float64)
    return pm - pc, pm, pc


_LABEL_BY_RUN_DIR = {
    "alpha_0_00": r"$\alpha=0$",
    "alpha_0_05": r"$\alpha=0.05$",
    "alpha_0_10": r"$\alpha=0.10$",
    "alpha_0_20": r"$\alpha=0.20$",
    "alpha_0_40": r"$\alpha=0.40$",
    "alpha_0_80": r"$\alpha=0.80$",
}


def _labels_for_run_dirs(run_dirs: list[Path]) -> list[str]:
    """Title from folder name ``alpha_*`` so merged ``--dirs`` stays correct."""
    out: list[str] = []
    for p in run_dirs:
        key = p.name
        out.append(_LABEL_BY_RUN_DIR.get(key, key))
    return out


def _center_index(L: float, n: int) -> int:
    dx = L / n
    xs = (np.arange(n, dtype=np.float64) + 0.5) * dx
    i_mid = int(np.argmin(np.abs(xs - L / 2)))
    return i_mid


def plot_sweep(
    run_dirs: list[Path],
    labels: list[str],
    out_dir: Path,
    *,
    dpi: int = 140,
) -> None:
    if len(run_dirs) != len(labels):
        raise ValueError("run_dirs and labels length mismatch")
    ncols = len(run_dirs)
    if ncols == 0:
        raise ValueError("no runs")

    diffs: list[np.ndarray] = []
    horiz: list[np.ndarray] = []
    vert: list[np.ndarray] = []
    L_f = 200.0

    for rdir in run_dirs:
        summ_path = rdir / "summary.json"
        h5_path = rdir / "snapshots.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        if summ_path.is_file():
            data = json.loads(summ_path.read_text())
            prm = data.get("parameters") or {}
            L_f = float(prm.get("L", L_f))
        diff, _, _ = _load_final_diff(h5_path)
        diffs.append(diff)
        idx = _center_index(L_f, diff.shape[0])
        # ``xy_grid`` indexing ``ij``: axis 0 is x, axis 1 is y.
        horiz.append(np.asarray(diff[:, idx]))
        vert.append(np.asarray(diff[idx, :]))

    nx, ny = diffs[0].shape[0], diffs[0].shape[1]
    x_line = (np.arange(nx) + 0.5) * (L_f / nx)
    y_line = (np.arange(ny) + 0.5) * (L_f / ny)

    fig0, axes0 = plt.subplots(3, ncols, figsize=(3.2 * ncols, 10.0))
    if ncols == 1:
        axes0 = axes0.reshape(3, 1)
    for j in range(ncols):
        ax = axes0[0, j]
        im = ax.imshow(
            diffs[j].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            aspect="equal",
        )
        ax.set_title(labels[j])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes0[1, j]
        ax.plot(x_line, horiz[j], color="tab:blue", lw=1.2)
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel(r"$\phi_m-\phi_c$ @ $y=L/2$")
        ax.set_xlim(0.0, L_f)
        ax.grid(True, alpha=0.3)
        ax = axes0[2, j]
        ax.plot(vert[j], y_line, color="tab:green", lw=1.2)
        ax.set_xlabel(r"$\phi_m-\phi_c$")
        if j == 0:
            ax.set_ylabel("y")
        ax.set_title("")  # column title already on top row
        ax.set_ylim(0.0, L_f)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    fig0.suptitle(
        r"Experiment 4 — rim $c_0(y)$ gradient: antiphase | "
        r"$y=L/2$ slice | $x=L/2$ slice (vertical)"
    )
    fig0.tight_layout()
    p0 = out_dir / "gravity_comparison.png"
    fig0.savefig(p0, dpi=dpi, bbox_inches="tight")
    plt.close(fig0)

    fig1, axes1 = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.8))
    if ncols == 1:
        axes1 = np.array([axes1])
    for j in range(ncols):
        ax = axes1[j]
        ax.plot(vert[j], y_line, color="tab:green", lw=1.2)
        ax.set_ylabel("y")
        ax.set_xlabel(r"$\phi_m-\phi_c$")
        ax.set_title(labels[j])
        ax.set_ylim(0.0, L_f)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    fig1.suptitle(r"Vertical profiles at $x=L/2$ (top vs bottom asymmetry)")
    fig1.tight_layout()
    p1 = out_dir / "gravity_vertical_profiles.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    print(f"Saved:\n  {p0}\n  {p1}")


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="JSON manifest from run_gravity_sweep (contains ordered run dirs)",
    )
    ap.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help=(
            "Run directories (repo-relative or absolute); order by ascending α "
            "(six folders for the full gravity set)"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for PNGs (default: results/agate_ch next to sweep or cwd)",
    )
    args = ap.parse_args(argv)

    if args.dirs:
        dirs = [Path(d) if Path(d).is_absolute() else root / d for d in args.dirs]
        labels = _labels_for_run_dirs(dirs)
        if args.output_dir:
            out = Path(args.output_dir)
            if not out.is_absolute():
                out = root / out
        else:
            out = root / "results" / "agate_ch"
    elif args.manifest:
        mp = Path(args.manifest)
        if not mp.is_absolute():
            mp = root / mp
        man = json.loads(mp.read_text())
        dirs = []
        for item in man["runs"]:
            rel = item["out_dir"]
            dirs.append(root / rel)
        labels = _labels_for_run_dirs(dirs)
        out = root / "results" / "agate_ch"
        if args.output_dir:
            out = Path(args.output_dir)
            if not out.is_absolute():
                out = root / out
    else:
        print("Provide --manifest or --dirs", file=sys.stderr)
        sys.exit(2)

    out.mkdir(parents=True, exist_ok=True)
    plot_sweep(dirs, labels, out)


if __name__ == "__main__":
    main()
