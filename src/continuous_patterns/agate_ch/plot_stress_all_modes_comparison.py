"""One row × six panels: φ_m − φ_c at final time for stress / anisotropy modes.

Defaults point at historical sweep outputs under ``results/agate_ch``; override with CLI flags.

Panels (left to right):

    stress_off | aniso_10x | flamant_B_1_0 | pure_shear | pressure_gradient | kirsch

Example::

    uv run python -m continuous_patterns.agate_ch.plot_stress_all_modes_comparison \\
        --pure-shear-dir results/agate_ch/stress_extra_sweep_<ts>/pure_shear_sigma_1_0 \\
        --pressure-gradient-dir .../pressure_gradient_sigma_1_0 \\
        --kirsch-dir .../kirsch_sigma_1_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_phi_diff(run_dir: Path) -> np.ndarray:
    npz_path = run_dir / "final_state.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing {npz_path} (need save_final_state and completed run)")
    z = np.load(npz_path)
    pm = np.asarray(z["phi_m"], dtype=np.float64)
    pc = np.asarray(z["phi_c"], dtype=np.float64)
    return pm - pc


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Six-panel φ_m−φ_c comparison.")
    ap.add_argument(
        "--stress-off-dir",
        type=str,
        default="results/agate_ch/stress_sweep_20260423_183637/stress_off",
    )
    ap.add_argument(
        "--aniso-dir",
        type=str,
        default="results/agate_ch/anisotropy_sweep_20260423_180434/aniso_10x",
    )
    ap.add_argument(
        "--flamant-dir",
        type=str,
        default="results/agate_ch/stress_sweep_20260423_183637/flamant_B_1_0",
    )
    ap.add_argument("--pure-shear-dir", type=str, required=True)
    ap.add_argument("--pressure-gradient-dir", type=str, required=True)
    ap.add_argument("--kirsch-dir", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="results/agate_ch/stress_all_modes_comparison.png",
        help="Output PNG path (relative to repo unless absolute).",
    )
    args = ap.parse_args(argv)

    specs = [
        ("stress_off", root / args.stress_off_dir),
        ("aniso_10x", root / args.aniso_dir),
        ("flamant_B_1_0", root / args.flamant_dir),
        ("pure_shear", root / args.pure_shear_dir),
        ("pressure_gradient", root / args.pressure_gradient_dir),
        ("kirsch", root / args.kirsch_dir),
    ]

    diffs = [load_phi_diff(d) for _, d in specs]
    vmax = max(float(np.percentile(np.abs(d), 99.5)) for d in diffs) or 1.0

    fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.2), constrained_layout=True)
    for ax, (title, _), arr in zip(axes, specs, diffs, strict=True):
        im = ax.imshow(
            arr.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label=r"$\phi_m - \phi_c$")
    fig.suptitle("Final-time phase contrast (six modes)", y=1.02, fontsize=12)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
