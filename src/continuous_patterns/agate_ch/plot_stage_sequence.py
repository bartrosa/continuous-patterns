"""Plot Stage I vs Stage II ``final_state.npz`` fields side by side.

Reads paths from ``results/agate_ch/stage_sequence_latest.json`` (written by
:mod:`continuous_patterns.agate_ch.run_sequence`) or from a user-supplied JSON,
then saves a comparison PNG (default ``stage_sequence_comparison.png``).

Example:
    uv run python -m continuous_patterns.agate_ch.plot_stage_sequence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.agate_ch.run import _repo_root, agate_ch_results_dir
from continuous_patterns.plot_captions import figure_save_png_with_params


def load_final_state(run_dir: Path) -> dict[str, np.ndarray]:
    """Load ``phi_m``, ``phi_c``, and ``c`` from ``run_dir/final_state.npz``.

    Args:
        run_dir: Simulation output directory.

    Returns:
        Dictionary with numpy arrays keyed by field name.

    Raises:
        FileNotFoundError: If ``final_state.npz`` does not exist.
        KeyError: If expected datasets are absent from the archive.
    """
    path = run_dir / "final_state.npz"
    data = np.load(path, allow_pickle=False)
    return {
        "phi_m": np.asarray(data["phi_m"]),
        "phi_c": np.asarray(data["phi_c"]),
        "c": np.asarray(data["c"]),
    }


def plot_comparison(
    run_a_dir: Path,
    run_b_dir: Path,
    output_path: Path,
) -> None:
    """Save a 2×3 figure of ``phi_m``, ``phi_c``, and ``phi_m - phi_c`` for both stages.

    Args:
        run_a_dir: Stage I directory with ``final_state.npz``.
        run_b_dir: Stage II directory with ``final_state.npz``.
        output_path: Destination PNG path.
    """
    state_a = load_final_state(run_a_dir)
    state_b = load_final_state(run_b_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (tag, state) in enumerate((("Stage I", state_a), ("Stage II", state_b))):
        pm, pc = state["phi_m"], state["phi_c"]
        for col, (key, title_suffix) in enumerate(
            (
                ("phi_m", r"$\phi_m$"),
                ("phi_c", r"$\phi_c$"),
                ("phi_diff", r"$\phi_m - \phi_c$"),
            )
        ):
            ax = axes[row, col]
            if key == "phi_diff":
                arr = pm - pc
                im = ax.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
            else:
                arr = state[key]
                im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1, origin="lower")
            ax.set_title(f"{tag}: {title_suffix}")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Sequential Stage I → Stage II (final fields)", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap: dict[str, object] = {
        "script": "plot_stage_sequence",
        "run_a_dir": str(run_a_dir.resolve()),
        "run_b_dir": str(run_b_dir.resolve()),
    }
    for tag, rd in (("stage_I", run_a_dir), ("stage_II", run_b_dir)):
        sj = rd / "summary.json"
        if sj.is_file():
            cap[f"{tag}_parameters"] = json.loads(sj.read_text()).get("parameters")
    figure_save_png_with_params(fig, output_path, cap, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    """CLI: resolve JSON paths and call :func:`plot_comparison`."""
    root = _repo_root()
    ap = argparse.ArgumentParser(
        description="Plot Stage I vs Stage II final fields from stage_sequence_latest.json.",
    )
    ap.add_argument(
        "--json",
        type=str,
        default="",
        help="path to stage_sequence_latest.json (default: results/agate_ch/)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="output PNG path",
    )
    args = ap.parse_args()

    js = Path(args.json) if args.json else agate_ch_results_dir(root) / "stage_sequence_latest.json"
    if not js.is_absolute():
        js = root / js
    if not js.is_file():
        print(f"Missing {js}", file=sys.stderr)
        sys.exit(1)

    seq = json.loads(js.read_text())
    run_a = Path(seq["run_a"])
    run_b = Path(seq["run_b"])

    out = (
        Path(args.output)
        if args.output
        else agate_ch_results_dir(root) / "stage_sequence_comparison.png"
    )
    if not out.is_absolute():
        out = root / out

    plot_comparison(run_a, run_b, out)


if __name__ == "__main__":
    main()
