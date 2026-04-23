"""Visualization helper for long Run B HDF5 snapshots.

Loads ``snapshots.h5`` from a ``stage_seq_run_b_long_*`` results directory (as
created by ``run_b_long``), selects approximately log-spaced snapshot indices,
and saves a multi-panel figure of ``phi_m - phi_c`` versus simulation time.

Typical usage::

    uv run python -m continuous_patterns.agate_ch.plot_run_b_time_series
    uv run python -m continuous_patterns.agate_ch.plot_run_b_time_series \\
        --run-dir results/agate_ch/stage_seq_run_b_long_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from continuous_patterns.plot_captions import figure_save_png_with_params


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute ``Path`` to the project root (contains ``results/agate_ch``).
    """
    return Path(__file__).resolve().parents[3]


def find_latest_run_b_long(results_root: Path | None = None) -> Path:
    """Locate the most recently modified long Run B output directory.

    Args:
        results_root: Directory that holds ``stage_seq_run_b_long_*`` folders.
            Defaults to ``<repo>/results/agate_ch``.

    Returns:
        Path to the lexicographically last matching ``stage_seq_run_b_long_*``
        directory (same as newest timestamp if names use ``YYYYMMDD_HHMMSS``).

    Raises:
        FileNotFoundError: If no matching directory exists.
    """
    root = results_root or (_repo_root() / "results" / "agate_ch")
    candidates = sorted(glob.glob(str(root / "stage_seq_run_b_long_*")))
    if not candidates:
        raise FileNotFoundError(
            f"No stage_seq_run_b_long_* directory under {root}. "
            "Use run_b_long.py (it writes stage_seq_run_b_long_<timestamp>/) "
            "or pass --run-dir explicitly."
        )
    return Path(candidates[-1])


def _iter_snap_steps(h5_path: Path) -> list[tuple[int, str]]:
    """List snapshot groups from an agate_ch ``snapshots.h5`` file.

    Args:
        h5_path: Path to ``snapshots.h5`` containing groups ``t_<step>``.

    Returns:
        Sorted list of ``(integer_step, group_name)`` tuples, ascending by step.
    """
    out: list[tuple[int, str]] = []
    with h5py.File(h5_path, "r") as h5:
        for name in h5.keys():
            if name.startswith("t_"):
                step = int(name.split("_", 1)[1])
                out.append((step, name))
    out.sort(key=lambda x: x[0])
    return out


def _pick_log_spaced_indices(n: int, k: int) -> list[int]:
    """Choose up to ``k`` indices from ``range(n)`` with spread favoring early and late times.

    Uses geometric spacing over index position so panels emphasize evolution from
    the first snapshots through late-time states (useful for long runs with many
    frames).

    Args:
        n: Number of available snapshots (length of the sorted snapshot list).
        k: Desired number of panels.

    Returns:
        Sorted unique indices into the snapshot list, length at most ``min(k, n)``.
    """
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # log-spaced in [1..n] mapped to indices
    edges = np.geomspace(1, n, num=k)
    idx = sorted({min(n - 1, max(0, int(round(x)) - 1)) for x in edges})
    # fill if duplicates collapsed count
    while len(idx) < min(k, n):
        for j in range(n):
            if j not in idx:
                idx.append(j)
                break
        idx.sort()
    return sorted(set(idx))[:k]


def plot_time_series(
    run_dir: Path,
    output_path: Path,
    *,
    n_panels: int = 9,
    dt: float | None = None,
) -> None:
    """Render a grid of ``phi_m - phi_c`` fields at selected times.

    Args:
        run_dir: Directory containing ``snapshots.h5`` (and optionally
            ``summary.json`` for ``dt``).
        output_path: Where to write the PNG figure.
        n_panels: Number of subplots; layout is chosen as a near-square grid.
        dt: Time step for axis labels. If ``None``, read ``parameters.dt`` from
            ``run_dir/summary.json``, else default ``0.01``.

    Raises:
        FileNotFoundError: If ``snapshots.h5`` is missing.
        ValueError: If the HDF5 file has no ``t_*`` groups.
    """
    h5_path = run_dir / "snapshots.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"No snapshots.h5 under {run_dir}")

    pairs = _iter_snap_steps(h5_path)
    if not pairs:
        raise ValueError(f"No t_* groups in {h5_path}")

    _dt = dt if dt is not None else _read_dt(run_dir)

    idx_pick = _pick_log_spaced_indices(len(pairs), n_panels)

    nrows = int(np.ceil(np.sqrt(len(idx_pick))))
    ncols = int(np.ceil(len(idx_pick) / nrows))
    fig_w = max(15, 5 * ncols)
    fig_h = max(15, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()

    with h5py.File(h5_path, "r") as h5:
        for ax, i in zip(axes_flat, idx_pick, strict=True):
            step, gname = pairs[i]
            g = h5[gname]
            pm = np.asarray(g["phi_m"])
            pc = np.asarray(g["phi_c"])
            diff = pm - pc
            t_phys = step * _dt
            im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            ax.set_title(f"t = {t_phys:g} (step {step})")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)

    for j in range(len(idx_pick), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r"Long Run B ($\gamma=6$): $\phi_m - \phi_c$ (log-spaced times)",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path = run_dir / "summary.json"
    ts_cfg: dict[str, object] = {
        "script": "plot_run_b_time_series",
        "run_dir": str(run_dir.resolve()),
        "n_panels": n_panels,
        "dt_used": _dt,
    }
    if summ_path.is_file():
        ts_cfg["parameters"] = json.loads(summ_path.read_text()).get("parameters")
    figure_save_png_with_params(fig, output_path, ts_cfg, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _read_dt(run_dir: Path) -> float:
    """Read simulation ``dt`` from ``summary.json`` when present.

    Args:
        run_dir: Run output directory.

    Returns:
        ``dt`` from embedded parameters, or ``0.01`` if unavailable.
    """
    summ = run_dir / "summary.json"
    if summ.is_file():
        data = json.loads(summ.read_text())
        prm = data.get("parameters") or {}
        if "dt" in prm:
            return float(prm["dt"])
    return 0.01


def main() -> None:
    """CLI: plot time series from latest or user-specified long Run B directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Run directory containing snapshots.h5 (default: latest stage_seq_run_b_long_*)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: <run-dir>/time_series_phi_diff.png)",
    )
    ap.add_argument("--panels", type=int, default=9, help="Number of subplot panels (default 9)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_b_long()
    if not run_dir.is_absolute():
        run_dir = _repo_root() / run_dir

    out = Path(args.output) if args.output else (run_dir / "time_series_phi_diff.png")

    plot_time_series(run_dir, out, n_panels=args.panels)


if __name__ == "__main__":
    main()
