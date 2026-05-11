#!/usr/bin/env python3
"""Regenerate slab_kinetic ``figures_final.png`` from existing sweep artifacts.

Example::

    uv run python scripts/regenerate_slab_kinetic_figures.py \\
      results/slab_kinetic/sweep_parametric/20260510T083051Z \\
      --only-run Da_1__kappa_3__threshold_pair_c1_0p4__c2_0p26

After validating layouts, rerun without ``--only-run`` and pass ``--summary``
to rebuild ``figure_sweep_summary.png`` (does not re-run simulations).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_plotting():
    from continuous_patterns.core.plotting import plot_slab_kinetic_figures_final

    return plot_slab_kinetic_figures_final


def _load_summary_plotter():
    gen_path = _REPO_ROOT / "scripts" / "generate_slab_kinetic_summary_figures.py"
    spec = importlib.util.spec_from_file_location("slab_kinetic_summary_gen", gen_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {gen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.plot_sweep_summary


def _iter_run_roots(sweep_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for summary_path in sorted(sweep_dir.rglob("summary.json")):
        roots.append(summary_path.parent)
    return roots


def _filter_runs(roots: list[Path], sweep_dir: Path, needle: str | None) -> list[Path]:
    if not needle:
        return roots
    out: list[Path] = []
    for r in roots:
        rel = str(r.relative_to(sweep_dir))
        if needle in rel or needle in r.name:
            out.append(r)
    return out


def regenerate_run(run_dir: Path, plot_fn) -> None:
    cfg_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    hist_path = run_dir / "history.npz"
    snap_path = run_dir / "snapshots.npz"
    for p in (cfg_path, summary_path, hist_path, snap_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hist = np.load(hist_path)
    snap = np.load(snap_path)

    transitions: list[tuple[float, str]] = []
    for item in summary.get("kinetic_transitions") or []:
        if isinstance(item, dict):
            transitions.append((float(item["t"]), str(item["kind"])))

    diagnostics = {k: v for k, v in summary.items() if k != "kinetic_transitions"}

    ts = run_dir.name
    slug = run_dir.parent.name
    title = f"{slug} — {ts}"

    plot_fn(
        run_dir,
        t_hist=np.asarray(hist["t_history"]),
        c0_hist=np.asarray(hist["c_at_x0_history"]),
        st_hist=np.asarray(hist["state_history"]),
        th_hist=np.asarray(hist["thickness_history"]),
        snapshots_t=np.asarray(snap["snapshots_t"]),
        snapshots_c=np.asarray(snap["snapshots_c"]),
        c1=float(cfg["physics"]["c1"]),
        c2=float(cfg["physics"]["c2"]),
        transitions=transitions,
        cfg=cfg,
        diagnostics=diagnostics,
        title=title,
        include_params_panel=True,
        dpi=120,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir",
        type=Path,
        help="Sweep timestamp directory containing run subfolders",
    )
    ap.add_argument(
        "--only-run",
        metavar="SUBSTRING",
        default=None,
        help="Only regenerate runs whose relative path or folder name contains this string",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Also regenerate figure_sweep_summary.png from manifest_axes.json",
    )
    args = ap.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        print(f"Not a directory: {sweep_dir}", file=sys.stderr)
        return 1

    plot_fn = _load_plotting()
    roots = _filter_runs(_iter_run_roots(sweep_dir), sweep_dir, args.only_run)
    if not roots:
        print("No matching run directories (need summary.json under sweep_dir).", file=sys.stderr)
        return 1

    for run_dir in roots:
        regenerate_run(run_dir, plot_fn)
        print(f"OK {run_dir.relative_to(sweep_dir)}")

    if args.summary:
        man_path = sweep_dir / "manifest_axes.json"
        if not man_path.is_file():
            print(f"Missing {man_path}; skipping summary figure.", file=sys.stderr)
            return 1
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        plot_sweep_summary = _load_summary_plotter()
        plot_sweep_summary(sweep_dir, manifest)
        print(f"Wrote {sweep_dir / 'figure_sweep_summary.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
