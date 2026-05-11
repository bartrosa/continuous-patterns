#!/usr/bin/env python3
"""Patch ``summary.json`` with ``mean_dwell_period``, refresh manifest, regenerate heatmaps.

Reads existing oscillation-window sweep results only (no simulation).

Example::

    uv run python scripts/regenerate_slab_kinetic_window_outputs.py \\
      results/slab_kinetic/sweep_oscillation_window/20260510T151325Z
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load_outputs_module():
    p = _REPO / "scripts" / "slab_kinetic_oscillation_window_outputs.py"
    spec = importlib.util.spec_from_file_location("osc_win_out", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_summaries_and_manifest(sweep_dir: Path) -> int:
    from continuous_patterns.core.diagnostics_kinetic import compute_mean_dwell_period

    sweep_dir = sweep_dir.resolve()
    man_path = sweep_dir / "manifest_axes.json"
    if not man_path.is_file():
        print(f"Missing {man_path}", file=sys.stderr)
        return 1

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    n_patched = 0

    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        rel = e.get("relative_path")
        if not rel:
            continue
        sp = sweep_dir / rel / "summary.json"
        if not sp.is_file():
            continue

        summary = json.loads(sp.read_text(encoding="utf-8"))
        dl_list = summary.get("dwell_L_samples")
        dh_list = summary.get("dwell_H_samples")
        if dl_list is None or dh_list is None:
            continue

        dl = np.asarray(dl_list, dtype=np.float64).ravel()
        dh = np.asarray(dh_list, dtype=np.float64).ravel()
        n_tr = int(summary.get("n_transitions", 0))
        md = compute_mean_dwell_period(dl, dh, n_tr)

        summary["mean_dwell_period"] = float(md["mean_dwell_period"])
        summary["mean_dwell_freq"] = float(md["mean_dwell_freq"])

        sp.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

        e["mean_dwell_period"] = summary["mean_dwell_period"]
        e["mean_dwell_freq"] = summary["mean_dwell_freq"]
        n_patched += 1

    man_path.write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Patched mean_dwell_* for {n_patched} success runs; updated manifest.")

    mod = _load_outputs_module()
    mod.write_oscillation_window_artifacts(sweep_dir, manifest)
    print(f"Wrote figures + table under {sweep_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir",
        type=Path,
        help="Timestamp directory under sweep_oscillation_window",
    )
    args = ap.parse_args(argv)
    return patch_summaries_and_manifest(args.sweep_dir)


if __name__ == "__main__":
    raise SystemExit(main())
