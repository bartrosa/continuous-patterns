#!/usr/bin/env python3
"""Embed NumPy oscillation-window reference cells into ``scripts/data/``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_OUT = _REPO / "scripts" / "data" / "oscillation_window_numpy_reference.py"


def _mean_dwell_from_summary(summary: dict, n_tr: int) -> float | None:
    from continuous_patterns.core.diagnostics_kinetic import compute_mean_dwell_period

    dl_list = summary.get("dwell_L_samples")
    dh_list = summary.get("dwell_H_samples")
    if dl_list is None or dh_list is None:
        return None
    md = compute_mean_dwell_period(dl_list, dh_list, n_tr)
    val = md.get("mean_dwell_period")
    return float(val) if val is not None else None


def _metrics_for_entry(sweep_dir: Path, entry: dict) -> tuple[int, float] | None:
    n_tr = entry.get("n_transitions")
    md = entry.get("mean_dwell_period")
    rel = entry.get("relative_path")
    summary: dict | None = None
    if rel:
        sp = sweep_dir / rel / "summary.json"
        if sp.is_file():
            summary = json.loads(sp.read_text(encoding="utf-8"))
            if n_tr is None:
                n_tr = summary.get("n_transitions")
            if md is None:
                md = summary.get("mean_dwell_period")

    if n_tr is None:
        return None
    n_tr = int(n_tr)
    if md is None and summary is not None:
        md = _mean_dwell_from_summary(summary, n_tr)
    if md is None:
        return None
    return n_tr, float(md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write hardcoded NumPy reference cells from a sweep manifest."
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Directory containing manifest_axes.json from the NumPy window sweep.",
    )
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    man_path = sweep_dir / "manifest_axes.json"
    if not man_path.is_file():
        print(f"Missing {man_path}", file=sys.stderr)
        return 1

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    backend = str(manifest.get("backend", "numpy")).lower()
    if backend == "jax":
        print(
            "ERROR: manifest backend is jax — embed NumPy reference from a numpy sweep, "
            "not the JAX batch output.",
            file=sys.stderr,
        )
        return 1

    cells: dict[tuple[float, float], dict[str, float | int]] = {}
    n_missing = 0
    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        c1 = float(e["c1"])
        c2 = float(e["c2"])
        metrics = _metrics_for_entry(sweep_dir, e)
        if metrics is None:
            n_missing += 1
            continue
        n_tr, md = metrics
        if not math.isfinite(md):
            n_missing += 1
            continue
        cells[(c1, c2)] = {
            "n_transitions": n_tr,
            "mean_dwell_period": md,
        }
    if n_missing:
        print(f"WARNING: skipped {n_missing} success entries without metrics", file=sys.stderr)

    lines = [
        '"""NumPy reference metrics for the 10×10 oscillation-window sweep (T=20, Da=1, κ=3).',
        "",
        f"Source manifest: ``{sweep_dir}``",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "REFERENCE_CELLS: dict[tuple[float, float], dict[str, float | int]] = {",
    ]
    for (c1, c2), vals in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append(
            f"    ({c1:.2f}, {c2:.2f}): "
            f'{{"n_transitions": {int(vals["n_transitions"])}, '
            f'"mean_dwell_period": {float(vals["mean_dwell_period"]):.10g}}},'
        )
    lines.extend(
        [
            "}",
            "",
            f'REFERENCE_SOURCE = "{sweep_dir}"',
            "",
        ]
    )
    _OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cells)} reference cells to {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
