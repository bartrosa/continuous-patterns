#!/usr/bin/env python3
"""Validate JAX oscillation-window sweep against hardcoded NumPy reference cells."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_reference() -> tuple[dict[tuple[float, float], dict[str, float | int]], str]:
    ref_path = _REPO / "scripts" / "data" / "oscillation_window_numpy_reference.py"
    spec = importlib.util.spec_from_file_location("osc_ref", ref_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ref_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cells = getattr(mod, "REFERENCE_CELLS", {})
    source = getattr(mod, "REFERENCE_SOURCE", str(ref_path))
    if not isinstance(cells, dict):
        cells = {}
    if len(cells) < 10:
        print(
            f"WARNING: reference table has {len(cells)} cells (expected ~62) — run:\n"
            "  uv run python scripts/embed_oscillation_window_reference.py "
            "results/slab_kinetic/sweep_oscillation_window/<numpy_timestamp>\n"
            "NumPy sweep may be restored from Trash if it was deleted.",
            file=sys.stderr,
        )
    return cells, str(source)


def _n_trans_ok(obs: int, ref: int) -> bool:
    tol = max(2, int(round(0.02 * ref)))
    return abs(obs - ref) <= tol


def _dwell_ok(obs: float, ref: float) -> bool:
    if ref <= 0 or not (obs == obs):
        return False
    return abs(obs - ref) / ref <= 0.02


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate JAX window sweep vs NumPy reference.")
    parser.add_argument("sweep_dir", type=Path, help="JAX sweep output directory.")
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    man_path = sweep_dir / "manifest_axes.json"
    if not man_path.is_file():
        print(f"Missing {man_path}", file=sys.stderr)
        return 1

    reference, ref_source = _load_reference()
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    n_pass = 0
    n_fail = 0
    n_skip = 0
    print(f"Reference source: {ref_source} ({len(reference)} cells)")
    print("| c1 | c2 | n_trans (jax/ref) | mean_dwell (jax/ref) | status |")
    print("|----|----|-------------------|----------------------|--------|")

    for e in manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        c1 = float(e["c1"])
        c2 = float(e["c2"])
        key = (round(c1, 2), round(c2, 2))
        ref = reference.get(key)
        if ref is None:
            for k, v in reference.items():
                if abs(k[0] - c1) < 1e-8 and abs(k[1] - c2) < 1e-8:
                    ref = v
                    break
        if ref is None:
            n_skip += 1
            print(f"| {c1} | {c2} | — | — | SKIP (no reference) |")
            continue

        obs_n = int(e.get("n_transitions", -1))
        obs_md = float(e.get("mean_dwell_period", float("nan")))
        ref_n = int(ref["n_transitions"])
        ref_md = float(ref["mean_dwell_period"])
        ok = _n_trans_ok(obs_n, ref_n) and _dwell_ok(obs_md, ref_md)
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        print(f"| {c1} | {c2} | {obs_n}/{ref_n} | {obs_md:.4g}/{ref_md:.4g} | {status} |")

    print(f"\nSummary: PASS={n_pass} FAIL={n_fail} SKIP={n_skip}")
    if n_pass == 0 and n_fail == 0 and n_skip > 0:
        return 2
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
