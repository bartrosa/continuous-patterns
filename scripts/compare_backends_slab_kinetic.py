#!/usr/bin/env python3
"""Compare slab_kinetic NumPy vs JAX backends on one config."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.models.slab_kinetic import simulate  # noqa: E402


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: dict[str, Any] = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = copy.deepcopy(value)


def _run_backend(cfg: dict[str, Any], backend: str) -> tuple[dict[str, Any], float]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("solver", {})
    run_cfg["solver"]["backend"] = backend
    t0 = time.perf_counter()
    res = simulate(run_cfg, show_progress=False)
    wall = time.perf_counter() - t0
    diag = res.diagnostics if isinstance(res.diagnostics, dict) else {}
    c = np.asarray(res.state_final.c, dtype=np.float64).ravel()
    return {
        "n_transitions": int(diag.get("n_transitions", 0)),
        "mean_dwell_period": float(diag.get("mean_dwell_period", float("nan"))),
        "thickness_final": float(diag.get("thickness_final", float("nan"))),
        "c_final_max_abs": float(np.max(np.abs(c))),
        "wall_time_s": wall,
    }, wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare slab_kinetic numpy vs jax.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Dotted overrides, e.g. physics.c1=0.40 time.T=20",
    )
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config if args.config.is_absolute() else _REPO / args.config)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        val: Any = raw
        try:
            val = float(raw)
        except ValueError:
            pass
        _set_dotted(cfg, key, val)

    np_metrics, np_wall = _run_backend(cfg, "numpy")
    jax_metrics, jax_wall = _run_backend(cfg, "jax")

    print("| metric | numpy | jax | abs/rel diff |")
    print("|--------|-------|-----|--------------|")
    for key in ("n_transitions", "mean_dwell_period", "thickness_final", "c_final_max_abs"):
        a = float(np_metrics[key]) if key != "n_transitions" else int(np_metrics[key])
        b = float(jax_metrics[key]) if key != "n_transitions" else int(jax_metrics[key])
        if key == "n_transitions":
            diff = f"{abs(b - a)}"
        elif key == "c_final_max_abs":
            diff = f"{abs(b - a):.3e}"
        else:
            den = max(abs(a), 1e-30)
            diff = f"{abs(b - a):.3e} ({abs(b - a) / den * 100:.3f}%)"
        print(f"| {key} | {a} | {b} | {diff} |")
    print(f"| wall_time_s | {np_wall:.3f} | {jax_wall:.3f} | — |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
