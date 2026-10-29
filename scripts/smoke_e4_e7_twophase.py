#!/usr/bin/env python3
"""Quick smoke test for E4/E5/E7 rows before a full matrix run (~minutes, not hours)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from continuous_patterns.core.banding_metrics import (  # noqa: E402
    compute_liesegang_laws,
    record_from_slab_twophase_result,
)
from continuous_patterns.models.slab_twophase import (  # noqa: E402
    estimate_integration_steps,
    simulate,
)


def _build(entry: dict, *, T_scale: float) -> dict:
    from scripts.run_banding_comparison import _build_run_config

    cfg = _build_run_config(entry, _REPO)
    cfg.setdefault("time", {})
    cfg["time"]["T"] = float(cfg["time"]["T"]) * float(T_scale)
    cfg.setdefault("output", {})
    cfg["output"]["save_evolution"] = False
    return cfg


def _check(entry: dict, res, *, T_scale: float) -> list[str]:
    from scripts.run_banding_comparison import _n_bands_from_record, _row_integrity_status

    label = str(entry["run_label"])
    rec = record_from_slab_twophase_result(res, label)
    nb = _n_bands_from_record(rec)
    laws = compute_liesegang_laws(rec)
    d_sp = laws.get("d_spacings")
    mean_sp = (
        float(np.mean(np.asarray(d_sp, dtype=np.float64)))
        if isinstance(d_sp, list) and d_sp
        else None
    )
    status, reasons = _row_integrity_status(
        "slab_twophase",
        res.config_resolved if isinstance(res.config_resolved, dict) else {},
        dict(res.diagnostics),
        run_label=label,
        n_bands=nb,
        mean_spacing=mean_sp,
    )
    diag = res.diagnostics
    n_proj = int(diag.get("n_projection_events", 0))
    n_steps = int(diag.get("n_steps", 1))
    lines = [
        f"{label}: status={status} n_bands={nb} n_proj={n_proj}/{n_steps} "
        f"({100.0 * n_proj / max(n_steps, 1):.2f}%) wall={diag.get('wall_time_s', 0):.1f}s "
        f"(T×{T_scale})",
    ]
    if reasons:
        lines.append(f"  reasons: {reasons}")
    if status not in ("OK", "WARN"):
        lines.append("  FAIL")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test TP rows from the E1–E7 matrix.")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=_REPO / "experiments/comparisons/e1_e8_matrix_v6.yaml",
    )
    parser.add_argument(
        "--T-scale",
        type=float,
        default=0.01,
        help="Scale factor for time.T (default 0.01 → E7 ~60 domain time).",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else _REPO / args.config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    targets = {
        "E4_TP_gamma4_open",
        "E5_TP_gamma0_open",
        "E6_TP_gamma4_closed",
        "E7_TP_gamma4_CH_baselineOstwald",
        "E8_TP_gamma4_CH_matchedOstwald",
    }
    entries = [e for e in raw["comparison"]["experiments"] if e["run_label"] in targets]

    failed = False
    for entry in entries:
        cfg = _build(entry, T_scale=args.T_scale)
        dt, n_steps = estimate_integration_steps(cfg)
        print(
            f"\n[{entry['run_label']}] dt={dt:.4e} n_steps={n_steps} T={cfg['time']['T']}",
            flush=True,
        )
        t0 = time.perf_counter()
        res = simulate(cfg, show_progress=True)
        print(f"  simulated in {time.perf_counter() - t0:.1f}s", flush=True)
        for line in _check(entry, res, T_scale=args.T_scale):
            print(line, flush=True)
            if line.endswith("FAIL"):
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
