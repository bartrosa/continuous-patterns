"""Run a generic axes-based Cartesian sweep."""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: dict[str, Any] = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _slug_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isfinite(v):
            s = f"{v:.6g}".replace("-", "m").replace(".", "p")
            return s
        return str(v).lower()
    return str(v).replace(" ", "_")


def _combo_name(combo: dict[str, Any]) -> str:
    parts = [f"{k}_{_slug_value(v)}" for k, v in combo.items()]
    return "__".join(parts)


def _front_origin_from_h5_or_final(
    run_dir: Path, *, rim_left_width_px: int, rim_width_px: int
) -> str:
    h5_path = run_dir / "snapshots.h5"
    phi_total: Any = None
    if h5_path.is_file():
        import h5py

        with h5py.File(h5_path, "r") as h5:
            keys = sorted([k for k in h5.keys() if k.startswith("snap_")])
            if keys:
                k_mid = keys[len(keys) // 2]
                g = h5[k_mid]
                phi_m = g["phi_m"][()]
                phi_c = g["phi_c"][()]
                phi_total = phi_m + phi_c
    if phi_total is None:
        arr = np.load(run_dir / "final_state.npz")
        phi_total = arr["phi_m"] + arr["phi_c"]

    n = int(phi_total.shape[0])
    left_hi = min(n, int(rim_left_width_px) + 10)
    right_lo = max(0, n - int(rim_width_px) - 10)
    left_mass = float(np.mean(phi_total[:left_hi, :]))
    right_mass = float(np.mean(phi_total[right_lo:, :]))
    if left_mass > right_mass + 0.1:
        return "LEFT"
    if right_mass > left_mass + 0.1:
        return "RIGHT"
    return "BOTH"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            "Usage: uv run python scripts/run_sweep_axes.py "
            "<sweep_yaml> <out_dir> [--chunk-size N]",
            file=sys.stderr,
        )
        return 2

    sweep_path = Path(args[0])
    out_root = Path(args[1])
    chunk_size = 2000
    if len(args) >= 4 and args[2] == "--chunk-size":
        chunk_size = int(args[3])

    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep config must be a mapping")
    sweep = raw.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("missing sweep section")
    axes = sweep.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("expected non-empty list in sweep.axes")

    base_cfg = load_run_config(Path(sweep["base_config"]))
    name = str(sweep.get("name", "sweep_axes"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root.mkdir(parents=True, exist_ok=True)

    overrides = raw.get("overrides", {})
    if isinstance(overrides, dict):
        for dotted_key, v in overrides.items():
            _set_dotted(base_cfg, dotted_key, copy.deepcopy(v))
    axis_defs: list[dict[str, Any]] = []
    for ax in axes:
        if not isinstance(ax, dict):
            raise ValueError("each axis must be a mapping")
        name_ax = str(ax.get("name", "")).strip()
        vals = list(ax.get("values", []))
        if not name_ax or not vals:
            raise ValueError("axis requires non-empty name and values")
        axis_defs.append(ax)

    entries: list[dict[str, Any]] = []
    axis_names = [str(ax["name"]) for ax in axis_defs]
    combos = [
        dict(zip(axis_names, vals, strict=True))
        for vals in itertools.product(*[ax["values"] for ax in axis_defs])
    ]
    for combo in tqdm(combos, total=len(combos), desc=f"Sweep {name}", unit="run", leave=True):
        cfg = copy.deepcopy(base_cfg)
        run_label = _combo_name(combo)
        cfg["experiment"]["name"] = run_label
        for ax in axis_defs:
            ax_name = str(ax["name"])
            ax_value = combo[ax_name]
            ax_path = ax.get("path")
            if isinstance(ax_path, str) and ax_path.strip():
                _set_dotted(cfg, ax_path, copy.deepcopy(ax_value))
            pov = ax.get("param_overrides")
            if isinstance(pov, dict):
                override_for_value = pov.get(ax_value)
                if override_for_value is None:
                    override_for_value = pov.get(str(ax_value))
                if override_for_value is not None:
                    if not isinstance(override_for_value, dict):
                        raise ValueError(
                            f"param_overrides for axis {ax_name!r} value "
                            f"{ax_value!r} must be a mapping"
                        )
                    for dotted_key, v in override_for_value.items():
                        _set_dotted(cfg, dotted_key, copy.deepcopy(v))

        status = "success"
        rel_path: str | None = None
        error: str | None = None
        band_count: int | None = None
        residual_pct: float | None = None
        wall_time_s: float | None = None
        front_origin: str | None = None
        try:
            t0 = time.perf_counter()
            result = run_one(
                cfg,
                results_root=out_root,
                chunk_size=chunk_size,
                write_artifacts=True,
                show_progress=False,
                log_level="WARNING",
            )
            assert result.paths is not None
            rel_path = result.paths.root.relative_to(out_root).as_posix()
            wall_time_s = float(time.perf_counter() - t0)
            slab_ax = result.diagnostics.get("slab_axial", {})
            dmb = result.diagnostics.get("dirichlet_mass_balance", {})
            band_count = (
                int(slab_ax.get("band_count_axial", 0)) if isinstance(slab_ax, dict) else None
            )
            residual_pct = float(dmb.get("residual_pct", 0.0)) if isinstance(dmb, dict) else None
            run_dir = out_root / rel_path
            rim_left = int(cfg.get("geometry", {}).get("rim_left_width_px", 0))
            rim_right = int(cfg.get("geometry", {}).get("rim_width_px", 0))
            front_origin = _front_origin_from_h5_or_final(
                run_dir,
                rim_left_width_px=rim_left,
                rim_width_px=rim_right,
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)

        entries.append(
            {
                "run_label": run_label,
                "parameters": combo,
                "status": status,
                "relative_path": rel_path,
                "band_count_axial": band_count,
                "residual_pct": residual_pct,
                "front_origin": front_origin,
                "wall_time_s": wall_time_s,
                "error": error,
            }
        )

    manifest = {
        "sweep_name": name,
        "timestamp": stamp,
        "source_sweep_yaml": str(sweep_path),
        "results_root": str(out_root.resolve()),
        "entries": entries,
    }
    (out_root / "manifest_axes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sweep complete: {out_root}")
    failed = [e for e in entries if e["status"] != "success"]
    if failed:
        print(f"WARNING: {len(failed)} runs failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
