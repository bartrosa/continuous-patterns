#!/usr/bin/env python3
"""EXP-D6 PART C: Heaney moving-front band-width diagnostic sweep."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_sweep_axes import _prune_slab_twophase_physics, _set_dotted  # noqa: E402

from continuous_patterns.core.io import load_run_config, validate_run_config_dict  # noqa: E402
from continuous_patterns.experiments.run import run_one  # noqa: E402

_MINI_OVERRIDES = {
    "geometry.n": 128,
    "time.T": 800.0,
    "output.evolution_dt": 25.0,
}


def _load_sweep(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping")
    return raw


def _build_rows(doc: dict[str, Any], *, mini: bool) -> list[dict[str, Any]]:
    sweep = doc["sweep"]
    base = load_run_config(_ROOT / sweep["base_config"])
    global_ov = doc.get("overrides") or {}
    rows: list[dict[str, Any]] = []
    for spec in sweep["runs"]:
        cfg = copy.deepcopy(base)
        for dotted, val in global_ov.items():
            _set_dotted(cfg, str(dotted), copy.deepcopy(val))
        for dotted, val in (spec.get("overrides") or {}).items():
            _set_dotted(cfg, str(dotted), copy.deepcopy(val))
        if mini:
            for dotted, val in _MINI_OVERRIDES.items():
                _set_dotted(cfg, str(dotted), copy.deepcopy(val))
        label = str(spec["run_label"])
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = label
        cfg["experiment"]["model"] = "slab_twophase"
        _prune_slab_twophase_physics(cfg)
        rows.append({"run_label": label, "cfg": validate_run_config_dict(cfg)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D6H Heaney moving-front sweep")
    parser.add_argument(
        "--sweep-yaml",
        type=Path,
        default=_ROOT / "experiments/sweeps/d6h_heaney_movingfront.yaml",
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args(argv)

    doc = _load_sweep(args.sweep_yaml.resolve())
    rows = _build_rows(doc, mini=args.mini)
    sweep_dir = args.results_root.resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="D6H sweep", unit="run"):
        label = row["run_label"]
        cfg = row["cfg"]
        phys = cfg["physics"]
        t0 = time.perf_counter()
        result = run_one(
            cfg,
            results_root=sweep_dir,
            chunk_size=args.chunk_size,
            write_artifacts=True,
            show_progress=True,
            log_level="WARNING",
        )
        assert result.paths is not None
        diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        entries.append(
            {
                "run_label": label,
                "status": "success",
                "relative_path": result.paths.root.relative_to(sweep_dir).as_posix(),
                "wall_time_s": float(time.perf_counter() - t0),
                "mass_residual_pct": diag.get("mass_residual_pct"),
                "clip_rate": diag.get("clip_rate"),
                "c1": float(phys["c1"]),
                "c2": float(phys["c2"]),
                "band_width": float(phys["c1"]) - float(phys["c2"]),
                "k_L_scale": float(phys.get("k_L_scale", 1.0)),
            }
        )

    manifest = {
        "sweep_name": doc["sweep"]["name"],
        "sweep_dir": str(sweep_dir.resolve()),
        "mini_profile": bool(args.mini),
        "entries": entries,
    }
    (sweep_dir / "manifest_d6h.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"D6H sweep complete: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
