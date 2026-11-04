#!/usr/bin/env python3
"""EXP-D1 discrimination sweep runner (--mini, --dry-run, full campaign)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import UTC, datetime
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
from continuous_patterns.models.slab_twophase import estimate_integration_steps  # noqa: E402

_DX_FIXED = 0.390625
_MINI_RUNS = ("D1A_v0p020_f30", "D1B_base", "D1C_g4")
_TIER2_HOURS = 12.0
_ANCHOR_LABEL = "D1A_v0p020_f30"


def _load_sweep_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping at root")
    return raw


def _block_id(run_label: str) -> str:
    if run_label.startswith("D1A_"):
        return "A"
    if run_label.startswith("D1B_"):
        return "B"
    if run_label.startswith("D1C_"):
        return "C"
    raise ValueError(f"unknown block for run_label={run_label!r}")


def _apply_block_overrides(
    cfg: dict[str, Any], block_overrides: dict[str, Any], block: str
) -> None:
    blk = block_overrides.get(block)
    if isinstance(blk, dict):
        for dotted, value in blk.items():
            _set_dotted(cfg, str(dotted), copy.deepcopy(value))


def _mini_overrides(run_label: str, cfg: dict[str, Any]) -> None:
    block = _block_id(run_label)
    cfg.setdefault("geometry", {})
    cfg["geometry"]["n"] = 128
    cfg["geometry"]["L"] = float(128 * _DX_FIXED)
    t = cfg.setdefault("time", {})
    if block == "A":
        v = float(cfg.get("physics", {}).get("v_front", 0.02))
        L = float(cfg["geometry"]["L"])
        t["T"] = float(1.1 * L / v)
    elif block == "B":
        t["T"] = 600.0
    else:
        t["T"] = 4000.0


def _build_run_table(
    sweep_doc: dict[str, Any],
    *,
    mini: bool,
    include_tier2: bool,
) -> list[dict[str, Any]]:
    sweep = sweep_doc["sweep"]
    base_path = _ROOT / sweep["base_config"]
    base_cfg = load_run_config(base_path)
    global_overrides = sweep_doc.get("overrides") or {}
    block_overrides = sweep_doc.get("block_overrides") or {}
    rows: list[dict[str, Any]] = []
    for spec in sweep["runs"]:
        run_label = str(spec["run_label"])
        if mini and run_label not in _MINI_RUNS:
            continue
        tier = str(spec.get("tier", "primary"))
        if tier == "optional" and not include_tier2 and not mini:
            continue
        cfg = copy.deepcopy(base_cfg)
        for dotted, value in global_overrides.items():
            _set_dotted(cfg, str(dotted), copy.deepcopy(value))
        for dotted, value in (spec.get("overrides") or {}).items():
            _set_dotted(cfg, str(dotted), copy.deepcopy(value))
        _apply_block_overrides(cfg, block_overrides, _block_id(run_label))
        if mini:
            _mini_overrides(run_label, cfg)
        cfg.setdefault("experiment", {})
        cfg["experiment"]["name"] = run_label
        cfg["experiment"]["model"] = "slab_twophase"
        _prune_slab_twophase_physics(cfg)
        rows.append(
            {
                "run_label": run_label,
                "tier": tier,
                "block": _block_id(run_label),
                "cfg": validate_run_config_dict(cfg),
            }
        )
    return rows


def _estimate_wall_s(cfg: dict[str, Any], *, step_cost_s: float) -> tuple[int, float]:
    _dt, n_steps = estimate_integration_steps(cfg)
    return n_steps, float(n_steps * step_cost_s)


def _print_dry_run_table(
    rows: list[dict[str, Any]],
    *,
    step_cost_s: float,
    step_cost_source: str,
) -> dict[str, Any]:
    print(f"# D1 dry-run estimate (step_cost={step_cost_s:.4e} s/step from {step_cost_source})")
    print("| run_label | block | tier | n | T | n_steps | est_wall_h |")
    print("|---|---|---|---:|---:|---:|---:|")
    total_h = 0.0
    tier2_flags: list[str] = []
    for row in rows:
        cfg = row["cfg"]
        n = int(cfg["geometry"]["n"])
        T = float(cfg["time"]["T"])
        n_steps, wall_s = _estimate_wall_s(cfg, step_cost_s=step_cost_s)
        wall_h = wall_s / 3600.0
        total_h += wall_h
        label = row["run_label"]
        if label == "D1A_v0p005_f30" and wall_h > _TIER2_HOURS:
            tier2_flags.append(label)
        print(
            f"| {label} | {row['block']} | {row['tier']} | {n} | {T:.4g} | {n_steps} | {wall_h:.2f} |"  # noqa: E501
        )
    print(f"\nTotal estimated wall time: {total_h:.2f} h ({len(rows)} runs)")
    if tier2_flags:
        print(
            f"\nTIER-2 WARNING: {', '.join(tier2_flags)} exceeds {_TIER2_HOURS:.0f} h — "
            "demote to optional tier unless explicitly included with --include-tier2"
        )
    return {"total_wall_h": total_h, "n_runs": len(rows), "tier2_over_12h": tier2_flags}


def _run_rows(
    rows: list[dict[str, Any]],
    *,
    sweep_dir: Path,
    chunk_size: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="D1 sweep", unit="run"):
        run_label = row["run_label"]
        cfg = row["cfg"]
        t0 = time.perf_counter()
        try:
            result = run_one(
                cfg,
                results_root=sweep_dir,
                chunk_size=chunk_size,
                write_artifacts=True,
                show_progress=True,
                log_level="WARNING",
            )
            assert result.paths is not None
            rel = result.paths.root.relative_to(sweep_dir).as_posix()
            wall_s = float(time.perf_counter() - t0)
            diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
            entries.append(
                {
                    "run_label": run_label,
                    "block": row["block"],
                    "tier": row["tier"],
                    "status": "success",
                    "relative_path": rel,
                    "wall_time_s": wall_s,
                    "mass_residual_pct": diag.get("mass_residual_pct"),
                    "clipped_mass_frac": diag.get("clipped_mass_frac"),
                    "phi_ledger_residual": diag.get("phi_ledger_residual"),
                    "n_steps": diag.get("n_steps"),
                    "dt": diag.get("dt"),
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "run_label": run_label,
                    "block": row["block"],
                    "tier": row["tier"],
                    "status": "failed",
                    "relative_path": None,
                    "wall_time_s": float(time.perf_counter() - t0),
                    "error": str(exc),
                }
            )
            raise
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D1 discrimination sweep")
    parser.add_argument(
        "--sweep-yaml",
        type=Path,
        default=_ROOT / "experiments/sweeps/d1_discrimination.yaml",
    )
    parser.add_argument(
        "--results-root", type=Path, default=_ROOT / "results/slab_twophase/d1_discrimination"
    )
    parser.add_argument(
        "--mini", action="store_true", help="n=128, shortened T, 3 representative runs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print run table with wall-time estimates"
    )
    parser.add_argument(
        "--include-tier2", action="store_true", help="include optional tier-2 runs (e.g. v=0.005)"
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="RUN_LABEL", help="run only these labels (e.g. D1B_base)"
    )
    parser.add_argument(
        "--into-sweep-dir",
        type=Path,
        help="write into an existing sweep directory and merge manifest_d1.json",
    )
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args(argv)

    sweep_doc = _load_sweep_yaml(args.sweep_yaml.resolve())
    rows = _build_run_table(sweep_doc, mini=args.mini, include_tier2=args.include_tier2)
    if args.labels:
        wanted = set(args.labels)
        rows = [r for r in rows if r["run_label"] in wanted]
        if not rows:
            raise RuntimeError(f"--labels: no matching runs in {sorted(wanted)}")

    if args.dry_run:
        timing_path = args.results_root / "mini_profile.json"
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            step_cost = float(timing["step_cost_s"])
            source = str(timing.get("source", timing_path))
        elif args.mini:
            print("# --dry-run with --mini: measuring step cost inline…")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            mini_dir = args.results_root / f"mini_{stamp}"
            mini_dir.mkdir(parents=True, exist_ok=True)
            entries = _run_rows(rows, sweep_dir=mini_dir, chunk_size=args.chunk_size)
            total_steps = sum(int(e.get("n_steps") or 0) for e in entries)
            total_wall = sum(float(e.get("wall_time_s") or 0.0) for e in entries)
            if total_steps <= 0:
                raise RuntimeError("mini profile produced zero steps")
            step_cost = total_wall / total_steps
            payload = {
                "step_cost_s": step_cost,
                "total_steps": total_steps,
                "total_wall_s": total_wall,
                "runs": [e["run_label"] for e in entries],
                "source": str(mini_dir),
            }
            args.results_root.mkdir(parents=True, exist_ok=True)
            timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            source = str(timing_path)
        else:
            raise FileNotFoundError(
                f"mini_profile.json not found at {timing_path}; run with --mini first "
                "(or combine --dry-run --mini to measure inline)"
            )
        est = _print_dry_run_table(rows, step_cost_s=step_cost, step_cost_source=source)
        est_path = args.results_root / "dry_run_estimate.json"
        args.results_root.mkdir(parents=True, exist_ok=True)
        est_path.write_text(json.dumps(est, indent=2), encoding="utf-8")
        print(f"\nWrote {est_path}")
        return 0

    if args.into_sweep_dir:
        sweep_dir = args.into_sweep_dir.resolve()
        if not sweep_dir.is_dir():
            raise FileNotFoundError(f"--into-sweep-dir not found: {sweep_dir}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        sweep_dir = args.results_root / stamp
        sweep_dir.mkdir(parents=True, exist_ok=False)
    entries = _run_rows(rows, sweep_dir=sweep_dir, chunk_size=args.chunk_size)

    if args.mini:
        total_steps = sum(int(e.get("n_steps") or 0) for e in entries)
        total_wall = sum(float(e.get("wall_time_s") or 0.0) for e in entries)
        profile = {
            "step_cost_s": total_wall / max(total_steps, 1),
            "total_steps": total_steps,
            "total_wall_s": total_wall,
            "runs": [e["run_label"] for e in entries],
            "source": str(sweep_dir),
        }
        args.results_root.mkdir(parents=True, exist_ok=True)
        (args.results_root / "mini_profile.json").write_text(
            json.dumps(profile, indent=2), encoding="utf-8"
        )

    manifest_path = sweep_dir / "manifest_d1.json"
    if args.into_sweep_dir and manifest_path.is_file():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        prev_entries = prev.get("entries", [])
        by_label = {str(e["run_label"]): e for e in prev_entries}
        for e in entries:
            by_label[str(e["run_label"])] = e
        entries = list(by_label.values())
        entries.sort(key=lambda e: str(e.get("run_label", "")))

    manifest = {
        "sweep_name": sweep_doc["sweep"]["name"],
        "sweep_dir": str(sweep_dir.resolve()),
        "mini": bool(args.mini),
        "include_tier2": bool(args.include_tier2),
        "entries": entries,
    }
    (sweep_dir / "manifest_d1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"D1 sweep complete: {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
