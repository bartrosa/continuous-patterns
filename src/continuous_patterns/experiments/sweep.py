"""Parameter sweep runner: sweep YAML → Cartesian grid → :func:`run_one`.

``python -m continuous_patterns.experiments.sweep --sweep …`` (``docs/ARCHITECTURE.md`` §6.2).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from continuous_patterns.core.io import dumps_json, load_run_config
from continuous_patterns.experiments.run import run_one

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_base_config_path(raw: Path) -> Path:
    """Resolve ``base_config`` relative to CWD first, then repo root."""
    if raw.is_absolute():
        return raw
    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.is_file():
        return cwd_candidate
    repo_candidate = _REPO_ROOT / raw
    if repo_candidate.is_file():
        return repo_candidate
    raise FileNotFoundError(
        f"base_config not found: {raw!s} (tried cwd: {cwd_candidate}, repo: {repo_candidate})"
    )


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg['a']['b']['c'] = value`` from dotted key ``a.b.c``."""
    keys = dotted.split(".")
    target: Any = cfg
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _extract_key_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Top-level scalar metrics for manifest overview (skip nested dicts/lists)."""
    out: dict[str, Any] = {}
    for k, v in diagnostics.items():
        if isinstance(v, dict | list | tuple | np.ndarray):
            continue
        if isinstance(v, bool | int | str):
            out[k] = v
        elif isinstance(v, float | np.floating):
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                out[k] = None
            else:
                out[k] = fv
        elif isinstance(v, np.generic):
            item: Any = v.item()
            if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
                out[k] = None
            elif isinstance(item, bool | int | float | str):
                out[k] = item
    return out


def _write_sweep_report(path: Path, manifest: dict[str, Any]) -> None:
    """Write a simple Markdown table of sweep outcomes."""
    lines = [
        f"# Sweep: {manifest['sweep_name']}",
        f"- Timestamp: {manifest['timestamp']}",
        f"- Base config: {manifest['base_config']}",
        f"- Total runs: {len(manifest['runs'])}",
        "",
        "## Runs",
        "",
        "| run_id | status | parameters | key metrics |",
        "|--------|--------|------------|-------------|",
    ]
    for entry in manifest["runs"]:
        params = ", ".join(f"{k}={v}" for k, v in entry["parameters"].items())
        params = params.replace("|", "\\|")
        metrics = ", ".join(f"{k}={v}" for k, v in entry["key_metrics"].items())
        metrics = metrics.replace("|", "\\|")
        lines.append(f"| {entry['run_id']} | {entry['status']} | {params} | {metrics} |")
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class SweepResult:
    """Completed sweep: root directory, manifest dict, per-run entries."""

    sweep_root: Path
    manifest: dict[str, Any]
    entries: list[dict[str, Any]]


def run_sweep(
    sweep_cfg: dict[str, Any],
    *,
    results_root: Path,
    chunk_size: int = 2000,
) -> SweepResult:
    """Run Cartesian product of ``grid``; write ``manifest.json`` and ``report.md``."""
    sweep_meta = sweep_cfg["sweep"]
    sweep_name = str(sweep_meta["name"])
    base_path = _resolve_base_config_path(Path(sweep_meta["base_config"]))
    base_cfg = load_run_config(base_path)

    overrides = dict(sweep_cfg.get("overrides", {}))
    for dotted_key, value in overrides.items():
        _set_dotted(base_cfg, dotted_key, copy.deepcopy(value))

    grid = dict(sweep_cfg.get("grid", {}))
    if not grid:
        combinations: list[dict[str, Any]] = [{}]
    else:
        keys = list(grid.keys())
        values_lists = [grid[k] for k in keys]
        combinations = [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values_lists)
        ]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = Path(results_root) / "sweeps" / f"{sweep_name}_{stamp}"
    sweep_root.mkdir(parents=True, exist_ok=False)

    manifest_entries: list[dict[str, Any]] = []
    for idx, combo in enumerate(combinations):
        run_id = f"run_{idx:04d}"
        cfg_point = copy.deepcopy(base_cfg)
        for dotted_key, value in combo.items():
            _set_dotted(cfg_point, dotted_key, copy.deepcopy(value))
        cfg_point["experiment"]["name"] = f"{sweep_name}_{run_id}"

        try:
            result = run_one(
                cfg_point,
                results_root=sweep_root,
                chunk_size=chunk_size,
                write_artifacts=True,
            )
            status = "success"
            assert result.paths is not None
            rel_path = result.paths.root.relative_to(sweep_root).as_posix()
            diagnostics_summary = _extract_key_metrics(result.diagnostics)
        except Exception as e:
            status = "failed"
            rel_path = None
            diagnostics_summary = {"error": str(e)}

        manifest_entries.append(
            {
                "run_id": run_id,
                "status": status,
                "relative_path": rel_path,
                "parameters": combo,
                "key_metrics": diagnostics_summary,
            }
        )

    manifest: dict[str, Any] = {
        "sweep_name": sweep_name,
        "timestamp": stamp,
        "base_config": str(base_path.resolve()),
        "overrides": overrides,
        "grid": grid,
        "runs": manifest_entries,
    }
    (sweep_root / "manifest.json").write_text(dumps_json(manifest), encoding="utf-8")
    _write_sweep_report(sweep_root / "report.md", manifest)

    return SweepResult(sweep_root=sweep_root, manifest=manifest, entries=manifest_entries)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--sweep``, ``--out-dir``, ``--chunk-size``."""
    parser = argparse.ArgumentParser(
        prog="continuous_patterns.experiments.sweep",
        description="Run a parameter sweep from a sweep YAML.",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Path to sweep YAML.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Results root (default: ./results).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="IMEX chunk size passed to each run.",
    )
    args = parser.parse_args(argv)

    try:
        sweep_path = Path(args.sweep)
        sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
        if not isinstance(sweep_cfg, dict):
            raise ValueError("sweep YAML root must be a mapping")
    except Exception as e:
        print(f"Sweep config load failed: {e}", file=sys.stderr)
        return 2

    try:
        result = run_sweep(
            sweep_cfg,
            results_root=Path(args.out_dir),
            chunk_size=int(args.chunk_size),
        )
    except Exception as e:
        print(f"Sweep failed: {e}", file=sys.stderr)
        return 1

    print(f"Sweep complete: {result.sweep_root}")
    failed = sum(1 for e in result.entries if e["status"] == "failed")
    if failed > 0:
        print(f"WARNING: {failed}/{len(result.entries)} runs failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
