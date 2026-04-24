"""Nested YAML loading (Pydantic v2), result paths, and artifact writers.

Canonical on-disk layout under ``results/`` per ``docs/ARCHITECTURE.md`` §3.8
and §5. No flat legacy config ingestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from continuous_patterns.core.plotting import plot_fields_final

_MIGRATION_HINT = (
    "Flat or non-nested configs are not supported. Convert archived YAML to nested "
    "templates (see docs/ARCHITECTURE.md §10, migration)."
)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    model: str


class RunConfigValidated(BaseModel):
    """Top-level nested run card (extra sections allowed)."""

    model_config = ConfigDict(extra="allow")

    experiment: ExperimentSpec
    physics: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, Any] = Field(default_factory=dict)
    integration: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResultPaths:
    """Standard artifact locations under a single timestamped run root."""

    root: Path
    summary_json: Path
    config_yaml: Path
    final_state_npz: Path


def load_run_config(path: Path) -> dict[str, Any]:
    """Load and validate **nested** YAML; reject flat legacy layouts.

    A valid file must contain a mapping ``experiment:`` with nested keys
    (at least ``model``). Top-level-only keys such as ``grid: 64`` without an
    ``experiment`` block are rejected.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if raw is None:
        raise ValueError("config YAML is empty")
    if not isinstance(raw, dict):
        raise ValueError(
            f"config root must be a mapping, got {type(raw).__name__}. {_MIGRATION_HINT}"
        )
    if "experiment" not in raw:
        raise ValueError(
            f"nested YAML required: missing top-level `experiment` block. {_MIGRATION_HINT}"
        )
    exp = raw["experiment"]
    if not isinstance(exp, dict):
        raise ValueError(
            f"`experiment` must be a nested mapping, got {type(exp).__name__}. {_MIGRATION_HINT}"
        )
    cfg = RunConfigValidated.model_validate(raw)
    return cfg.model_dump(mode="python")


def save_run_config(path: Path, cfg: dict[str, Any]) -> None:
    """Validate ``cfg`` and write nested YAML (same schema as :func:`load_run_config`)."""
    validated = RunConfigValidated.model_validate(cfg)
    dumped = validated.model_dump(mode="python")
    text = yaml.safe_dump(dumped, sort_keys=False, default_flow_style=False)
    Path(path).write_text(text, encoding="utf-8")


def allocate_run_dir(*, experiment_name: str, results_root: Path) -> ResultPaths:
    """Create ``results_root / experiment_name / <UTC timestamp>/`` and return paths."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(results_root) / experiment_name / stamp
    root.mkdir(parents=True, exist_ok=False)
    return ResultPaths(
        root=root,
        summary_json=root / "summary.json",
        config_yaml=root / "config.yaml",
        final_state_npz=root / "final_state.npz",
    )


def save_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write ``summary.json`` (UTF-8, indented)."""
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def save_final_state_npz(
    path: Path,
    *,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    c: np.ndarray,
    chi: np.ndarray | None = None,
) -> None:
    """Write compressed ``final_state.npz`` with NumPy arrays."""
    data: dict[str, np.ndarray] = {
        "phi_m": np.asarray(phi_m),
        "phi_c": np.asarray(phi_c),
        "c": np.asarray(c),
    }
    if chi is not None:
        data["chi"] = np.asarray(chi)
    np.savez_compressed(path, **data)


def write_figures_final(
    run_root: Path,
    *,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    c: np.ndarray,
    L: float,
    R: float,
    chi: np.ndarray | None = None,
) -> Path:
    """Default figure export into ``run_root / figures_final.png`` (uses :mod:`plotting`)."""
    return plot_fields_final(phi_m, phi_c, c, L=L, R=R, path=run_root, chi=chi)
