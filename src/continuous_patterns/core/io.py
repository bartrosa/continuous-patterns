"""Nested YAML loading (Pydantic v2), result paths, and artifact writers.

Canonical on-disk layout under ``results/`` per ``docs/ARCHITECTURE.md`` §3.8
and §5. No flat legacy config ingestion. Run cards are validated with typed
nested models (``ExperimentSpec``, ``GeometrySpec``, …); ``physics`` and
``initial`` remain plain dicts until model-specific schemas exist (§2.8).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from continuous_patterns.core.plotting import plot_fields_final

_MIGRATION_HINT = (
    "Flat or non-nested configs are not supported. Convert archived YAML to nested "
    "templates (see docs/ARCHITECTURE.md §10, migration)."
)


class ExperimentSpec(BaseModel):
    """Experiment identity and RNG seed."""

    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    model: Literal["agate_ch", "agate_stage2"]
    seed: int = 42


class GeometrySpec(BaseModel):
    """Domain and cavity grid (Stage II bulk may set ``R: 0``)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["circular_cavity"] = "circular_cavity"
    L: float = Field(gt=0)
    R: float = Field(ge=0)
    n: int = Field(gt=0)
    eps_scale: float = Field(default=2.0, gt=0)


class StressSpec(BaseModel):
    """Prescribed Cauchy stress mode and ψ-coupling strength."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "none",
        "uniform_uniaxial",
        "uniform_biaxial",
        "pure_shear",
        "flamant_two_point",
        "pressure_gradient",
        "kirsch",
    ] = "none"
    sigma_0: float = 0.0
    stress_coupling_B: float = 0.0
    stress_eps_factor: float = Field(default=3.0, gt=0)


class TimeSpec(BaseModel):
    """Horizon and snapshot cadence."""

    model_config = ConfigDict(extra="forbid")

    dt: float = Field(gt=0)
    T: float = Field(gt=0)
    snapshot_every: int = Field(default=500, ge=1)


class OutputSpec(BaseModel):
    """Output toggles (extensible for future diagnostics keys)."""

    model_config = ConfigDict(extra="allow")

    save_final_state: bool = True
    flux_sample_dt: float | None = None
    record_spectral_mass_diagnostic: bool = False


class RunConfigValidated(BaseModel):
    """Validated nested run card (strict top-level; permissive ``physics`` / ``initial``)."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentSpec
    geometry: GeometrySpec
    physics: dict[str, Any] = Field(default_factory=dict)
    stress: StressSpec = Field(default_factory=StressSpec)
    time: TimeSpec
    output: OutputSpec = Field(default_factory=OutputSpec)
    initial: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResultPaths:
    """Standard artifact locations under a single timestamped run root."""

    root: Path
    summary_json: Path
    config_yaml: Path
    final_state_npz: Path


def load_run_config(path: Path) -> dict[str, Any]:
    """Load and validate nested run YAML; reject flat or unknown top-level keys.

    Parameters
    ----------
    path
        Path to ``.yaml`` on disk.

    Returns
    -------
    dict
        Nested configuration (Python primitives and nested dicts).

    Raises
    ------
    ValueError
        If the file is empty, not a mapping, or missing a valid ``experiment`` block.
    FileNotFoundError
        If ``path`` does not exist (from ``Path.read_text``).
    pydantic.ValidationError
        If the document fails schema validation.
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


def _json_numpy_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with ``None``; recurse dicts/lists (JSON-safe ``summary.json``)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float | np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.generic):
        return _sanitize_for_json(obj.item())
    return obj


def dumps_json(payload: Any) -> str:
    """Serialize ``payload`` to a JSON string (NumPy-aware; NaN/Inf → null)."""
    clean = _sanitize_for_json(payload)
    return json.dumps(clean, indent=2, allow_nan=False, default=_json_numpy_default)


def save_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write ``summary.json`` (UTF-8, indented). Supports NumPy; NaN/Inf → null."""
    Path(path).write_text(dumps_json(payload), encoding="utf-8")


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
