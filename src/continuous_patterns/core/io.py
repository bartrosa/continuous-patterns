"""Layered nested YAML loading (Pydantic v2), result paths, and artifact writers.

Merge order for ``load_run_config`` is documented in ``docs/ARCHITECTURE.md`` §10.
Canonical on-disk layout under ``results/`` per §3.8 and §5. No flat legacy
config ingestion. Run cards are validated with typed nested models
(``ExperimentSpec``, ``GeometrySpec``, …); ``physics`` and ``initial`` remain
plain dicts until model-specific schemas exist (§2.8).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from continuous_patterns.core.plotting import plot_fields_final

_MIGRATION_HINT = (
    "Flat or non-nested configs are not supported. Convert archived YAML to nested "
    "experiment cards under ``experiments/canonical/`` (see docs/ARCHITECTURE.md §10)."
)


class ExperimentSpec(BaseModel):
    """Experiment identity and RNG seed."""

    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    model: Literal["agate_ch", "agate_stage2"]
    seed: int = 42
    description: str | None = None


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
    record_evolution_gif: bool = False
    save_snapshots_h5: bool = False
    save_jablczynski_plot: bool = False
    gif_max_frames: int = Field(default=120, ge=8, le=500)
    gif_fps: int = Field(default=10, ge=1, le=60)
    include_params_panel: bool = True
    log_level: str = "INFO"


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
    log_file: Path
    snapshots_h5: Path
    jablczynski_plot: Path


_SOLVER_SETTINGS_DEFAULT = Path("experiments/solver_settings.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``copy(base)``. Override wins on conflicts.

    Dict values recurse; scalars and lists are replaced wholesale by ``override``.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_library_defaults() -> dict[str, Any]:
    """Load bundled ``solver_defaults.yaml`` from package data."""
    path = resources.files("continuous_patterns.defaults").joinpath("solver_defaults.yaml")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _load_user_solver_settings(path: Path | None = None) -> dict[str, Any]:
    """Load optional per-machine overrides from ``experiments/solver_settings.yaml``."""
    p = path or _SOLVER_SETTINGS_DEFAULT
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_run_config(
    experiment_path: Path | str,
    *,
    user_settings_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate a nested run YAML with layered defaults.

    Merge order (later overrides earlier):

    1. Library defaults (``continuous_patterns.defaults.solver_defaults.yaml``).
    2. User overrides: ``experiments/solver_settings.yaml`` when present, or
       ``user_settings_path`` when passed explicitly.
    3. Experiment YAML at ``experiment_path``.

    Parameters
    ----------
    experiment_path
        Path to the experiment ``.yaml`` on disk.
    user_settings_path
        Optional explicit path to user solver settings; default is
        ``experiments/solver_settings.yaml`` relative to the process CWD.

    Returns
    -------
    dict
        Nested configuration (Python primitives and nested dicts).

    Raises
    ------
    ValueError
        If the experiment file is empty, not a mapping, or missing a valid ``experiment`` block.
    FileNotFoundError
        If ``experiment_path`` does not exist.
    pydantic.ValidationError
        If the merged document fails schema validation.
    """
    experiment_path = Path(experiment_path)
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    defaults = _load_library_defaults()
    user = _load_user_solver_settings(
        Path(user_settings_path) if user_settings_path is not None else None
    )
    raw_exp = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    if raw_exp is None:
        raise ValueError("config YAML is empty")
    if not isinstance(raw_exp, dict):
        raise ValueError(
            f"config root must be a mapping, got {type(raw_exp).__name__}. {_MIGRATION_HINT}"
        )

    merged = _deep_merge(defaults, user)
    merged = _deep_merge(merged, raw_exp)

    # Option D (spectral mass drift) is always recorded when supported by the model driver.
    merged.setdefault("output", {})
    merged["output"]["record_spectral_mass_diagnostic"] = True

    if "experiment" not in merged:
        raise ValueError(
            f"nested YAML required: missing top-level `experiment` block. {_MIGRATION_HINT}"
        )
    exp = merged["experiment"]
    if not isinstance(exp, dict):
        raise ValueError(
            f"`experiment` must be a nested mapping, got {type(exp).__name__}. {_MIGRATION_HINT}"
        )
    cfg = RunConfigValidated.model_validate(merged)
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
        log_file=root / "run.log",
        snapshots_h5=root / "snapshots.h5",
        jablczynski_plot=root / "jablczynski.png",
    )


def save_snapshots_h5(
    path: Path | str,
    snapshots: list[dict[str, Any]],
    *,
    dt: float,
    cfg_summary: dict[str, Any] | None = None,
) -> None:
    """Write snapshot list to HDF5 (gzip fields; metadata in ``/meta``)."""
    import h5py

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["dt"] = float(dt)
        meta.attrs["n_snapshots"] = int(len(snapshots))
        if snapshots:
            shp = list(np.asarray(snapshots[0]["phi_m"]).shape)
            meta.attrs["field_shape"] = shp
        if cfg_summary is not None:
            meta.attrs["config_json"] = json.dumps(cfg_summary, default=str)
        for idx, snap in enumerate(snapshots):
            g = h5.create_group(f"snap_{idx:05d}")
            g.attrs["step"] = int(snap["step"])
            g.attrs["t"] = float(snap["t"])
            for field in ("phi_m", "phi_c", "c"):
                arr = np.asarray(snap[field], dtype=np.float32)
                g.create_dataset(
                    field,
                    data=arr,
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                )


def load_snapshots_h5(path: Path | str) -> list[dict[str, Any]]:
    """Load ``snapshots.h5`` written by :func:`save_snapshots_h5`."""
    import h5py

    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    with h5py.File(p, "r") as h5:
        if "meta" not in h5:
            return []
        n = int(h5["meta"].attrs.get("n_snapshots", 0))
        for idx in range(n):
            g = h5[f"snap_{idx:05d}"]
            out.append(
                {
                    "step": int(g.attrs["step"]),
                    "t": float(g.attrs["t"]),
                    "phi_m": np.asarray(g["phi_m"]),
                    "phi_c": np.asarray(g["phi_c"]),
                    "c": np.asarray(g["c"]),
                }
            )
    return out


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
    title: str | None = None,
    params: dict[str, Any] | None = None,
    include_params_panel: bool = True,
) -> Path:
    """Default figure export into ``run_root / figures_final.png`` (uses :mod:`plotting`)."""
    return plot_fields_final(
        phi_m,
        phi_c,
        c,
        L=L,
        R=R,
        path=run_root,
        title=title,
        params=params,
        include_params_panel=include_params_panel,
    )
