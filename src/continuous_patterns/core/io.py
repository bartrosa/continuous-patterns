"""Layered nested YAML loading (Pydantic v2), result paths, and artifact writers.

Merge order for ``load_run_config`` is documented in ``docs/ARCHITECTURE.md`` §10.
Canonical on-disk layout under ``results/`` per §3.8 and §5. No flat legacy
config ingestion. Run cards are validated with typed nested models
(``ExperimentSpec``, ``GeometrySpec``, …); ``physics`` and ``initial`` remain
plain dicts until model-specific schemas exist (§2.8).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from continuous_patterns.core.plotting import plot_fields_final

logger = logging.getLogger(__name__)

_LEGACY_MODEL_ALIASES: dict[str, str] = {
    "agate_ch": "cavity_reactive",
    "agate_stage2": "bulk_relaxation",
}


def resolve_model_name(name: str) -> str:
    """Map deprecated model keys to current names; log once per legacy key usage."""
    if name in _LEGACY_MODEL_ALIASES:
        new_name = _LEGACY_MODEL_ALIASES[name]
        logger.warning(
            "Model name %r is deprecated; use %r. "
            "Legacy alias will be removed in a future version.",
            name,
            new_name,
        )
        return new_name
    return name


_MIGRATION_HINT = (
    "Flat or non-nested configs are not supported. Convert archived YAML to nested "
    "experiment cards under ``experiments/canonical/`` (see docs/ARCHITECTURE.md §10)."
)


class ExperimentSpec(BaseModel):
    """Experiment identity and RNG seed."""

    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    model: Literal["cavity_reactive", "bulk_relaxation"]
    seed: int = 42
    description: str | None = None
    scenario: str | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _coerce_legacy_model_name(cls, v: object) -> object:
        if isinstance(v, str):
            return resolve_model_name(v)
        return v


class GeometrySpec(BaseModel):
    """Domain and cavity grid (Stage II bulk may set ``R: 0`` on ``circular_cavity``)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "circular_cavity",
        "elliptic_cavity",
        "polygon_cavity",
        "wedge_cavity",
        "rectangular_slot",
    ] = "circular_cavity"
    L: float = Field(gt=0)
    n: int = Field(gt=0)
    eps_scale: float = Field(default=2.0, gt=0)
    R: float | None = None
    a: float | None = None
    b: float | None = None
    theta: float | None = None
    width: float | None = None
    height: float | None = None
    n_sides: int | None = None
    vertices: list[list[float]] | None = None
    theta_offset: float | None = None
    R_inner: float | None = None
    R_outer: float | None = None
    opening_angle: float | None = None
    theta_center: float | None = None

    @model_validator(mode="after")
    def _geometry_required_fields(self) -> Self:
        t = self.type
        missing: list[str] = []

        def forbid(keys: frozenset[str], label: str) -> None:
            data = self.model_dump()
            bad = [k for k in keys if data.get(k) is not None]
            if bad:
                raise ValueError(f"geometry.type={label!r} must not set: {sorted(bad)}")

        if t == "circular_cavity":
            forbid(
                frozenset(
                    {
                        "a",
                        "b",
                        "theta",
                        "width",
                        "height",
                        "n_sides",
                        "vertices",
                        "theta_offset",
                        "R_inner",
                        "R_outer",
                        "opening_angle",
                        "theta_center",
                    }
                ),
                "circular_cavity",
            )
            if self.R is None:
                missing.append("R")
            elif self.R < 0:
                raise ValueError("geometry.R must be >= 0 for circular_cavity")
        elif t == "elliptic_cavity":
            forbid(
                frozenset(
                    {
                        "R",
                        "width",
                        "height",
                        "n_sides",
                        "vertices",
                        "theta_offset",
                        "R_inner",
                        "R_outer",
                        "opening_angle",
                    }
                ),
                "elliptic_cavity",
            )
            if self.a is None:
                missing.append("a")
            if self.b is None:
                missing.append("b")
        elif t == "polygon_cavity":
            forbid(
                frozenset(
                    {
                        "a",
                        "b",
                        "theta",
                        "width",
                        "height",
                        "R_inner",
                        "R_outer",
                        "opening_angle",
                        "theta_center",
                    }
                ),
                "polygon_cavity",
            )
            reg = self.n_sides is not None and self.R is not None
            expl = self.vertices is not None
            if reg == expl:
                raise ValueError(
                    "geometry.type='polygon_cavity' requires either (n_sides and R) "
                    "or vertices, exclusively"
                )
            if expl:
                for i, p in enumerate(self.vertices or []):
                    if len(p) != 2:
                        raise ValueError(f"geometry.vertices[{i}] must be [x, y]")
        elif t == "wedge_cavity":
            forbid(
                frozenset(
                    {
                        "R",
                        "a",
                        "b",
                        "theta",
                        "width",
                        "height",
                        "n_sides",
                        "vertices",
                        "theta_offset",
                    }
                ),
                "wedge_cavity",
            )
            if self.R_inner is None:
                missing.append("R_inner")
            if self.R_outer is None:
                missing.append("R_outer")
            if self.opening_angle is None:
                missing.append("opening_angle")
            if (
                self.R_inner is not None
                and self.R_outer is not None
                and not (0.0 < self.R_inner < self.R_outer)
            ):
                raise ValueError("wedge_cavity requires R_inner < R_outer (both > 0)")
            if self.opening_angle is not None and not (
                0.0 < self.opening_angle <= 2.0 * math.pi + 1e-9
            ):
                raise ValueError("wedge_cavity opening_angle must be in (0, 2π]")
        elif t == "rectangular_slot":
            forbid(
                frozenset(
                    {
                        "R",
                        "a",
                        "b",
                        "n_sides",
                        "vertices",
                        "theta_offset",
                        "R_inner",
                        "R_outer",
                        "opening_angle",
                        "theta_center",
                    }
                ),
                "rectangular_slot",
            )
            if self.width is None:
                missing.append("width")
            if self.height is None:
                missing.append("height")

        if missing:
            raise ValueError(
                f"geometry.type={t!r} missing required field(s): {missing} "
                f"(see docs for required keys per type)"
            )
        return self


class PorePressureSpec(BaseModel):
    """Optional Terzaghi/Biot pore-pressure modifier on normal stresses."""

    model_config = ConfigDict(extra="forbid")

    field: Literal["uniform", "hydrostatic"] = "uniform"
    p0: float = Field(ge=0.0)
    biot_alpha: float = Field(default=1.0, ge=0.0, le=1.0)


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
        "lithostatic",
        "tectonic_far_field",
        "inglis",
    ] = "none"
    sigma_0: float = 0.0
    stress_coupling_B: float = 0.0
    stress_eps_factor: float = Field(default=3.0, gt=0)
    rho_g_dim: float | None = Field(default=None, gt=0)
    lateral_K: float | None = Field(default=None, gt=0)
    S_H: float | None = None
    S_h: float | None = None
    S_V: float | None = None
    theta_SH: float | None = None
    R: float | None = Field(default=None, gt=0)
    a: float | None = Field(default=None, gt=0)
    b: float | None = Field(default=None, gt=0)
    theta: float | None = None
    S_xx_far: float | None = None
    S_yy_far: float | None = None
    S_xy_far: float | None = None
    pore_pressure: PorePressureSpec | None = None

    @model_validator(mode="after")
    def _stress_mode_fields(self) -> Self:
        m = self.mode
        d = self.model_dump()
        missing: list[str] = []

        def forbid(keys: frozenset[str], label: str) -> None:
            bad = [k for k in keys if d.get(k) is not None]
            if bad:
                raise ValueError(f"stress.mode={label!r} must not set: {sorted(bad)}")

        if m == "none":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "R",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                "none",
            )
        elif m in ("uniform_uniaxial", "uniform_biaxial", "pure_shear"):
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "R",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                m,
            )
        elif m == "flamant_two_point":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                "flamant_two_point",
            )
        elif m == "pressure_gradient":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "R",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                "pressure_gradient",
            )
        elif m == "lithostatic":
            forbid(
                frozenset(
                    {
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "R",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                "lithostatic",
            )
            if self.rho_g_dim is None:
                missing.append("rho_g_dim")
            if self.lateral_K is None:
                missing.append("lateral_K")
            elif not (0.0 < float(self.lateral_K) <= 1.0 + 1e-9):
                raise ValueError("lithostatic requires 0 < lateral_K <= 1")
        elif m == "tectonic_far_field":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "R",
                        "a",
                        "b",
                        "theta",
                        "S_xx_far",
                        "S_yy_far",
                        "S_xy_far",
                    }
                ),
                "tectonic_far_field",
            )
            for key in ("S_H", "S_h", "S_V"):
                if d.get(key) is None:
                    missing.append(key)
        elif m == "kirsch":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "a",
                        "b",
                        "theta",
                    }
                ),
                "kirsch",
            )
            if self.S_xx_far is None:
                missing.append("S_xx_far")
            if self.S_yy_far is None:
                missing.append("S_yy_far")
        elif m == "inglis":
            forbid(
                frozenset(
                    {
                        "rho_g_dim",
                        "lateral_K",
                        "S_H",
                        "S_h",
                        "S_V",
                        "theta_SH",
                        "R",
                    }
                ),
                "inglis",
            )
            if self.S_xx_far is None:
                missing.append("S_xx_far")
            if self.S_yy_far is None:
                missing.append("S_yy_far")
        if missing:
            raise ValueError(
                f"stress.mode={m!r} missing required field(s): {missing} (see docs for this mode)"
            )
        return self


class GravitySpec(BaseModel):
    """Optional gravity: rim ramp + body-force advection (Package 4)."""

    model_config = ConfigDict(extra="forbid")

    rim_alpha: float = 0.0
    g_c: float = 0.0
    g_phi_m: float = 0.0
    g_phi_c: float = 0.0
    g_phi_q: float = 0.0
    g_phi_imp: float = 0.0


class TimeSpec(BaseModel):
    """Horizon and snapshot cadence."""

    model_config = ConfigDict(extra="forbid")

    dt: float = Field(gt=0)
    T: float = Field(gt=0)
    snapshot_every: int = Field(default=500, ge=1)


class PhaseSpec(BaseModel):
    """One crystalline phase: bulk potential dispatch + CH mobility / bookkeeping."""

    model_config = ConfigDict(extra="forbid")

    potential: Literal["double_well", "tilted_well", "asymmetric_well", "zero"] = "double_well"
    potential_kwargs: dict[str, float] = Field(default_factory=dict)
    mobility: float = 1.0
    rho: float = 1.0
    psi_sign: float = 0.0
    active: bool = True


class PhasesSpec(BaseModel):
    """Moganite + chalcedony required; optional α-quartz and impurity placeholders."""

    model_config = ConfigDict(extra="forbid")

    moganite: PhaseSpec
    chalcedony: PhaseSpec
    alpha_quartz: PhaseSpec | None = None
    impurity: PhaseSpec | None = None


class AgingSpec(BaseModel):
    """Optional kinetic aging of moganite toward chalcedony / α-quartz."""

    model_config = ConfigDict(extra="forbid")

    active: bool = False
    k_age: float = Field(default=0.0, ge=0.0)
    q_to_quartz: float = Field(default=0.0, ge=0.0, le=1.0)


def apply_physics_phases_legacy_shim(physics: dict[str, Any]) -> None:
    """Inject ``physics['phases']`` from legacy flat keys when ``phases`` is absent (in-place).

    Legacy shim (potential dispatch refactor, Step 1): if ``physics.phases`` is
    missing, build it from optional ``W``, ``M_m``, ``M_c``, ``rho_m``,
    ``rho_c`` so existing canonical YAMLs stay unchanged. Same mapping as
    ``RunConfigValidated`` ``mode='before'`` validator.
    """
    if "phases" in physics:
        return
    physics["phases"] = {
        "moganite": {
            "potential": "double_well",
            "potential_kwargs": {"W": float(physics.get("W", 1.0))},
            "mobility": float(physics.get("M_m", 1.0)),
            "rho": float(physics.get("rho_m", 1.0)),
            "psi_sign": 1.0,
        },
        "chalcedony": {
            "potential": "double_well",
            "potential_kwargs": {"W": float(physics.get("W", 1.0))},
            "mobility": float(physics.get("M_c", 1.0)),
            "rho": float(physics.get("rho_c", 1.0)),
            "psi_sign": -1.0,
        },
    }


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
    gravity: GravitySpec = Field(default_factory=GravitySpec)
    time: TimeSpec
    output: OutputSpec = Field(default_factory=OutputSpec)
    initial: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _inject_physics_phases_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            phys = data.get("physics")
            if phys is None:
                data["physics"] = {}
                phys = data["physics"]
            if isinstance(phys, dict):
                apply_physics_phases_legacy_shim(phys)
        return data

    @model_validator(mode="after")
    def _normalize_physics_subdicts(self) -> Self:
        phys = self.physics
        phases = phys.get("phases")
        if phases is not None:
            phys["phases"] = PhasesSpec.model_validate(phases).model_dump(mode="python")
        ag = phys.get("aging")
        if ag is not None:
            phys["aging"] = AgingSpec.model_validate(ag).model_dump(mode="python")
        else:
            phys["aging"] = AgingSpec().model_dump(mode="python")
        return self

    @model_validator(mode="after")
    def _stress_geometry_coupling(self) -> Self:
        st = self.stress
        g = self.geometry
        if st.mode == "kirsch":
            if g.type != "circular_cavity":
                raise ValueError("stress.mode='kirsch' requires geometry.type='circular_cavity'")
            gr = g.R
            if gr is None:
                raise ValueError("kirsch requires geometry.R for circular_cavity")
            if st.R is not None and abs(float(st.R) - float(gr)) > 1e-9:
                raise ValueError(
                    f"stress.R={st.R!r} conflicts with geometry.R={gr!r} for kirsch mode"
                )
            if st.R is None:
                return self.model_copy(update={"stress": st.model_copy(update={"R": float(gr)})})
        elif st.mode == "inglis":
            if g.type != "elliptic_cavity":
                raise ValueError("stress.mode='inglis' requires geometry.type='elliptic_cavity'")
            ga, gb, gt = g.a, g.b, g.theta
            if ga is None or gb is None:
                raise ValueError("inglis requires geometry.a and geometry.b")
            gtheta = 0.0 if gt is None else float(gt)
            upd: dict[str, Any] = {}
            if st.a is not None and abs(float(st.a) - float(ga)) > 1e-9:
                raise ValueError(f"stress.a={st.a!r} conflicts with geometry.a={ga!r}")
            if st.b is not None and abs(float(st.b) - float(gb)) > 1e-9:
                raise ValueError(f"stress.b={st.b!r} conflicts with geometry.b={gb!r}")
            if st.theta is not None and abs(float(st.theta) - gtheta) > 1e-9:
                raise ValueError(f"stress.theta={st.theta!r} conflicts with geometry.theta")
            if st.a is None:
                upd["a"] = float(ga)
            if st.b is None:
                upd["b"] = float(gb)
            if st.theta is None and gt is not None:
                upd["theta"] = gtheta
            if upd:
                return self.model_copy(update={"stress": st.model_copy(update=upd)})
        return self


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

# Preset fragments merged after library+user defaults and before the experiment YAML
# (see Package 6 — ``initial.scenario``).
# Aging scenarios (``closed_aging``, ``open_aging``) initialise a small chalcedony seed density
# (``phi_c_init`` = 0.05) because uniform moganite is kinetically trapped by the double-well —
# see ``docs/PHYSICS.md`` §3.5.
SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "open_inflow": {
        "physics": {"reaction_active": True, "dirichlet_active": True},
        "initial": {
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_m_noise": 0.01,
            "phi_c_noise": 0.01,
        },
    },
    "closed_supersaturated": {
        "physics": {"reaction_active": True, "dirichlet_active": False},
        "initial": {
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_m_noise": 0.01,
            "phi_c_noise": 0.01,
            "c_init_factor": 1.5,
        },
    },
    "closed_aging": {
        "physics": {
            "reaction_active": False,
            "dirichlet_active": False,
            "aging": {"active": True, "k_age": 0.01, "q_to_quartz": 0.0},
        },
        "initial": {
            "phi_m_init": 0.95,
            "phi_c_init": 0.05,
            "phi_m_noise": 0.005,
            "phi_c_noise": 0.02,
            "c_init": 0.0,
        },
    },
    "open_aging": {
        "physics": {
            "reaction_active": True,
            "dirichlet_active": True,
            "aging": {"active": True, "k_age": 0.005, "q_to_quartz": 0.0},
        },
        "initial": {
            "phi_m_init": 0.8,
            "phi_c_init": 0.05,
            "phi_m_noise": 0.01,
            "phi_c_noise": 0.02,
        },
    },
    "bulk_relaxation": {
        "experiment": {"model": "bulk_relaxation"},
        "physics": {"reaction_active": False, "dirichlet_active": False},
        "initial": {
            "phi_m_init": 0.5,
            "phi_c_init": 0.5,
            "phi_m_noise": 0.05,
            "phi_c_noise": 0.05,
            "c_init": 0.0,
        },
    },
}

_warned_gif_hard_disabled: bool = False
_warned_h5_hard_disabled: bool = False


def _allow_expensive_output() -> bool:
    """When true, YAML may enable GIF / HDF5 snapshot paths (opt-in)."""
    v = os.environ.get("CP_ALLOW_EXPENSIVE_OUTPUT", "").strip().lower()
    return v in ("1", "true", "yes")


def _coerce_expensive_output_flags(merged: dict[str, Any]) -> None:
    """Force ``record_evolution_gif`` / ``save_snapshots_h5`` off unless env allows (in-place)."""
    global _warned_gif_hard_disabled, _warned_h5_hard_disabled
    if _allow_expensive_output():
        return
    outp = merged.setdefault("output", {})
    if bool(outp.get("record_evolution_gif")):
        outp["record_evolution_gif"] = False
        if not _warned_gif_hard_disabled:
            logger.warning(
                "GIF generation hard-disabled by default; "
                "set CP_ALLOW_EXPENSIVE_OUTPUT=1 to re-enable."
            )
            _warned_gif_hard_disabled = True
    if bool(outp.get("save_snapshots_h5")):
        outp["save_snapshots_h5"] = False
        if not _warned_h5_hard_disabled:
            logger.warning(
                "HDF5 snapshot writes hard-disabled by default; "
                "set CP_ALLOW_EXPENSIVE_OUTPUT=1 to re-enable."
            )
            _warned_h5_hard_disabled = True


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
    3. Scenario preset from ``initial.scenario`` when set (see ``SCENARIO_PRESETS``).
    4. Experiment YAML at ``experiment_path``.

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
    scenario_key: str | None = None
    init_raw = raw_exp.get("initial")
    if isinstance(init_raw, dict):
        sk = init_raw.get("scenario")
        if sk is not None:
            scenario_key = str(sk)
            if scenario_key not in SCENARIO_PRESETS:
                raise ValueError(
                    f"initial.scenario={scenario_key!r} is unknown; "
                    f"allowed: {sorted(SCENARIO_PRESETS)}"
                )
            merged = _deep_merge(merged, SCENARIO_PRESETS[scenario_key])
    merged = _deep_merge(merged, raw_exp)
    if scenario_key is not None:
        exp_block = merged.setdefault("experiment", {})
        if isinstance(exp_block, dict):
            exp_block.setdefault("scenario", scenario_key)
    _coerce_expensive_output_flags(merged)

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
            for field in ("phi_m", "phi_c", "phi_q", "phi_imp", "c"):
                if field not in snap:
                    continue
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
            row: dict[str, Any] = {
                "step": int(g.attrs["step"]),
                "t": float(g.attrs["t"]),
                "phi_m": np.asarray(g["phi_m"]),
                "phi_c": np.asarray(g["phi_c"]),
                "c": np.asarray(g["c"]),
            }
            if "phi_q" in g:
                row["phi_q"] = np.asarray(g["phi_q"])
            if "phi_imp" in g:
                row["phi_imp"] = np.asarray(g["phi_imp"])
            out.append(row)
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
    phi_q: np.ndarray | None = None,
    phi_imp: np.ndarray | None = None,
    chi: np.ndarray | None = None,
) -> None:
    """Write compressed ``final_state.npz`` with NumPy arrays."""
    data: dict[str, np.ndarray] = {
        "phi_m": np.asarray(phi_m),
        "phi_c": np.asarray(phi_c),
        "c": np.asarray(c),
    }
    if phi_q is not None:
        data["phi_q"] = np.asarray(phi_q)
    if phi_imp is not None:
        data["phi_imp"] = np.asarray(phi_imp)
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
