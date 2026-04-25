"""Shared result types for integrators and I/O (avoid circular imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import Array

from continuous_patterns.core.io import ResultPaths


@dataclass(frozen=True)
class PhasePotentialParams:
    """Static per-phase CH knobs and bulk-potential dispatch key (no JAX arrays).

    Used by :class:`continuous_patterns.core.imex.SimParams` and built from YAML
    ``physics.phases.*`` after validation. ``kind`` selects a callable in
    :data:`continuous_patterns.core.potentials.POTENTIAL_BUILDERS`.
    """

    kind: str = "double_well"
    W: float = 1.0
    tilt: float = 0.0
    phi_left: float = 0.0
    phi_right: float = 1.0
    mobility: float = 1.0
    rho: float = 1.0
    psi_sign: float = 0.0
    active: bool = True


@dataclass(frozen=True)
class SimState:
    """Model fields at a single time level."""

    phi_m: Array
    phi_c: Array
    phi_q: Array
    phi_imp: Array
    c: Array
    t: float


@dataclass
class SimResult:
    """Outcome of a full simulation run."""

    state_final: SimState
    meta: dict[str, Any]
    diagnostics: dict[str, Any]
    config_resolved: dict[str, Any]
    paths: ResultPaths | None = None
