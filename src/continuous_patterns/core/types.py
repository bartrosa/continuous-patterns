"""Shared result types for integrators and I/O (avoid circular imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import Array

from continuous_patterns.core.io import ResultPaths


@dataclass(frozen=True)
class SimState:
    """Model fields at a single time level."""

    phi_m: Array
    phi_c: Array
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
