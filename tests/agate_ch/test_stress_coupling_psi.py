"""Stress coupling uses ψ = φ_m − φ_c (Experiment 6 fix)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytest.importorskip("jax")

from continuous_patterns.agate_ch.model import build_geometry
from continuous_patterns.agate_ch.solver import stress_contribution_to_mu


def test_psi_zero_implies_no_stress_mu() -> None:
    geom = build_geometry(
        200.0,
        80.0,
        64,
        stress_mode="pure_shear",
        sigma_0=2.5,
    )
    n = 64
    base = jnp.linspace(0.0, 0.2, n * n, dtype=jnp.float32).reshape(n, n)
    phim = base
    phic = base
    dmm, dcc = stress_contribution_to_mu(phim, phic, geom, jnp.float32(1.0))
    assert float(jnp.max(jnp.abs(dmm))) < 1e-7
    assert float(jnp.max(jnp.abs(dcc))) < 1e-7


def test_stress_split_opposes_sign() -> None:
    import jax

    geom = build_geometry(
        200.0,
        80.0,
        64,
        stress_mode="pure_shear",
        sigma_0=1.0,
    )
    n = 64
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    phim = jax.random.uniform(k1, (n, n), dtype=jnp.float32) * 0.1
    phic = jax.random.uniform(k2, (n, n), dtype=jnp.float32) * 0.05
    dmm, dcc = stress_contribution_to_mu(phim, phic, geom, jnp.float32(1.0))
    assert float(jnp.max(jnp.abs(dmm + dcc))) < 1e-6
