"""Unit tests for :mod:`continuous_patterns.core.potentials`."""

from __future__ import annotations

import jax.numpy as jnp

from continuous_patterns.core.potentials import (
    POTENTIAL_BUILDERS,
    asymmetric_well_prime,
    barrier_prime,
    double_well_prime,
    tilted_well_prime,
    zero_potential,
)


def test_prime_functions_preserve_shape_dtype() -> None:
    phi = jnp.linspace(-0.1, 1.1, 12, dtype=jnp.float64).reshape(3, 4)
    for fn, kwargs in (
        (double_well_prime, {"W": 1.0}),
        (tilted_well_prime, {"W": 1.0, "tilt": 0.1}),
        (asymmetric_well_prime, {"W": 1.0, "phi_left": 0.1, "phi_right": 0.9}),
        (barrier_prime, {"lambda_bar": 10.0}),
    ):
        out = fn(phi, **kwargs)
        assert out.shape == phi.shape
        assert out.dtype == phi.dtype
    z = zero_potential(phi)
    assert z.shape == phi.shape
    assert z.dtype == phi.dtype
    assert jnp.allclose(z, 0.0)


def test_double_well_zeros() -> None:
    for v in (0.0, 0.5, 1.0):
        phi = jnp.asarray(v, dtype=jnp.float64)
        assert float(double_well_prime(phi, W=1.0)) == 0.0


def test_tilted_well_matches_double_well_when_tilt_zero() -> None:
    phi = jnp.linspace(0.0, 1.0, 17, dtype=jnp.float64)
    a = double_well_prime(phi, W=2.0)
    b = tilted_well_prime(phi, W=2.0, tilt=0.0)
    assert jnp.allclose(a, b, rtol=0.0, atol=1e-15)


def test_asymmetric_well_zeros() -> None:
    pl, pr = 0.15, 0.85
    mid = 0.5 * (pl + pr)
    for v in (pl, pr, mid):
        phi = jnp.asarray(v, dtype=jnp.float64)
        assert float(asymmetric_well_prime(phi, W=1.0, phi_left=pl, phi_right=pr)) == 0.0


def test_barrier_prime_legacy_regression() -> None:
    """Pointwise match to former ``_barrier_prime`` in ``imex`` (quadratic outside [0,1])."""
    lam = 10.0
    for v in (-0.2, 0.0, 0.5, 1.0, 1.2):
        phi = jnp.asarray(v, dtype=jnp.float64)
        neg_excess = jnp.maximum(-phi, 0.0)
        pos_excess = jnp.maximum(phi - 1.0, 0.0)
        legacy = -2.0 * lam * neg_excess + 2.0 * lam * pos_excess
        assert float(barrier_prime(phi, lambda_bar=lam)) == float(legacy)


def test_potential_builders_dispatch() -> None:
    phi = jnp.array([[0.2, 0.7], [0.4, 0.9]], dtype=jnp.float64)
    cases: list[tuple[str, dict, jnp.ndarray]] = [
        ("double_well", {"W": 1.5}, double_well_prime(phi, W=1.5)),
        ("tilted_well", {"W": 1.0, "tilt": 0.03}, tilted_well_prime(phi, W=1.0, tilt=0.03)),
        (
            "asymmetric_well",
            {"W": 1.0, "phi_left": 0.0, "phi_right": 1.0},
            asymmetric_well_prime(phi, W=1.0, phi_left=0.0, phi_right=1.0),
        ),
        ("zero", {}, zero_potential(phi)),
    ]
    for kind, kwargs, expected in cases:
        builder = POTENTIAL_BUILDERS[kind]
        got = builder(phi, **kwargs)
        assert jnp.allclose(got, expected, rtol=0.0, atol=1e-15)
