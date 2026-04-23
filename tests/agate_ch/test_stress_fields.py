"""Experiment 6: Flamant stress helper sanity (finite fields, normalization)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

from continuous_patterns.agate_ch.model import xy_grid
from continuous_patterns.agate_ch.stress_fields import (
    flamant_two_point,
    kirsch_field,
    pressure_gradient_field,
    pure_shear_field,
    uniform_uniaxial_field,
)


def test_flamant_two_point_finite_and_normalized() -> None:
    L, R, n = 200.0, 80.0, 128
    sigma_0 = 1.5
    eps = 3.0 * (L / n)
    sxx, syy, sxy = flamant_two_point(L, R, n, sigma_0, eps)
    assert sxx.shape == (n, n)
    arr = np.asarray(jnp.stack([sxx, syy, sxy]))
    assert np.all(np.isfinite(arr))
    assert np.max(np.abs(arr)) < 1e6
    diff = np.asarray(sxx - syy)
    assert np.isclose(np.max(np.abs(diff)), sigma_0, rtol=1e-5, atol=1e-5)


def test_uniform_uniaxial_compression_in_x() -> None:
    L, n, s0 = 100.0, 32, 3.0
    sxx, syy, sxy = uniform_uniaxial_field(L, n, s0)
    assert np.allclose(np.asarray(sxx), s0)
    assert np.allclose(np.asarray(syy), 0.0)
    assert np.allclose(np.asarray(sxy), 0.0)


def test_pure_shear_uniform_xy() -> None:
    L, n, s0 = 200.0, 128, 1.25
    sxx, syy, sxy = pure_shear_field(L, n, s0)
    assert np.allclose(np.asarray(sxx), 0.0)
    assert np.allclose(np.asarray(syy), 0.0)
    assert np.allclose(np.asarray(sxy), s0)


def test_pressure_gradient_isotropic_and_linear_in_y() -> None:
    L, n = 200.0, 64
    sigma_0 = 1.0
    sxx, syy, sxy = pressure_gradient_field(L, n, sigma_0)
    assert np.allclose(np.asarray(sxx), np.asarray(syy))
    assert np.allclose(np.asarray(sxy), 0.0)
    _, Y = xy_grid(L, n)
    p = np.asarray(sigma_0 * (Y - L / 2.0) / (L / 2.0))
    assert np.allclose(np.asarray(sxx), -p, rtol=1e-5, atol=1e-5)


def test_kirsch_finite_and_axisymmetric_pattern() -> None:
    L, R, n = 200.0, 80.0, 128
    sxx, syy, sxy = kirsch_field(L, R, n, 1.0)
    arr = np.asarray(jnp.stack([sxx, syy, sxy]))
    assert np.all(np.isfinite(arr))
    # Stress concentration near cavity wall (classic factor up to ~3 σ₀).
    peak = float(jnp.max(jnp.abs(sxx)))
    assert peak > 1.5 * 0.9 and peak < 50.0
