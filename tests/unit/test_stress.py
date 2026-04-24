"""Unit tests for :mod:`continuous_patterns.core.stress`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.spectral import divergence_real, grad_real, k_vectors, laplacian_real
from continuous_patterns.core.stress import (
    STRESS_BUILDERS,
    flamant_two_point,
    kirsch,
    none,
    pressure_gradient,
    pure_shear,
    stress_contribution_to_mu,
    stress_mu_hat,
    uniform_biaxial,
    uniform_uniaxial,
)


def _grid_psi_k(
    *, L: float, n: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dx = L / n
    dtype = jnp.float64
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    k_sq, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
    return x, y, kx_wave, ky_wave, k_sq


def test_uniform_uniaxial_zero_sigma_all_zero() -> None:
    sxx, syy, sxy = uniform_uniaxial(L=10.0, n=16, sigma_0=0.0)
    assert sxx.shape == syy.shape == sxy.shape == (16, 16)
    assert float(jnp.max(jnp.abs(sxx))) == 0.0
    assert float(jnp.max(jnp.abs(syy))) == 0.0
    assert float(jnp.max(jnp.abs(sxy))) == 0.0


def test_pure_shear_zero_sigma_all_zero() -> None:
    sxx, syy, sxy = pure_shear(L=10.0, n=16, sigma_0=0.0)
    assert float(jnp.max(jnp.abs(sxx))) == 0.0
    assert float(jnp.max(jnp.abs(syy))) == 0.0
    assert float(jnp.max(jnp.abs(sxy))) == 0.0


def test_all_builders_shapes() -> None:
    L, n, R = 8.0, 24, 2.0
    builders = [
        lambda: none(L=L, n=n),
        lambda: uniform_uniaxial(L=L, n=n, sigma_0=0.3),
        lambda: uniform_biaxial(L=L, n=n, sigma_0=0.3),
        lambda: pure_shear(L=L, n=n, sigma_0=0.3),
        lambda: flamant_two_point(L=L, R=R, n=n, sigma_0=0.25),
        lambda: pressure_gradient(L=L, n=n, sigma_0=0.2),
    ]
    for b in builders:
        sxx, syy, sxy = b()
        assert sxx.shape == (n, n)


def test_uniform_uniaxial_constant() -> None:
    sxx, syy, sxy = uniform_uniaxial(L=10.0, n=32, sigma_0=0.5)
    assert jnp.allclose(sxx, 0.5)
    assert jnp.allclose(syy, 0.0)
    assert jnp.allclose(sxy, 0.0)


def test_uniform_biaxial_symmetric() -> None:
    sxx, syy, sxy = uniform_biaxial(L=10.0, n=20, sigma_0=-0.25)
    assert jnp.allclose(sxx, -0.25)
    assert jnp.allclose(syy, -0.25)
    assert jnp.allclose(sxy, 0.0)


def test_pure_shear_only_offdiagonal() -> None:
    sxx, syy, sxy = pure_shear(L=10.0, n=18, sigma_0=0.4)
    assert jnp.allclose(sxx, 0.0)
    assert jnp.allclose(syy, 0.0)
    assert jnp.allclose(sxy, 0.4)


def test_flamant_finite_and_rescaled() -> None:
    L, n, R, s0 = 10.0, 64, 2.5, 0.25
    sxx, syy, sxy = flamant_two_point(L=L, R=R, n=n, sigma_0=s0, stress_eps_factor=3.0)
    assert jnp.all(jnp.isfinite(sxx))
    assert jnp.all(jnp.isfinite(syy))
    assert jnp.all(jnp.isfinite(sxy))
    dev = sxx - syy
    assert float(jnp.max(jnp.abs(dev))) == pytest.approx(float(s0), rel=0, abs=1e-10)


def test_flamant_symmetric_about_vertical_midline() -> None:
    L, n, R = 10.0, 48, 2.0
    sxx, _, _ = flamant_two_point(L=L, R=R, n=n, sigma_0=0.3)
    mirror = sxx[::-1, :]
    assert jnp.allclose(sxx, mirror, rtol=1e-12, atol=1e-12)


def test_pressure_gradient_linearity_and_isotropy() -> None:
    L, n, s0 = 10.0, 64, 0.3
    sxx, syy, sxy = pressure_gradient(L=L, n=n, sigma_0=s0)
    assert jnp.allclose(sxx, syy)
    assert jnp.allclose(sxy, 0.0)
    _, y, _, _, _ = _grid_psi_k(L=L, n=n)
    p = s0 * (y - 0.5 * L) / (0.5 * L)
    assert jnp.allclose(sxx, -p)
    j_mid = int(jnp.argmin(jnp.abs(y[0, :] - 0.5 * L)))
    j_lo, j_top = 0, n - 1
    dx = L / n
    # Cell centres miss ``y = L/2`` when ``n`` is even; error is ``O(s0 * dx / L)``.
    assert float(sxx[0, j_mid]) == pytest.approx(0.0, abs=0.02)
    assert float(sxx[0, j_top]) == pytest.approx(-s0, rel=0, abs=2.0 * s0 * dx / L)
    assert float(sxx[0, j_lo]) == pytest.approx(s0, rel=0, abs=2.0 * s0 * dx / L)


def test_kirsch_and_builder_raise() -> None:
    with pytest.raises(NotImplementedError):
        kirsch(L=10.0, R=2.0, n=16, sigma_0=0.1)
    with pytest.raises(NotImplementedError):
        STRESS_BUILDERS["kirsch"](L=10.0, R=2.0, n=16, sigma_0=0.1)


def test_stress_mu_hat_uniform_uniaxial_matches_laplacian_identity() -> None:
    """For constant ``σ_xx`` and ``ψ = sin(kx x)``, ``μ = -B σ_xx ψ_xx`` (no mixed terms)."""
    L, n = 10.0, 64
    kx = 2.0 * jnp.pi / L
    x, _, kx_wave, ky_wave, k_sq = _grid_psi_k(L=L, n=n)
    psi = jnp.sin(kx * x).astype(jnp.float64)
    s0, B = 0.35, 1.1
    sxx, syy, sxy = uniform_uniaxial(L=L, n=n, sigma_0=s0)
    mu_hat = stress_mu_hat(psi, sxx, sxy, syy, kx_wave, ky_wave, B)
    mu = jnp.real(jnp.fft.ifft2(mu_hat))
    lap = laplacian_real(psi, k_sq)
    expected = -B * s0 * lap
    assert jnp.allclose(mu, expected, rtol=1e-9, atol=1e-9)


def test_split_sum_and_difference() -> None:
    L, n = 8.0, 40
    x, y, kx_wave, ky_wave, _ = _grid_psi_k(L=L, n=n)
    phi_m = jnp.sin(2.0 * jnp.pi * x / L).astype(jnp.float64)
    phi_c = jnp.cos(2.0 * jnp.pi * y / L).astype(jnp.float64) * 0.25
    sxx, syy, sxy = uniform_biaxial(L=L, n=n, sigma_0=0.2)
    B = 0.7
    dm, dc = stress_contribution_to_mu(phi_m, phi_c, sxx, sxy, syy, kx_wave, ky_wave, B)
    assert jnp.allclose(dm + dc, 0.0, rtol=0, atol=1e-14)
    mu = dm - dc
    psi = phi_m - phi_c
    mu_direct = -B * divergence_from_sigma_psi(psi, sxx, sxy, sxy, syy, kx_wave, ky_wave)
    assert jnp.allclose(mu, mu_direct, rtol=1e-10, atol=1e-10)


def divergence_from_sigma_psi(
    psi: jax.Array,
    sxx: jax.Array,
    sxy_a: jax.Array,
    sxy_b: jax.Array,
    syy: jax.Array,
    kx: jax.Array,
    ky: jax.Array,
) -> jax.Array:
    gx, gy = grad_real(psi, kx, ky)
    fx = sxx * gx + sxy_a * gy
    fy = sxy_b * gx + syy * gy
    return divergence_real(fx, fy, kx, ky)


def test_mu_stress_periodic_mean_zero() -> None:
    """Integral of ``μ_stress`` over the torus ≈ 0 (smooth ``ψ``, uniform ``σ``)."""
    L, n = 1.0, 32
    x, y, kx_wave, ky_wave, k_sq = _grid_psi_k(L=L, n=n)
    psi = (
        jnp.sin(2.0 * jnp.pi * x / L) * jnp.cos(2.0 * jnp.pi * y / L)
        + 0.3 * jnp.sin(4.0 * jnp.pi * x / L)
    ).astype(jnp.float64)
    sxx, syy, sxy = uniform_uniaxial(L=L, n=n, sigma_0=0.5)
    B = 1.0
    mu_hat = stress_mu_hat(psi, sxx, sxy, syy, kx_wave, ky_wave, B)
    mu = jnp.real(jnp.fft.ifft2(mu_hat))
    dx = L / n
    total = jnp.sum(mu) * (dx * dx)
    assert float(jnp.abs(total)) < 1e-12


def test_stress_builders_none_dispatch() -> None:
    a = none(L=5.0, n=8)
    b = STRESS_BUILDERS["none"](L=5.0, n=8)
    for u, v in zip(a, b, strict=True):
        assert jnp.allclose(u, v)
