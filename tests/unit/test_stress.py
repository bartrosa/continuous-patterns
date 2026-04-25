"""Unit tests for :mod:`continuous_patterns.core.stress`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.spectral import divergence_real, grad_real, k_vectors, laplacian_real
from continuous_patterns.core.stress import (
    STRESS_BUILDERS,
    apply_pore_pressure,
    flamant_two_point,
    inglis,
    kirsch,
    lithostatic,
    none,
    pressure_gradient,
    pure_shear,
    stress_contribution_to_mu,
    stress_mu_hat,
    tectonic_far_field,
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
        lambda: lithostatic(L=L, n=n, rho_g_dim=0.01, lateral_K=0.5),
        lambda: tectonic_far_field(L=L, n=n, S_H=1.0, S_h=0.4, S_V=0.7, theta_SH=0.1),
        lambda: kirsch(L=L, R=R, n=n, S_xx_far=1.0, S_yy_far=0.2, S_xy_far=0.05),
        lambda: inglis(L=L, n=n, a=R, b=R, theta=0.0, S_xx_far=1.0, S_yy_far=0.0),
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
    assert float(jnp.max(jnp.abs(dev))) == pytest.approx(float(s0), rel=0, abs=5e-6)
    max_entry = jnp.max(jnp.abs(sxx)) + jnp.max(jnp.abs(syy)) + jnp.max(jnp.abs(sxy))
    assert float(max_entry) < 15.0 * float(s0)


def test_flamant_symmetric_about_vertical_midline() -> None:
    L, n, R = 10.0, 48, 2.0
    sxx, _, _ = flamant_two_point(L=L, R=R, n=n, sigma_0=0.3)
    mirror = sxx[::-1, :]
    assert jnp.allclose(sxx, mirror, rtol=1e-6, atol=1e-6)


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


def test_kirsch_far_field_asymptotic() -> None:
    """At large ``r``, ``σ`` approaches the remote tensor (Kirsch)."""
    L, R, n = 200.0, 20.0, 256
    sxx, syy, sxy = kirsch(
        L=L, R=R, n=n, S_xx_far=1.0, S_yy_far=0.3, S_xy_far=0.1, dtype=jnp.float64
    )
    dx = L / n
    ii = jnp.arange(n, dtype=jnp.float64)[:, None]
    jj = jnp.arange(n, dtype=jnp.float64)[None, :]
    x = (ii + 0.5) * dx
    y = (jj + 0.5) * dx
    xc = 0.5 * L
    yc = 0.5 * L
    r = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    r_max = jnp.max(r)
    mask = r > 0.85 * r_max
    assert float(jnp.max(jnp.abs(sxx - 1.0)[mask])) < 0.1
    assert float(jnp.max(jnp.abs(syy - 0.3)[mask])) < 0.1
    assert float(jnp.max(jnp.abs(sxy - 0.1)[mask])) < 0.1


def test_kirsch_builder_dispatch() -> None:
    sxx, syy, sxy = STRESS_BUILDERS["kirsch"](
        L=10.0, R=2.0, n=32, S_xx_far=-0.2, S_yy_far=0.4, S_xy_far=0.0
    )
    ref = kirsch(L=10.0, R=2.0, n=32, S_xx_far=-0.2, S_yy_far=0.4, S_xy_far=0.0)
    for a, b in zip((sxx, syy, sxy), ref, strict=True):
        assert jnp.allclose(a, b)


def test_lithostatic_compression_and_hydrostatic() -> None:
    L, n = 10.0, 64
    sxx, syy, sxy = lithostatic(L=L, n=n, rho_g_dim=0.02, lateral_K=0.5)
    assert jnp.all(syy < 0)
    assert jnp.all(sxx < 0)
    assert jnp.allclose(sxy, 0.0)
    sxx1, syy1, sxy1 = lithostatic(L=L, n=n, rho_g_dim=0.02, lateral_K=1.0)
    assert jnp.allclose(sxx1, syy1)
    assert jnp.allclose(sxy1, 0.0)


def test_tectonic_far_field_rotation() -> None:
    L, n = 12.0, 48
    sxx, syy, sxy = tectonic_far_field(
        L=L, n=n, S_H=1.1, S_h=0.2, S_V=0.5, theta_SH=0.0, dtype=jnp.float64
    )
    assert jnp.allclose(sxx, 1.1)
    assert jnp.allclose(syy, 0.2)
    assert jnp.allclose(sxy, 0.0)
    th = jnp.pi / 4.0
    sx, sy, sxy45 = tectonic_far_field(
        L=L, n=n, S_H=2.0, S_h=1.0, S_V=0.0, theta_SH=float(th), dtype=jnp.float64
    )
    c = float(jnp.cos(th))
    s = float(jnp.sin(th))
    ex = 2.0 * c * c + 1.0 * s * s
    ey = 2.0 * s * s + 1.0 * c * c
    exy = (2.0 - 1.0) * c * s
    assert jnp.allclose(sx, ex)
    assert jnp.allclose(sy, ey)
    assert jnp.allclose(sxy45, exy)


def test_inglis_circle_matches_kirsch() -> None:
    L, R, n = 50.0, 12.0, 128
    a = inglis(
        L=L,
        n=n,
        a=R,
        b=R,
        theta=0.0,
        S_xx_far=1.0,
        S_yy_far=0.0,
        S_xy_far=0.0,
        dtype=jnp.float32,
    )
    k = kirsch(L=L, R=R, n=n, S_xx_far=1.0, S_yy_far=0.0, S_xy_far=0.0, dtype=jnp.float32)
    for u, v in zip(a, k, strict=True):
        assert jnp.allclose(u, v, atol=1e-5, rtol=1e-4)


def test_apply_pore_pressure_biot() -> None:
    n = 8
    sxx = jnp.ones((n, n))
    syy = 2.0 * jnp.ones((n, n))
    sxy = 0.5 * jnp.ones((n, n))
    p = 0.25 * jnp.ones((n, n))
    ox, oy, oxy = apply_pore_pressure(sxx, syy, sxy, p_pore=p, biot_alpha=0.8)
    assert jnp.allclose(ox, 1.0 - 0.8 * 0.25)
    assert jnp.allclose(oy, 2.0 - 0.8 * 0.25)
    assert jnp.allclose(oxy, 0.5)


def test_stress_mu_hat_pure_shear_matches_mixed_derivative() -> None:
    """Pure shear ``σ_xy`` must couple via mixed derivatives (guards Phase 2 Bug B)."""
    L, n = 10.0, 64
    kx = 2.0 * jnp.pi / L
    ky = 2.0 * jnp.pi / L
    x, y, kx_wave, ky_wave, _ = _grid_psi_k(L=L, n=n)
    psi = (jnp.sin(kx * x) * jnp.sin(ky * y)).astype(jnp.float64)
    s0, B = 0.3, 1.0
    sxx, syy, sxy = pure_shear(L=L, n=n, sigma_0=s0)
    mu_hat = stress_mu_hat(psi, sxx, sxy, syy, kx_wave, ky_wave, B)
    mu = jnp.real(jnp.fft.ifft2(mu_hat))
    expected = -2.0 * B * s0 * kx * ky * jnp.cos(kx * x) * jnp.cos(ky * y)
    assert jnp.allclose(mu, expected, rtol=2e-5, atol=2e-5)


def test_sigma_symmetry_in_coupling() -> None:
    """Same ``σ_xy`` in both flux rows matches the ``stress_contribution_to_mu`` path."""
    L, n = 12.0, 32
    x, y, kx_wave, ky_wave, _ = _grid_psi_k(L=L, n=n)
    phi_m = jnp.sin(2.0 * jnp.pi * x / L).astype(jnp.float64)
    phi_c = 0.15 * jnp.cos(2.0 * jnp.pi * y / L).astype(jnp.float64)
    sxx, syy, sxy = pure_shear(L=L, n=n, sigma_0=0.4)
    B = 0.85
    dm, dc = stress_contribution_to_mu(phi_m, phi_c, sxx, sxy, syy, kx_wave, ky_wave, B)
    psi = phi_m - phi_c
    div = divergence_from_sigma_psi(psi, sxx, sxy, sxy, syy, kx_wave, ky_wave)
    assert jnp.allclose(dm - dc, -B * div, rtol=1e-12, atol=1e-12)


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
    assert jnp.allclose(mu, expected, rtol=2e-5, atol=2e-5)


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
    assert float(jnp.abs(total)) < 1e-5


def test_stress_builders_none_dispatch() -> None:
    a = none(L=5.0, n=8)
    b = STRESS_BUILDERS["none"](L=5.0, n=8)
    for u, v in zip(a, b, strict=True):
        assert jnp.allclose(u, v)
