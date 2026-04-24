"""Unit tests for :mod:`continuous_patterns.core.spectral`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.spectral import (
    divergence_real,
    grad_real,
    k_vectors,
    laplacian_real,
)


def _cell_centred_xy(*, L: float, n: int) -> tuple[jax.Array, jax.Array]:
    """Cell-centre coordinates in float64 for sampling smooth test fields."""
    dx = L / n
    i = jnp.arange(n, dtype=jnp.float64)[:, None]
    j = jnp.arange(n, dtype=jnp.float64)[None, :]
    x = jnp.broadcast_to((i + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((j + 0.5) * dx, (n, n))
    return x, y


def test_fft_round_trip_identity() -> None:
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (48, 48), dtype=jnp.float32)
    u_rec = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(u)))
    assert jnp.allclose(u, u_rec, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_laplacian_sine_eigenmode_x_float64(p: int) -> None:
    """Pseudospectral Laplacian is machine-precision exact on band-limited sin at float64."""
    L, n = 10.0, 64
    kx = 2.0 * jnp.pi * p / L
    x, _ = _cell_centred_xy(L=L, n=n)
    u = jnp.sin(kx * x).astype(jnp.float64)
    k_sq, _, _, _, _, _ = k_vectors(L=L, n=n)
    lap = laplacian_real(u, k_sq)
    expected = -(kx**2) * u
    err = jnp.max(jnp.abs(lap - expected))
    assert float(err) < 1e-10


@pytest.mark.parametrize("p", [1, 2, 3])
def test_laplacian_sine_eigenmode_x_float32(p: int) -> None:
    """Float32 state should stay within a practical error budget vs continuum −k²·sin."""
    L, n = 10.0, 64
    kx = 2.0 * jnp.pi * p / L
    x, _ = _cell_centred_xy(L=L, n=n)
    u = jnp.sin(kx * x).astype(jnp.float32)
    k_sq, _, _, _, _, _ = k_vectors(L=L, n=n)
    lap = laplacian_real(u, k_sq)
    expected = (-(kx**2) * u).astype(jnp.float32)
    err = jnp.max(jnp.abs(lap - expected))
    assert float(err) < 1e-4


@pytest.mark.parametrize("p", [1, 2, 3])
def test_spectral_gradient_sign_cos_kx(p: int) -> None:
    r"""``\partial/\partial x \cos(k x) = -k \sin(k x)`` via ``kx_wave`` / FFT."""
    L, n = 10.0, 64
    kx_val = 2.0 * jnp.pi * p / L
    x, _ = _cell_centred_xy(L=L, n=n)
    u = jnp.cos(kx_val * x).astype(jnp.float64)
    _, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
    gx, gy = grad_real(u, kx_wave, ky_wave)
    expected_gx = (-kx_val * jnp.sin(kx_val * x)).astype(jnp.float64)
    assert jnp.allclose(gx, expected_gx, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(gy, jnp.zeros_like(gy), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_gradient_sine_matches_analytic_float64(axis: str) -> None:
    """Gradients of low-mode sines match analytic derivatives at float64 (float32 redundant)."""
    L, n = 10.0, 64
    x, y = _cell_centred_xy(L=L, n=n)
    k1 = 2.0 * jnp.pi / L
    if axis == "x":
        u = jnp.sin(k1 * x).astype(jnp.float64)
        _, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
        gx, gy = grad_real(u, kx_wave, ky_wave)
        gx_e = (k1 * jnp.cos(k1 * x)).astype(jnp.float64)
        gy_e = jnp.zeros_like(gx_e)
    else:
        u = jnp.sin(k1 * y).astype(jnp.float64)
        _, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
        gx, gy = grad_real(u, kx_wave, ky_wave)
        gx_e = jnp.zeros_like(u)
        gy_e = (k1 * jnp.cos(k1 * y)).astype(jnp.float64)
    assert jnp.allclose(gx, gx_e, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(gy, gy_e, rtol=1e-12, atol=1e-12)


def test_divergence_grad_equals_laplacian_smooth() -> None:
    L, n = 8.0, 32
    x, y = _cell_centred_xy(L=L, n=n)
    u = (jnp.sin(2.0 * jnp.pi * x / L) * jnp.cos(4.0 * jnp.pi * y / L)).astype(jnp.float32)
    k_sq, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
    gx, gy = grad_real(u, kx_wave, ky_wave)
    div_g = divergence_real(gx, gy, kx_wave, ky_wave)
    lap_u = laplacian_real(u, k_sq)
    assert jnp.allclose(div_g, lap_u, rtol=2e-4, atol=2e-5)


def test_k_vectors_symmetry_and_dc() -> None:
    L, n = 7.0, 48
    k_sq, _, _, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    assert kx_wave.shape == ky_wave.shape == (n, n)
    assert k_four.shape == (n, n)
    assert jnp.allclose(k_four, k_sq * k_sq)
    assert float(jnp.min(k_sq)) >= 0.0
    assert float(k_sq[0, 0]) == pytest.approx(0.0, abs=1e-12)


def test_k_vectors_small_n() -> None:
    k_sq, _, _, kx_wave, ky_wave, k_four = k_vectors(L=1.0, n=4)
    assert kx_wave.shape == (4, 4)
    assert float(k_sq[0, 0]) == pytest.approx(0.0, abs=1e-14)
    assert float(jnp.min(k_sq)) >= 0.0
    assert jnp.allclose(k_four, k_sq * k_sq)


def test_k_vectors_arbitrary_L_period() -> None:
    """Non-commensurate L should not assume L is 2π or a power of two."""
    L, n = 7.0, 16
    k_sq, _, _, kx_wave, ky_wave, _ = k_vectors(L=L, n=n)
    assert kx_wave.shape == (n, n)
    assert float(jnp.min(k_sq)) >= 0.0
