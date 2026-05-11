"""Unit tests for :mod:`continuous_patterns.core.spectral_ops`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.spectral_ops import NeumannPeriodicOps, PeriodicOps


def test_periodic_ops_matches_legacy_fft_roundtrip() -> None:
    """PeriodicOps forward/inverse exactly mirror fft2/ifft2 calls."""
    n, L = 64, 10.0
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (n, n), dtype=jnp.float32)
    ops = PeriodicOps(n=n, L=L)

    legacy_hat = jnp.fft.fft2(field)
    via_ops_hat = ops.forward(field)
    assert jnp.array_equal(legacy_hat, via_ops_hat)

    legacy = jnp.fft.ifft2(legacy_hat)
    via_ops = ops.inverse(via_ops_hat)
    assert jnp.array_equal(legacy, via_ops)


def test_periodic_ops_symbols_match_k_vectors() -> None:
    n, L = 32, 7.5
    ops = PeriodicOps(n=n, L=L)
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)

    assert jnp.array_equal(ops.laplacian_symbol(), k_sq)
    assert jnp.array_equal(ops.kx_sq_symbol(), kx_sq)
    assert jnp.array_equal(ops.ky_sq_symbol(), ky_sq)
    assert jnp.array_equal(ops.kx_wave_symbol(), kx_wave)
    assert jnp.array_equal(ops.ky_wave_symbol(), ky_wave)
    assert jnp.array_equal(ops.biharmonic_symbol(), k_four)


def test_neumann_periodic_forward_inverse_roundtrip() -> None:
    n, L = 48, 10.0
    key = jax.random.PRNGKey(42)
    field = jax.random.normal(key, (n, n), dtype=jnp.float32)
    ops = NeumannPeriodicOps(n=n, L=L)

    field_back = ops.inverse(ops.forward(field))
    assert jnp.allclose(field_back.real, field, rtol=1e-5, atol=1e-6)


def test_neumann_periodic_no_flux_in_x_spectrum() -> None:
    """DCT-II x-spectrum uses nonnegative ``m*pi/L`` modes for Neumann walls."""
    n, L = 64, 10.0
    ops_n = NeumannPeriodicOps(n=n, L=L)
    kx_wave = ops_n.kx_wave_symbol()[:, 0]
    kx_sq = ops_n.kx_sq_symbol()[:, 0]

    expected_kx = (jnp.pi / L) * jnp.arange(n, dtype=kx_wave.dtype)
    assert jnp.allclose(kx_wave, expected_kx, rtol=1e-12, atol=1e-12)
    assert jnp.all(kx_wave >= 0.0)
    assert jnp.allclose(kx_sq, kx_wave * kx_wave, rtol=1e-12, atol=1e-12)


def test_neumann_periodic_keeps_fft_mode_layout_in_y() -> None:
    """y-axis frequency layout remains FFT-like in mixed DCTx-FFTy operator."""
    n, L = 32, 6.0
    ops = NeumannPeriodicOps(n=n, L=L)

    kx_sq = ops.kx_sq_symbol()
    ky_sq = ops.ky_sq_symbol()
    ky_wave = ops.ky_wave_symbol()
    fft_ky = (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=L / n)

    assert kx_sq.shape == (n, n)
    assert ky_sq.shape == (n, n)
    assert ky_wave.shape == (n, n)
    assert jnp.allclose(ky_wave[0, :], fft_ky, rtol=1e-7, atol=1e-7)
