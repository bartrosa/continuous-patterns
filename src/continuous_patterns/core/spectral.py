"""Pseudospectral operators on a periodic, cell-centred square grid.

Wavenumbers follow ``k = 2π · fftfreq(n, d=dx)`` with ``dx = L / n``. Laplacian
and gradients are applied in Fourier space; products that require dealiasing
are handled by callers (``docs/PHYSICS.md`` §8.1). JIT is applied at integrator
level, not inside these primitives (``docs/ARCHITECTURE.md`` §3.1).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def k_vectors(
    *,
    L: float,
    n: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Build spectral symbol arrays for an ``n × n`` periodic cell-centred grid.

    Parameters
    ----------
    L
        Domain side length (same in x and y).
    n
        Number of cell-centred points per axis (``n ≥ 1``).

    Returns
    -------
    tuple
        ``(k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four)`` where ``k_*_wave`` are
        angular wavenumbers (rad / length), ``k_sq = kx_wave² + ky_wave²``,
        ``kx_sq = kx_wave²``, ``ky_sq = ky_wave²``, ``k_four = k_sq²``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    dx = L / n
    fx = jnp.fft.fftfreq(n, d=dx)
    k1d = (2.0 * jnp.pi) * fx
    kx_wave, ky_wave = jnp.broadcast_arrays(k1d[:, jnp.newaxis], k1d[jnp.newaxis, :])
    kx_sq = kx_wave * kx_wave
    ky_sq = ky_wave * ky_wave
    k_sq = kx_sq + ky_sq
    k_four = k_sq * k_sq
    return k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four


def laplacian_real(u: ArrayLike, k_sq: ArrayLike) -> Array:
    """Spatial Laplacian ``∇² u`` via pseudospectral FFT, real output.

    Parameters
    ----------
    u
        Real scalar field on the grid, shape ``(n, n)``.
    k_sq
        Non-negative symbol ``kx² + ky²`` from :func:`k_vectors`. The operator is
        ``real(ifft2(-k_sq * fft2(u)))``, i.e. Fourier multiplication by ``-|k|²``.

    Returns
    -------
    Array
        Real ``∇² u``, same shape as ``u``.
    """
    u_arr = jnp.asarray(u)
    k_arr = jnp.asarray(k_sq)
    return jnp.real(jnp.fft.ifft2(-k_arr * jnp.fft.fft2(u_arr)))


def grad_real(
    u: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
) -> tuple[Array, Array]:
    """Gradient ``(∂_x u, ∂_y u)`` pseudospectral, real outputs.

    Parameters
    ----------
    u
        Real scalar field, shape ``(n, n)``.
    kx_wave, ky_wave
        Angular wavenumber grids from :func:`k_vectors`, broadcastable to FFT
        of ``u``.

    Returns
    -------
    tuple[Array, Array]
        ``(∂_x u, ∂_y u)`` as real arrays.
    """
    u_arr = jnp.asarray(u)
    u_hat = jnp.fft.fft2(u_arr)
    kx = jnp.asarray(kx_wave)
    ky = jnp.asarray(ky_wave)
    gx = jnp.real(jnp.fft.ifft2(1j * kx * u_hat))
    gy = jnp.real(jnp.fft.ifft2(1j * ky * u_hat))
    return gx, gy


def divergence_real(
    ux: ArrayLike,
    uy: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
) -> Array:
    """Divergence ``∂_x u_x + ∂_y u_y`` pseudospectral, real output.

    Parameters
    ----------
    ux, uy
        Real vector components on the grid, shape ``(n, n)``.
    kx_wave, ky_wave
        Angular wavenumber grids from :func:`k_vectors`.

    Returns
    -------
    Array
        Real divergence field.
    """
    ux_arr = jnp.asarray(ux)
    uy_arr = jnp.asarray(uy)
    kx = jnp.asarray(kx_wave)
    ky = jnp.asarray(ky_wave)
    ux_hat = jnp.fft.fft2(ux_arr)
    uy_hat = jnp.fft.fft2(uy_arr)
    div_hat = 1j * kx * ux_hat + 1j * ky * uy_hat
    return jnp.real(jnp.fft.ifft2(div_hat))
