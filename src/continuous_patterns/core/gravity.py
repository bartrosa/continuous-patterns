"""Gravity effects: rim Dirichlet ramp and body-force advection (see Package 4 spec)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike


def _cell_grid_y(*, L: float, n: int, dtype: DTypeLike) -> tuple[Array, Array, float]:
    dx = L / n
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    return x, y, float(dx)


def rim_ramp_field(
    *,
    L: float,
    n: int,
    c0: float,
    rim_alpha: float,
    dtype: DTypeLike = jnp.float32,
) -> Array:
    """Linear vertical ramp for rim Dirichlet value ``c_rim(y)``.

    ``c_rim(y) = c0 + rim_alpha * c0 * (y - L/2) / (L/2)`` so ``rim_alpha=0`` is uniform ``c0``,
    ``rim_alpha=+1`` doubles ``c0`` at ``y=L`` and vanishes adjustment at mid-height.
    """
    _, y, _ = _cell_grid_y(L=L, n=n, dtype=dtype)
    half = 0.5 * L
    return (
        jnp.asarray(c0, dtype=dtype)
        + jnp.asarray(float(rim_alpha) * float(c0), dtype=dtype) * (y - half) / half
    )


def body_force_potential(
    *,
    L: float,
    n: int,
    g_value: float,
    dtype: DTypeLike = jnp.float32,
) -> Array:
    """Linear ``μ_grav(y) = g_value · (y - L/2)`` on the cell-centred grid (Package 4)."""
    _, y, _ = _cell_grid_y(L=L, n=n, dtype=dtype)
    gv = jnp.asarray(float(g_value), dtype=dtype)
    return gv * (y - jnp.asarray(0.5 * L, dtype=dtype))


def body_force_advection_y(
    field_hat: Array,
    ky_wave: Array,
) -> Array:
    """Return ``∂_y u`` via pseudospectral ``real(ifft2(i * ky * u_hat))``."""
    dy_hat = 1j * ky_wave * field_hat
    return jnp.real(jnp.fft.ifft2(dy_hat))
