"""Shared grid and SDF helpers for cavity mask builders (internal)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike


def cell_centered_xy(
    *, L: float, n: int, dtype: DTypeLike
) -> tuple[Array, Array, float, float, float]:
    """Cell-centred periodic grid matching :mod:`continuous_patterns.core.spectral`."""
    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    return x, y, float(dx), float(xc), float(yc)


def eps_transition(*, dx: float, eps_scale: float, dtype: DTypeLike) -> Array:
    """``max(eps_scale * dx, dx)`` as a JAX scalar matching ``circular_cavity_masks``."""
    v = float(eps_scale) * dx
    return jnp.maximum(jnp.asarray(v, dtype=dtype), jnp.asarray(dx, dtype=dtype))


def angle_wrap_pi(d: Array) -> Array:
    """Wrap angle difference to ``(-π, π]``."""
    return (d + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def point_to_segment_distance_sq(
    px: Array, py: Array, ax: Array, ay: Array, bx: Array, by: Array
) -> Array:
    """Squared distance from points ``(px, py)`` to segment ``AB`` (broadcasting)."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby + 1e-30
    t = jnp.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    qx = ax + t * abx
    qy = ay + t * aby
    return (px - qx) ** 2 + (py - qy) ** 2


def point_in_polygon_crossings(px: Array, py: Array, vx: Array, vy: Array) -> Array:
    """Ray-cast parity: count edge crossings; odd means inside (vectorized over ``K`` edges).

    Parameters
    ----------
    px, py
        Shape ``(...)`` (typically ``(n, n)``).
    vx, vy
        Closed polygon vertices ``(x_k, y_k)``, shape ``(K,)`` each (``K ≥ 3``).
    """
    xi, yi = vx, vy
    xj, yj = jnp.roll(vx, -1), jnp.roll(vy, -1)
    px_ = px[..., None]
    py_ = py[..., None]
    yi_ = yi[None, :]
    yj_ = yj[None, :]
    xi_ = xi[None, :]
    xj_ = xj[None, :]
    denom = (yj_ - yi_) + jnp.asarray(1e-20, dtype=px.dtype)
    intersect = ((yi_ > py_) != (yj_ > py_)) & (px_ < (xj_ - xi_) * (py_ - yi_) / denom + xi_)
    return jnp.sum(intersect.astype(jnp.int32), axis=-1) % 2 == 1


def batch_min_dist_sq_to_segments(px: Array, py: Array, vx: Array, vy: Array) -> Array:
    """Minimum squared distance from each point ``(px, py)`` to any polygon edge."""
    xi, yi = vx, vy
    xj, yj = jnp.roll(vx, -1), jnp.roll(vy, -1)
    return jnp.min(
        point_to_segment_distance_sq(
            px[..., None],
            py[..., None],
            xi[None, None, :],
            yi[None, None, :],
            xj[None, None, :],
            yj[None, None, :],
        ),
        axis=-1,
    )
