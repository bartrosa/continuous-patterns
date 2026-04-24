"""Cavity and domain mask builders for agate simulations.

Builds χ (cavity indicator), rim enforcement masks, and accounting annuli on a
cell-centred periodic grid. Dispatch via ``MASK_BUILDERS`` (see
``docs/ARCHITECTURE.md`` §3.2). Stage II uses bulk masks (e.g. χ ≡ 1).

Formulas follow ``docs/PHYSICS.md`` §6.1: tanh transition for χ, a Gaussian-like
rim mask centred at ``r ≈ R``, and a hard annulus ``R - 2Δx ≤ r < R`` for
``ring_accounting``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array


def circular_cavity_masks(
    *,
    L: float,
    R: float,
    n: int,
    eps_scale: float = 2.0,
) -> dict[str, Array | float | int]:
    """Build smooth circular cavity masks on an ``n × n`` cell-centred grid.

    Grid layout matches :func:`continuous_patterns.core.spectral.k_vectors`
    conventions used in tests: row index ``i`` carries ``x = (i + ½)Δx`` and
    column index ``j`` carries ``y = (j + ½)Δx``, with ``Δx = L / n`` and domain
    ``[0, L)²``. Cavity centre is ``(xc, yc) = (L/2, L/2)``.

    Parameters
    ----------
    L
        Domain side length.
    R
        Cavity nominal radius (``R > 0``).
    n
        Grid size per axis.
    eps_scale
        ``ε_χ`` in PHYSICS: transition width scales as ``ε_χ Δx`` (default ``2``).

    Returns
    -------
    dict
        Arrays ``chi``, ``ring``, ``ring_accounting``, ``rv`` and scalars
        ``dx``, ``xc``, ``yc``, ``R``, ``L``, ``n``.

    Raises
    ------
    ValueError
        If ``R <= 0`` or ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R}")

    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    dtype = jnp.float64

    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x_cent = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y_cent = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    rv = jnp.sqrt((x_cent - xc) ** 2 + (y_cent - yc) ** 2)

    eps_chi = float(eps_scale) * dx
    eps_chi = jnp.maximum(jnp.asarray(eps_chi, dtype=dtype), jnp.asarray(dx, dtype=dtype))
    chi = 0.5 * (1.0 - jnp.tanh((rv - R) / eps_chi))

    sigma_ring = float(eps_scale) * dx
    sigma_ring = jnp.maximum(jnp.asarray(sigma_ring, dtype=dtype), jnp.asarray(dx, dtype=dtype))
    ring = jnp.exp(-0.5 * ((rv - R) / sigma_ring) ** 2)
    ring = ring / jnp.maximum(jnp.max(ring), jnp.asarray(1e-30, dtype=dtype))

    ring_accounting = jnp.where(
        (rv >= R - 2.0 * dx) & (rv < R),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )

    return {
        "chi": chi,
        "ring": ring,
        "ring_accounting": ring_accounting,
        "rv": rv,
        "dx": float(dx),
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R),
        "L": float(L),
        "n": int(n),
    }


def elliptic_cavity_masks(*args: Any, **kwargs: Any) -> dict[str, Array | float | int]:
    """Reserved extension for non-circular cavities (not implemented)."""
    raise NotImplementedError("Elliptic cavity masks are not implemented yet.")


MASK_BUILDERS: dict[str, Callable[..., dict[str, Array | float | int]]] = {
    "circular_cavity": circular_cavity_masks,
    "elliptic_cavity": elliptic_cavity_masks,
}
