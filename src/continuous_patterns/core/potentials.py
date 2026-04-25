"""Bulk Cahn–Hilliard potential derivatives and outer barrier (JAX).

Each **builder** in :data:`POTENTIAL_BUILDERS` implements ``∂f/∂φ`` for a local
bulk free-energy density (derivative only; the energy itself is not needed in
IMEX). Builders take ``phi`` as the first positional argument and model
parameters as keyword arguments; return value matches ``phi`` shape and dtype.

The **outer barrier** derivative is separate: :func:`barrier_prime` is always
added on top of the bulk term in the IMEX step (not dispatched). See
``docs/PHYSICS.md`` §4.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Bulk potentials — ∂f/∂φ
# ---------------------------------------------------------------------------


def double_well_prime(phi: Array, *, W: float) -> Array:
    """Derivative of the symmetric quartic double well ``W φ² (1-φ)²``.

    Returns ``2 W φ (1-φ) (1-2φ)``.

    Parameters
    ----------
    phi
        Phase field on the grid.
    W
        Barrier height between wells.

    Returns
    -------
    jax.Array
        Same shape and dtype as ``phi``.
    """
    return 2.0 * W * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def tilted_well_prime(phi: Array, *, W: float, tilt: float) -> Array:
    """Double-well derivative plus a constant tilt in ``μ``.

    Returns ``double_well_prime(phi, W=W) + tilt``.

    Parameters
    ----------
    phi
        Phase field.
    W
        Double-well strength.
    tilt
        Constant bias (linear tilt in the free energy).

    Returns
    -------
    jax.Array
        Same shape and dtype as ``phi``.
    """
    return double_well_prime(phi, W=W) + jnp.asarray(tilt, dtype=phi.dtype)


def asymmetric_well_prime(phi: Array, *, W: float, phi_left: float, phi_right: float) -> Array:
    """Double-well style quartic with minima at ``phi_left`` and ``phi_right``.

    Returns ``2W (φ - φ_l) (φ - φ_r) (2φ - φ_l - φ_r)``.

    Parameters
    ----------
    phi
        Phase field.
    W
        Energy scale.
    phi_left, phi_right
        Locations of the two wells.

    Returns
    -------
    jax.Array
        Same shape and dtype as ``phi``.
    """
    pl = jnp.asarray(phi_left, dtype=phi.dtype)
    pr = jnp.asarray(phi_right, dtype=phi.dtype)
    w = jnp.asarray(W, dtype=phi.dtype)
    return 2.0 * w * (phi - pl) * (phi - pr) * (2.0 * phi - pl - pr)


def zero_potential(phi: Array) -> Array:
    """Zero bulk driving force (inactive phase).

    Parameters
    ----------
    phi
        Phase field (used only for shape/dtype).

    Returns
    -------
    jax.Array
        Zeros matching ``phi``.
    """
    return jnp.zeros_like(phi)


def barrier_prime(phi: Array, *, lambda_bar: float) -> Array:
    """Derivative of the outer barrier penalizing ``φ < 0`` and ``φ > 1``.

    Matches the legacy ``_barrier_prime`` implementation in ``core/imex`` before
    refactor: quadratic penalty outside ``[0, 1]``.

    Parameters
    ----------
    phi
        Phase field.
    lambda_bar
        Barrier stiffness ``λ̄``.

    Returns
    -------
    jax.Array
        Same shape and dtype as ``phi``.
    """
    lam = jnp.asarray(lambda_bar, dtype=phi.dtype)
    neg_excess = jnp.maximum(-phi, 0.0)
    pos_excess = jnp.maximum(phi - 1.0, 0.0)
    return -2.0 * lam * neg_excess + 2.0 * lam * pos_excess


POTENTIAL_BUILDERS: dict[str, Callable[..., Array]] = {
    "double_well": double_well_prime,
    "tilted_well": tilted_well_prime,
    "asymmetric_well": asymmetric_well_prime,
    "zero": zero_potential,
}
