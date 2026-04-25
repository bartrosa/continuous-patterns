"""Shared JAX time-stepping helpers for model drivers (internal)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from continuous_patterns.core.imex import Geometry, SimParams, imex_step


def make_chunk_runner(
    geom: Geometry, prm: SimParams, dt: float
) -> Callable[
    [tuple[Array, Array, Array, Array, Array], int],
    tuple[tuple[Array, Array, Array, Array, Array], Array],
]:
    """Build a JIT-compiled chunk integrator for :func:`imex_step`.

    Parameters
    ----------
    geom, prm
        Closed over by the returned callable (fixed for the chunk).
    dt
        Scalar time step passed to each ``imex_step``.

    Returns
    -------
    Callable
        ``(state, n_steps) -> (state, total_injection)`` with ``n_steps`` static
        for XLA. ``total_injection`` sums per-step rim Dirichlet mass injection.
    """

    @partial(jax.jit, static_argnames=("n_steps",))
    def run_chunk(
        state: tuple[Array, Array, Array, Array, Array], n_steps: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        def body(
            _i: int,
            carry: tuple[tuple[Array, Array, Array, Array, Array], Array],
        ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
            s, inj_total = carry
            new_s, (_delta, inj_step) = imex_step(s, geom, prm, float(dt))
            return (new_s, inj_total + inj_step)

        zero_inj = jnp.asarray(0.0, dtype=state[0].dtype)
        return jax.lax.fori_loop(0, n_steps, body, (state, zero_inj))

    return run_chunk
