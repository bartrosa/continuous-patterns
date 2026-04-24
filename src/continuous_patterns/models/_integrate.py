"""Shared JAX time-stepping helpers for model drivers (internal)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
from jax import Array

from continuous_patterns.core.imex import Geometry, SimParams, imex_step


def make_chunk_runner(
    geom: Geometry, prm: SimParams, dt: float
) -> Callable[[tuple[Array, Array, Array], int], tuple[Array, Array, Array]]:
    """JIT ``fori_loop`` over ``n_steps`` calls to :func:`imex_step`."""

    @partial(jax.jit, static_argnames=("n_steps",))
    def run_chunk(state: tuple[Array, Array, Array], n_steps: int) -> tuple[Array, Array, Array]:
        def body(_i: int, s: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            ns, _ = imex_step(s, geom, prm, float(dt))
            return ns

        return jax.lax.fori_loop(0, n_steps, body, state)

    return run_chunk
