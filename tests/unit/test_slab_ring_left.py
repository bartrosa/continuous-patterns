"""Unit tests for slab left-wall mask."""

from __future__ import annotations

import jax.numpy as jnp

from continuous_patterns.core.masks import slab_masks


def test_slab_ring_left_first_rows() -> None:
    m = slab_masks(
        L=100.0, n=64, rim_width_px=2, rim_left_width_px=3, eps_scale=2.0, dtype=jnp.float32
    )
    ring_left = m["ring_left"]
    assert ring_left.shape == (64, 64)
    assert float(jnp.mean(ring_left[:3, :])) == 1.0
    assert float(jnp.max(ring_left[3:, :])) == 0.0
