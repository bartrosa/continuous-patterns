"""Unit tests for :mod:`continuous_patterns.core.masks`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.core.masks import (
    MASK_BUILDERS,
    circular_cavity_masks,
)


def _cell_xy(*, L: float, n: int) -> tuple[jax.Array, jax.Array]:
    dx = L / n
    ii = jnp.arange(n, dtype=jnp.float64)[:, None]
    jj = jnp.arange(n, dtype=jnp.float64)[None, :]
    x = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    return x, y


@pytest.fixture
def cavity_small() -> dict:
    return circular_cavity_masks(L=10.0, R=2.5, n=64, eps_scale=2.0)


def test_chi_inside_outside_transition(cavity_small: dict) -> None:
    chi = cavity_small["chi"]
    rv = cavity_small["rv"]
    R = float(cavity_small["R"])
    dx = float(cavity_small["dx"])
    margin = 6.0 * dx
    inside = rv < (R - margin)
    outside = rv > (R + margin)
    assert float(jnp.min(chi[inside])) > 0.92
    assert float(jnp.max(chi[outside])) < 0.08


def test_ring_peaks_near_R_bounded_width(cavity_small: dict) -> None:
    ring = cavity_small["ring"]
    rv = cavity_small["rv"]
    R = float(cavity_small["R"])
    dx = float(cavity_small["dx"])
    ij = jnp.unravel_index(jnp.argmax(ring), ring.shape)
    r_peak = float(rv[ij])
    assert abs(r_peak - R) < 1.5 * dx
    thr = 0.05 * float(jnp.max(ring))
    mask = ring > thr
    r_lo = float(jnp.min(jnp.where(mask, rv, jnp.inf)))
    r_hi = float(jnp.max(jnp.where(mask, rv, -jnp.inf)))
    assert r_hi - r_lo < 16.0 * dx


def test_ring_accounting_annulus(cavity_small: dict) -> None:
    acc = cavity_small["ring_accounting"]
    rv = cavity_small["rv"]
    R = float(cavity_small["R"])
    dx = float(cavity_small["dx"])
    active = acc > 0.5
    assert jnp.all(rv[active] >= R - 2.0 * dx - 1e-10)
    assert jnp.all(rv[active] < R + 1e-10)
    far_inside = rv < (R - 2.5 * dx)
    far_outside = rv > (R + 2.0 * dx)
    assert float(jnp.max(acc[far_inside])) == 0.0
    assert float(jnp.max(acc[far_outside])) == 0.0


def test_rv_matches_euclidean(cavity_small: dict) -> None:
    x, y = _cell_xy(L=float(cavity_small["L"]), n=int(cavity_small["n"]))
    xc = float(cavity_small["xc"])
    yc = float(cavity_small["yc"])
    rv = cavity_small["rv"]
    expected = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    assert jnp.allclose(rv, expected, rtol=1e-12, atol=1e-12)


def test_mask_builders_circular_matches_direct() -> None:
    kwargs = {"L": 12.0, "R": 3.0, "n": 48, "eps_scale": 2.0}
    d0 = circular_cavity_masks(**kwargs)
    d1 = MASK_BUILDERS["circular_cavity"](**kwargs)
    for key in ("chi", "ring", "ring_accounting", "rv"):
        assert jnp.allclose(d0[key], d1[key], rtol=0, atol=0)
    for key in ("dx", "xc", "yc", "R", "L", "n"):
        assert d0[key] == d1[key]


def test_mask_builders_elliptic_raises() -> None:
    with pytest.raises(NotImplementedError):
        MASK_BUILDERS["elliptic_cavity"](L=10.0, R=2.0, n=16)
