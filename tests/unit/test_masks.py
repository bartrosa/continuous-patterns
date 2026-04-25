"""Unit tests for :mod:`continuous_patterns.core.masks`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuous_patterns.core.masks import (
    MASK_BUILDERS,
    circular_cavity_masks,
    elliptic_cavity_masks,
    polygon_cavity_masks,
    rectangular_slot_cavity_masks,
    wedge_cavity_masks,
)


def _cell_xy(*, L: float, n: int, dtype: Any = jnp.float32) -> tuple[jax.Array, jax.Array]:
    dx = L / n
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
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
    dtype = cavity_small["rv"].dtype
    x, y = _cell_xy(L=float(cavity_small["L"]), n=int(cavity_small["n"]), dtype=dtype)
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


def _assert_mask_contract(m: dict, *, n: int, dtype: Any) -> None:
    for k in ("chi", "ring", "ring_accounting", "rv"):
        assert m[k].shape == (n, n)
        assert m[k].dtype == dtype
    assert float(jnp.min(m["chi"])) >= -1e-5
    assert float(jnp.max(m["chi"])) <= 1.0 + 1e-5
    assert float(jnp.max(m["ring"])) <= 1.0 + 1e-5
    ra = m["ring_accounting"]
    assert jnp.all(jnp.logical_or(jnp.isclose(ra, 0.0), jnp.isclose(ra, 1.0)))


def test_elliptic_reduces_to_circular() -> None:
    L, R, n = 10.0, 2.5, 128
    dtype = jnp.float32
    c = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0, dtype=dtype)
    e = elliptic_cavity_masks(L=L, n=n, a=R, b=R, theta=0.0, eps_scale=2.0, dtype=dtype)
    for key in ("chi", "ring", "ring_accounting", "rv"):
        assert jnp.allclose(c[key], e[key], rtol=0, atol=2e-5)
    for key in ("dx", "xc", "yc", "L", "n"):
        assert abs(float(c[key]) - float(e[key])) < 1e-9 or c[key] == e[key]


def test_polygon_converges_to_circle() -> None:
    L, R, n = 10.0, 2.5, 128
    dtype = jnp.float32
    c = circular_cavity_masks(L=L, R=R, n=n, eps_scale=2.0, dtype=dtype)
    p = polygon_cavity_masks(
        L=L, n=n, n_sides=200, R=R, theta_offset=0.0, eps_scale=2.0, dtype=dtype
    )
    assert float(jnp.mean(jnp.abs(p["chi"] - c["chi"]))) < 0.05


def test_elliptic_polygon_wedge_rect_dispatch_and_contract() -> None:
    dtype = jnp.float32
    n = 64
    el = elliptic_cavity_masks(L=100.0, n=n, a=20.0, b=12.0, theta=0.3, dtype=dtype)
    _assert_mask_contract(el, n=n, dtype=dtype)
    assert abs(float(el["R"]) - float(jnp.sqrt(20.0 * 12.0))) < 1e-5

    poly = polygon_cavity_masks(
        L=100.0, n=n, n_sides=6, R=22.0, theta_offset=0.0, eps_scale=2.0, dtype=dtype
    )
    _assert_mask_contract(poly, n=n, dtype=dtype)
    n_s, R_c = 6, 22.0
    area_hex = 0.5 * n_s * R_c**2 * np.sin(2 * np.pi / n_s)
    R_exp = float(np.sqrt(area_hex / np.pi))
    assert abs(float(poly["R"]) - R_exp) < 0.5

    wedge = wedge_cavity_masks(
        L=200.0,
        n=n,
        R_inner=5.0,
        R_outer=40.0,
        opening_angle=float(2 * jnp.pi / 3),
        theta_center=float(jnp.pi / 2),
        dtype=dtype,
    )
    _assert_mask_contract(wedge, n=n, dtype=dtype)
    area_w = 0.5 * float(2 * jnp.pi / 3) * (40.0**2 - 5.0**2)
    assert abs(float(wedge["R"]) - float(jnp.sqrt(area_w / jnp.pi))) < 1e-3

    slot = rectangular_slot_cavity_masks(
        L=200.0, n=n, width=60.0, height=20.0, theta=0.0, dtype=dtype
    )
    _assert_mask_contract(slot, n=n, dtype=dtype)
    R_slot = float(jnp.sqrt(60.0 * 20.0 / jnp.pi))
    assert abs(float(slot["R"]) - R_slot) < 1e-5


def test_rectangular_slot_alignment_symmetry() -> None:
    m = rectangular_slot_cavity_masks(L=100.0, n=128, width=40.0, height=16.0, theta=0.0)
    chi = m["chi"]
    xc = int(m["n"]) // 2
    assert jnp.allclose(chi[:, xc], chi[::-1, xc], atol=1e-5)
    assert jnp.allclose(chi[xc, :], chi[xc, ::-1], atol=1e-5)


def test_wedge_full_opening_approximates_annulus() -> None:
    """Large opening + tiny inner radius → χ similar to circular annulus away from centre."""
    L, n = 100.0, 128
    R_outer = 35.0
    R_inner = 0.01 * R_outer
    dtype = jnp.float32
    w = wedge_cavity_masks(
        L=L,
        n=n,
        R_inner=R_inner,
        R_outer=R_outer,
        opening_angle=float(2 * jnp.pi),
        theta_center=0.0,
        dtype=dtype,
    )
    c = circular_cavity_masks(L=L, R=R_outer, n=n, dtype=dtype)
    rv = w["rv"]
    mask = rv > 5.0 * float(w["dx"])
    assert float(jnp.mean(jnp.abs(w["chi"][mask] - c["chi"][mask]))) < 0.12


def test_polygon_explicit_vertices_square() -> None:
    L = 100.0
    n = 64
    verts = [[30.0, 30.0], [70.0, 30.0], [70.0, 70.0], [30.0, 70.0]]
    m = polygon_cavity_masks(L=L, n=n, vertices=verts, eps_scale=2.0)
    _assert_mask_contract(m, n=n, dtype=jnp.float32)
    assert float(m["chi"][32, 32]) > 0.85


def test_polygon_mutually_exclusive_raises() -> None:
    verts = [[0.1, 0.1], [1, 0.1], [1, 1]]
    with pytest.raises(ValueError, match="exactly one"):
        polygon_cavity_masks(L=10.0, n=16, n_sides=6, R=2.0, vertices=verts)


def test_mask_builders_dispatch_new_types() -> None:
    e = MASK_BUILDERS["elliptic_cavity"](L=50.0, n=32, a=10.0, b=8.0, theta=0.1)
    assert e["chi"].shape == (32, 32)
    p = MASK_BUILDERS["polygon_cavity"](L=50.0, n=32, n_sides=5, R=10.0)
    assert p["chi"].shape == (32, 32)
