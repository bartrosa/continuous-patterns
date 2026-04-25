"""Cavity and domain mask builders for agate simulations.

Builds χ (cavity indicator), rim enforcement masks, and accounting annuli on a
cell-centred periodic grid. Dispatch via ``MASK_BUILDERS`` (see
``docs/ARCHITECTURE.md`` §3.2). Stage II uses bulk masks (e.g. χ ≡ 1).

Formulas follow ``docs/PHYSICS.md`` §6.1: tanh transition for χ, a Gaussian-like
rim mask centred at ``r ≈ R``, and a hard annulus ``R - 2Δx ≤ r < R`` for
``ring_accounting``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from continuous_patterns.core._geometry_helpers import (
    angle_wrap_pi,
    batch_min_dist_sq_to_segments,
    cell_centered_xy,
    eps_transition,
    point_in_polygon_crossings,
)


def circular_cavity_masks(
    *,
    L: float,
    R: float,
    n: int,
    eps_scale: float = 2.0,
    dtype: DTypeLike = jnp.float32,
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
    dtype
        JAX array dtype for mask grids (default ``float32``; use ``float64`` when
        ``jax_enable_x64`` / ``precision: float64`` is active).

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


def _domain_margin_ok(*, L: float, n: int, max_extent: float) -> bool:
    dx = L / n
    return max_extent + 2.0 * dx < 0.5 * L - 1e-9


def elliptic_cavity_masks(
    *,
    L: float,
    n: int,
    a: float,
    b: float,
    theta: float = 0.0,
    eps_scale: float = 2.0,
    dtype: DTypeLike = jnp.float32,
) -> dict[str, Array | float | int]:
    """Smooth elliptic cavity centred at ``(L/2, L/2)`` with semi-axes ``a``, ``b``.

    Rotated coordinates ``(x', y')`` use ``theta`` (radians). The pseudo-radius
    ``r_eff = sqrt(F) * sqrt(a*b)`` with ``F = (x'/a)² + (y'/b)²`` equals the
    effective radius ``R = sqrt(a*b)`` on the ellipse boundary, matching the
    circular case when ``a = b = R``.

    The rim mask peaks where ``r_eff ≈ R``. Because curvature varies around the
    ellipse, the physical width of the Gaussian rim is **anisotropic** (narrower
    near high-curvature tips), which is geologically reasonable.

    ``rv`` is the usual **Euclidean** distance from the domain centre (for radial
    diagnostics); it is **not** ``r_eff``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if a <= 0 or b <= 0:
        raise ValueError(f"semi-axes a, b must be > 0, got a={a}, b={b}")
    if not _domain_margin_ok(L=L, n=n, max_extent=float(max(a, b))):
        raise ValueError(
            "elliptic cavity (max semi-axis + 2·dx) must stay inside half-domain "
            f"(L={L}, n={n}, max(a,b)={max(a, b)})."
        )

    x, y, dx, xc, yc = cell_centered_xy(L=L, n=n, dtype=dtype)
    cos_t = jnp.asarray(float(np.cos(theta)), dtype=dtype)
    sin_t = jnp.asarray(float(np.sin(theta)), dtype=dtype)
    xp = (x - xc) * cos_t + (y - yc) * sin_t
    yp = -(x - xc) * sin_t + (y - yc) * cos_t
    aa = jnp.asarray(float(a), dtype=dtype)
    bb = jnp.asarray(float(b), dtype=dtype)
    F = (xp / aa) ** 2 + (yp / bb) ** 2
    geom_mean = jnp.sqrt(aa * bb)
    r_eff = jnp.sqrt(jnp.maximum(F, jnp.asarray(0.0, dtype=dtype))) * geom_mean
    R_eff = float(np.sqrt(a * b))
    R_arr = jnp.asarray(R_eff, dtype=dtype)

    rv = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    eps_chi = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    chi = 0.5 * (1.0 - jnp.tanh((r_eff - R_arr) / eps_chi))

    sigma_ring = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    ring = jnp.exp(-0.5 * ((r_eff - R_arr) / sigma_ring) ** 2)
    ring = ring / jnp.maximum(jnp.max(ring), jnp.asarray(1e-30, dtype=dtype))

    ring_accounting = jnp.where(
        (r_eff >= R_arr - 2.0 * dx) & (r_eff < R_arr),
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
        "R": float(R_eff),
        "L": float(L),
        "n": int(n),
    }


def polygon_cavity_masks(
    *,
    L: float,
    n: int,
    n_sides: int | None = None,
    R: float | None = None,
    vertices: Sequence[tuple[float, float]] | None = None,
    theta_offset: float = 0.0,
    eps_scale: float = 2.0,
    dtype: DTypeLike = jnp.float32,
) -> dict[str, Array | float | int]:
    """Smooth polygonal cavity from regular ``(n_sides, R)`` or explicit ``vertices``.

    Signed boundary distance ``d`` is negative inside, positive outside (min
    Euclidean distance to edges with ray-cast sign). ``chi = 0.5·(1 - tanh(d/ε))``.

    Parameters
    ----------
    n_sides, R
        Regular polygon mode (mutually exclusive with ``vertices``).
    vertices
        CCW ``(x, y)`` tuples in domain coordinates; mutually exclusive with
        ``(n_sides, R)``. Convexity is not validated.
    """
    reg = n_sides is not None and R is not None
    expl = vertices is not None
    if reg == expl:
        raise ValueError("polygon_cavity_masks: set exactly one of (n_sides and R) or vertices")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    margin = 2.0 * dx

    if reg:
        assert n_sides is not None and R is not None
        if n_sides < 3:
            raise ValueError(f"n_sides must be >= 3, got {n_sides}")
        if R <= 0:
            raise ValueError(f"R must be > 0, got {R}")
        if R + margin >= 0.5 * L:
            raise ValueError("regular polygon circumradius + margin must stay inside half-domain")
        angles = theta_offset + (2.0 * np.pi / float(n_sides)) * np.arange(
            n_sides, dtype=np.float64
        )
        vx_np = xc + R * np.cos(angles)
        vy_np = yc + R * np.sin(angles)
    else:
        assert vertices is not None
        if len(vertices) < 3:
            raise ValueError("vertices must have length >= 3")
        vx_np = np.array([p[0] for p in vertices], dtype=np.float64)
        vy_np = np.array([p[1] for p in vertices], dtype=np.float64)
        if (
            np.any(vx_np < margin)
            or np.any(vx_np > L - margin)
            or np.any(vy_np < margin)
            or np.any(vy_np > L - margin)
        ):
            raise ValueError("polygon vertices must lie inside [2·dx, L - 2·dx]²")

    area_np = 0.5 * np.abs(np.sum(vx_np * np.roll(vy_np, -1) - np.roll(vx_np, -1) * vy_np))
    if area_np <= 0:
        raise ValueError("degenerate polygon area")
    R_eff = float(np.sqrt(area_np / np.pi))

    vx = jnp.asarray(vx_np, dtype=dtype)
    vy = jnp.asarray(vy_np, dtype=dtype)

    x, y, dx, xc, yc = cell_centered_xy(L=L, n=n, dtype=dtype)
    d_min = jnp.sqrt(
        jnp.maximum(
            batch_min_dist_sq_to_segments(x, y, vx, vy),
            jnp.asarray(0.0, dtype=dtype),
        )
    )
    inside = point_in_polygon_crossings(x, y, vx, vy)
    signed_d = jnp.where(inside, -d_min, d_min)

    eps_chi = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    chi = 0.5 * (1.0 - jnp.tanh(signed_d / eps_chi))

    sigma_ring = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    ring = jnp.exp(-0.5 * (signed_d / sigma_ring) ** 2)
    ring = ring / jnp.maximum(jnp.max(ring), jnp.asarray(1e-30, dtype=dtype))

    ring_accounting = jnp.where(
        (signed_d >= -2.0 * dx) & (signed_d < 0.0),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    rv = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    return {
        "chi": chi,
        "ring": ring,
        "ring_accounting": ring_accounting,
        "rv": rv,
        "dx": float(dx),
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R_eff),
        "L": float(L),
        "n": int(n),
    }


def wedge_cavity_masks(
    *,
    L: float,
    n: int,
    R_inner: float,
    R_outer: float,
    opening_angle: float,
    theta_center: float = 0.0,
    eps_scale: float = 2.0,
    dtype: DTypeLike = jnp.float32,
) -> dict[str, Array | float | int]:
    """Smooth annular sector (wedge): intersection of annulus and angular sector.

    ``opening_angle`` is the full opening (radians). ``theta_center`` is the
    bisector direction. The outer circular arc carries the rim peak
    (``ring``), modelling a fluid inlet along the outer fracture arc.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 < R_inner < R_outer):
        raise ValueError(f"require 0 < R_inner < R_outer, got R_inner={R_inner}, R_outer={R_outer}")
    if not (0.0 < opening_angle <= 2.0 * np.pi + 1e-12):
        raise ValueError(f"opening_angle must be in (0, 2π], got {opening_angle}")
    dx = L / n
    if R_outer + 2.0 * dx >= 0.5 * L - 1e-9:
        raise ValueError("wedge R_outer + 2·dx must stay inside half-domain from cavity centre")

    x, y, dx, xc, yc = cell_centered_xy(L=L, n=n, dtype=dtype)
    rv = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    ang = jnp.arctan2(y - yc, x - xc)
    dtheta = jnp.abs(angle_wrap_pi(ang - jnp.asarray(float(theta_center), dtype=dtype)))
    half_open = jnp.asarray(0.5 * float(opening_angle), dtype=dtype)

    eps_r = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    eps_ang = jnp.maximum(
        float(eps_scale) * dx / jnp.maximum(R_outer, jnp.asarray(1e-6, dtype=dtype)),
        jnp.asarray(float(eps_scale) * dx, dtype=dtype),
    )

    chi_outer = 0.5 * (1.0 - jnp.tanh((rv - jnp.asarray(float(R_outer), dtype=dtype)) / eps_r))
    chi_inner = 0.5 * (1.0 + jnp.tanh((rv - jnp.asarray(float(R_inner), dtype=dtype)) / eps_r))
    chi_angular = 0.5 * (1.0 - jnp.tanh((dtheta - half_open) / eps_ang))
    chi = chi_outer * chi_inner * chi_angular

    sigma_ring = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    ring_raw = jnp.exp(-0.5 * ((rv - jnp.asarray(float(R_outer), dtype=dtype)) / sigma_ring) ** 2)
    ring_raw = ring_raw * chi_inner * chi_angular
    ring = ring_raw / jnp.maximum(jnp.max(ring_raw), jnp.asarray(1e-30, dtype=dtype))

    ang_margin = 2.0 * dx / jnp.maximum(R_outer, jnp.asarray(1e-6, dtype=dtype))
    ring_accounting = jnp.where(
        (rv >= jnp.asarray(float(R_outer), dtype=dtype) - 2.0 * dx)
        & (rv < jnp.asarray(float(R_outer), dtype=dtype))
        & (dtheta <= half_open - ang_margin),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )

    area = 0.5 * float(opening_angle) * (R_outer**2 - R_inner**2)
    R_eff = float(np.sqrt(max(area, 1e-30) / np.pi))

    return {
        "chi": chi,
        "ring": ring,
        "ring_accounting": ring_accounting,
        "rv": rv,
        "dx": float(dx),
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R_eff),
        "L": float(L),
        "n": int(n),
    }


def rectangular_slot_cavity_masks(
    *,
    L: float,
    n: int,
    width: float,
    height: float,
    theta: float = 0.0,
    eps_scale: float = 2.0,
    dtype: DTypeLike = jnp.float32,
) -> dict[str, Array | float | int]:
    """Smooth rotated rectangular cavity using signed box distance ``d``.

    Inside ``d < 0``, boundary ``d = 0``. Corners are slightly rounded in the
    smoothed ``χ``/``ring`` fields, avoiding sharp spectral hot spots.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be > 0, got width={width}, height={height}")
    dx = L / n
    half_diag = 0.5 * float(np.hypot(width, height))
    if half_diag + 2.0 * dx >= 0.5 * L - 1e-9:
        raise ValueError(
            "rotated slot half-diagonal + 2·dx must stay inside half-domain "
            f"(L={L}, n={n}, width={width}, height={height})."
        )

    x, y, dx, xc, yc = cell_centered_xy(L=L, n=n, dtype=dtype)
    cos_t = jnp.asarray(float(np.cos(theta)), dtype=dtype)
    sin_t = jnp.asarray(float(np.sin(theta)), dtype=dtype)
    xp = (x - xc) * cos_t + (y - yc) * sin_t
    yp = -(x - xc) * sin_t + (y - yc) * cos_t
    hw = jnp.asarray(0.5 * float(width), dtype=dtype)
    hh = jnp.asarray(0.5 * float(height), dtype=dtype)
    d = jnp.maximum(jnp.abs(xp) - hw, jnp.abs(yp) - hh)

    eps_chi = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    chi = 0.5 * (1.0 - jnp.tanh(d / eps_chi))

    sigma_ring = eps_transition(dx=dx, eps_scale=eps_scale, dtype=dtype)
    ring = jnp.exp(-0.5 * (d / sigma_ring) ** 2)
    ring = ring / jnp.maximum(jnp.max(ring), jnp.asarray(1e-30, dtype=dtype))

    ring_accounting = jnp.where(
        (d >= -2.0 * dx) & (d < 0.0),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    rv = jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    R_eff = float(np.sqrt(width * height / np.pi))

    return {
        "chi": chi,
        "ring": ring,
        "ring_accounting": ring_accounting,
        "rv": rv,
        "dx": float(dx),
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R_eff),
        "L": float(L),
        "n": int(n),
    }


MASK_BUILDERS: dict[str, Callable[..., dict[str, Array | float | int]]] = {
    "circular_cavity": circular_cavity_masks,
    "elliptic_cavity": elliptic_cavity_masks,
    "polygon_cavity": polygon_cavity_masks,
    "wedge_cavity": wedge_cavity_masks,
    "rectangular_slot": rectangular_slot_cavity_masks,
}
