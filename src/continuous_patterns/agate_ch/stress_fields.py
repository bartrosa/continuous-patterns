"""Spatial stress tensors for Experiment 6.

Includes **Flamant** half-space point loads (two-point cavity squeeze), plus analytic
helpers: **pure shear**, **linear isotropic pressure gradient**, and **Kirsch**
hole-in-plate (effective interior via ``r_eff = max(r, R)``).

Flamant components in local ``(X, Y)`` with ``Y > 0`` into the medium:

    σ_yy = -(2P/π) · Y³ / (X² + Y²)²
    σ_xx = -(2P/π) · X²Y / (X² + Y²)²
    σ_xy = -(2P/π) · XY² / (X² + Y²)²

Near-load regularization: ``σ_ij ← σ_ij · r²/(r² + ε²)``, ``r² = X² + Y²``.
"""

from __future__ import annotations

import jax.numpy as jnp

from continuous_patterns.agate_ch.model import xy_grid


def flamant_single(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    x0: float,
    y0: float,
    force_dir: str,
    P: float,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flamant stress for one normal point load at ``(x0, y0)`` on a straight boundary.

    Args:
        X, Y: Global grid (cell-centred), shape ``(n, n)``.
        x0, y0: Load position on the cavity wall.
        force_dir: ``\"down\"`` — load on upper surface, medium below (smaller ``y``);
            ``\"up\"`` — load on lower surface, medium above (larger ``y``).
        P: Load intensity (magnitude; formulas use the same sign convention as above).
        eps: Regularization length for ``r²/(r²+ε²)``.

    Returns:
        ``(sigma_xx, sigma_yy, sigma_xy)`` on the grid.
    """
    tiny_y = jnp.float32(max(float(eps) * 1e-3, 1e-9))
    Xl = X - jnp.float32(x0)
    if force_dir == "down":
        # Half-space below the upper boundary: depth into medium is ``y0 - y``.
        Yl = jnp.maximum(jnp.float32(y0) - Y, tiny_y)
    elif force_dir == "up":
        # Half-space above the lower boundary.
        Yl = jnp.maximum(Y - jnp.float32(y0), tiny_y)
    else:
        raise ValueError(f"force_dir must be 'up' or 'down', got {force_dir!r}")

    rsq = Xl * Xl + Yl * Yl
    reg = rsq / (rsq + jnp.float32(eps) ** 2)
    tiny = jnp.float32(1e-20)
    denom = (rsq * rsq) + tiny

    two_p_over_pi = jnp.asarray((2.0 / jnp.pi) * float(P), dtype=jnp.float32)
    s_yy = -two_p_over_pi * (Yl * Yl * Yl) / denom
    s_xx = -two_p_over_pi * (Xl * Xl * Yl) / denom
    s_xy = -two_p_over_pi * (Xl * Yl * Yl) / denom

    s_xx = s_xx * reg
    s_yy = s_yy * reg
    s_xy = s_xy * reg
    return s_xx, s_yy, s_xy


def flamant_two_point(
    L: float,
    R: float,
    n: int,
    sigma_0: float,
    eps: float,
    *,
    P_unit: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two Flamant loads compressing the cavity (top + bottom on the vertical axis).

    Upper load at ``(L/2, L/2 + R)`` pushing ``down``; lower at ``(L/2, L/2 - R)``
    pushing ``up``. Superposes stresses then scales so
    ``max |σ_xx − σ_yy| = sigma_0``.

    Args:
        L: Domain side length.
        R: Cavity radius (loads sit on the vertical diameter through the centre).
        n: Grid resolution.
        sigma_0: Target amplitude for ``max |σ_xx − σ_yy|`` after normalization.
        eps: Regularization length (typically ``stress_eps_factor * dx``).
        P_unit: Magnitude used for each Flamant field before global scaling.

    Returns:
        ``(sigma_xx, sigma_yy, sigma_xy)`` arrays ``(n, n)``, float32.
    """
    X, Y = xy_grid(L, n)
    x_up = 0.5 * L
    y_up = 0.5 * L + R
    x_lo = 0.5 * L
    y_lo = 0.5 * L - R

    ax_u, ay_u, axy_u = flamant_single(X, Y, x_up, y_up, "down", P_unit, eps)
    ax_l, ay_l, axy_l = flamant_single(X, Y, x_lo, y_lo, "up", P_unit, eps)

    sxx = ax_u + ax_l
    syy = ay_u + ay_l
    sxy = axy_u + axy_l

    diff = sxx - syy
    peak = jnp.max(jnp.abs(diff))
    scale = jnp.where(
        peak > jnp.float32(1e-30),
        jnp.float32(float(sigma_0)) / peak,
        jnp.float32(0.0),
    )
    return sxx * scale, syy * scale, sxy * scale


def principal_sigma_max(sxx: jnp.ndarray, syy: jnp.ndarray, sxy: jnp.ndarray) -> jnp.ndarray:
    """Larger in-plane principal stress (2×2 symmetric tensor)."""
    tr = (sxx + syy) * jnp.float32(0.5)
    radius = jnp.sqrt(((sxx - syy) * jnp.float32(0.5)) ** 2 + sxy * sxy)
    return tr + radius


def uniform_uniaxial_field(
    L: float, n: int, sigma_0: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform uniaxial stress: ``σ_xx = +σ₀``, ``σ_yy = σ_xy = 0``.

    Sign convention: positive ``σ₀`` adds positively to effective stiffness along **x**
    (same qualitative role as larger ``κ_x`` in anisotropic-gradient runs — e.g. horizontal
    bands when ``κ_x > κ_y``). Not a literal Cauchy stress sign; ``σ₀`` is a coupling strength.
    """
    del L
    sxx = jnp.full((n, n), jnp.float32(sigma_0), dtype=jnp.float32)
    z = jnp.zeros((n, n), dtype=jnp.float32)
    return sxx, z, z


def pure_shear_field(
    L: float, n: int, sigma_0: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Uniform pure shear: ``σ_xy = σ₀``, ``σ_xx = σ_yy = 0`` (cell-centred grid)."""
    del L  # uniform field
    z = jnp.zeros((n, n), dtype=jnp.float32)
    sxy = jnp.full((n, n), jnp.float32(sigma_0), dtype=jnp.float32)
    return z, z, sxy


def pressure_gradient_field(
    L: float, n: int, sigma_0: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear isotropic pressure: ``p(y) = σ₀ (y - L/2) / (L/2)``, ``σ = -p I``.

    Uses the same cell-centred ``(x, y)`` grid as
    :func:`~continuous_patterns.agate_ch.model.xy_grid`.
    """
    _, Y = xy_grid(L, n)
    half_l = jnp.float32(L / 2.0)
    p = jnp.float32(sigma_0) * (Y - half_l) / jnp.maximum(half_l, jnp.float32(1e-12))
    sxx = -p
    syy = -p
    sxy = jnp.zeros((n, n), dtype=jnp.float32)
    return sxx, syy, sxy


def kirsch_field(
    L: float, R_cav: float, n: int, sigma_0: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Kirsch classical hole-in-plate stress (far-field compression ``σ₀`` along **x**).

    Uses polar ``(r, θ)`` about the cavity centre. For ``r < R_cav`` the analytic
    Kirsch formulas are ill-posed at the origin; we evaluate the tensor with
    ``r_eff = max(r, R_cav)`` so interior points see the **rim-equivalent**
    stress (effective field for coupling to φ in the cavity).

    Convention: ``σ₀ > 0`` compressive far field in **x** (same sign family as Flamant tests).
    """
    X, Y = xy_grid(L, n)
    xc = jnp.float32(L / 2.0)
    yc = jnp.float32(L / 2.0)
    dx_c = X - xc
    dy_c = Y - yc
    r = jnp.sqrt(dx_c * dx_c + dy_c * dy_c)
    theta = jnp.arctan2(dy_c, dx_c)
    # Interior of hole: use rim radius for traction-free field evaluation.
    r_eff = jnp.maximum(r, jnp.float32(R_cav))
    R = jnp.float32(R_cav)
    sig0 = jnp.float32(sigma_0)

    ratio2 = (R / r_eff) ** 2
    ratio4 = ratio2 * ratio2
    cos2t = jnp.cos(2.0 * theta)
    sin2t = jnp.sin(2.0 * theta)

    half = jnp.float32(0.5) * sig0
    s_rr = half * ((1.0 - ratio2) + (1.0 - 4.0 * ratio2 + 3.0 * ratio4) * cos2t)
    s_tt = half * ((1.0 + ratio2) - (1.0 + 3.0 * ratio4) * cos2t)
    s_rt = -half * (1.0 + 2.0 * ratio2 - 3.0 * ratio4) * sin2t

    c = jnp.cos(theta)
    s = jnp.sin(theta)
    sxx = s_rr * c * c + s_tt * s * s - 2.0 * s_rt * s * c
    syy = s_rr * s * s + s_tt * c * c + 2.0 * s_rt * s * c
    sxy = (s_rr - s_tt) * s * c + s_rt * (c * c - s * s)
    return sxx.astype(jnp.float32), syy.astype(jnp.float32), sxy.astype(jnp.float32)
