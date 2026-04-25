"""Prescribed Cauchy stress tensors and ψ-split stress chemical potential.

Field builders return ``(sigma_xx, sigma_yy, sigma_xy)`` on the cell-centred
``n × n`` grid (same ``(x, y)`` layout as :mod:`continuous_patterns.core.masks`).
Dispatch via ``STRESS_BUILDERS``. Coupling follows ``docs/PHYSICS.md`` §5–7 and
``docs/ARCHITECTURE.md`` §3.3: :math:`\\mu_{\\mathrm{stress}} = -B\\,\\nabla\\cdot
(\\sigma\\nabla\\psi)` with pseudospectral ``grad_real`` /
``divergence_real`` from :mod:`continuous_patterns.core.spectral`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from continuous_patterns.core.spectral import divergence_real, grad_real

# ---------------------------------------------------------------------------
# Grid helper (matches masks.py / spectral tests: x on rows, y on columns)
# ---------------------------------------------------------------------------


def _cell_grid(*, L: float, n: int, dtype: DTypeLike) -> tuple[Array, Array, float]:
    dx = L / n
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    return x, y, float(dx)


def _zeros(*, n: int, dtype: DTypeLike) -> tuple[Array, Array, Array]:
    z = jnp.zeros((n, n), dtype=dtype)
    return z, z, z


# ---------------------------------------------------------------------------
# Prescribed stress builders (PHYSICS §7)
# ---------------------------------------------------------------------------


def none(
    *, L: float, n: int, dtype: DTypeLike = jnp.float32, **_: Any
) -> tuple[Array, Array, Array]:
    """All-zero stress tensor (explicit ``none`` mode)."""
    _ = L
    return _zeros(n=n, dtype=dtype)


def uniform_uniaxial(
    *, L: float, n: int, sigma_0: float, dtype: DTypeLike = jnp.float32
) -> tuple[Array, Array, Array]:
    """``σ_xx = σ₀``, ``σ_yy = σ_xy = 0`` (PHYSICS §7.1)."""
    _ = L  # domain length; kept for a uniform builder keyword API across modes
    z = jnp.zeros((n, n), dtype=dtype)
    sxx = jnp.full((n, n), float(sigma_0), dtype=dtype)
    return sxx, z, z


def uniform_biaxial(
    *, L: float, n: int, sigma_0: float, dtype: DTypeLike = jnp.float32
) -> tuple[Array, Array, Array]:
    """``σ_xx = σ_yy = σ₀``, ``σ_xy = 0`` (PHYSICS §7.2)."""
    _ = L
    z = jnp.zeros((n, n), dtype=dtype)
    s = jnp.full((n, n), float(sigma_0), dtype=dtype)
    return s, s, z


def pure_shear(
    *, L: float, n: int, sigma_0: float, dtype: DTypeLike = jnp.float32
) -> tuple[Array, Array, Array]:
    """``σ_xy = σ₀``, ``σ_xx = σ_yy = 0`` (PHYSICS §7.3)."""
    _ = L
    z = jnp.zeros((n, n), dtype=dtype)
    sxy = jnp.full((n, n), float(sigma_0), dtype=dtype)
    return z, z, sxy


def _flamant_weighted(h: Array, d: Array, P: float, eps: float) -> tuple[Array, Array, Array]:
    """Single Flamant half-space contribution (PHYSICS §7.4), Cartesian regularization."""
    r2 = h * h + d * d
    den = r2 * (r2 + eps * eps) + 1e-30
    fac = (-2.0 * float(P)) / jnp.pi
    sxx = fac * h * h * d / den
    syy = fac * d * d * d / den
    sxy = fac * h * d * d / den
    return sxx, syy, sxy


def flamant_two_point(
    *,
    L: float,
    R: float,
    n: int,
    sigma_0: float,
    stress_eps_factor: float = 3.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Two opposing Flamant half-space loads on the vertical diameter (PHYSICS §7.4)."""
    x, y, dx = _cell_grid(L=L, n=n, dtype=dtype)
    xc = 0.5 * L
    yc = 0.5 * L
    eps = float(stress_eps_factor) * dx
    h = x - xc
    y_upper = yc + R
    y_lower = yc - R
    d_up = y_upper - y
    d_lo = y - y_lower
    su = _flamant_weighted(h, d_up, +1.0, eps)
    sl = _flamant_weighted(h, d_lo, -1.0, eps)
    sxx = su[0] + sl[0]
    syy = su[1] + sl[1]
    sxy = su[2] + sl[2]
    dev = sxx - syy
    peak = jnp.max(jnp.abs(dev))
    dt = dev.dtype
    scale = jnp.asarray(float(sigma_0), dtype=dt) / jnp.maximum(peak, jnp.asarray(1e-30, dtype=dt))
    return scale * sxx, scale * syy, scale * sxy


def pressure_gradient(
    *, L: float, n: int, sigma_0: float, dtype: DTypeLike = jnp.float32
) -> tuple[Array, Array, Array]:
    """Isotropic linear ``y``-pressure (PHYSICS §7.5): ``p(y)=σ₀(y-L/2)/(L/2)``, ``σ=-p I``."""
    _, y, _ = _cell_grid(L=L, n=n, dtype=dtype)
    p = float(sigma_0) * (y - 0.5 * L) / (0.5 * L)
    sxx = -p
    syy = -p
    z = jnp.zeros_like(sxx)
    return sxx, syy, z


def _polar_to_cartesian(
    srr: Array, stt: Array, srt: Array, th: Array
) -> tuple[Array, Array, Array]:
    c = jnp.cos(th)
    s = jnp.sin(th)
    sxx = srr * c * c + stt * s * s - 2.0 * srt * s * c
    syy = srr * s * s + stt * c * c + 2.0 * srt * s * c
    sxy = (srr - stt) * s * c + srt * (c * c - s * s)
    return sxx, syy, sxy


def kirsch(
    *,
    L: float,
    R: float,
    n: int,
    S_xx_far: float,
    S_yy_far: float,
    S_xy_far: float = 0.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Kirsch (1898): infinite plate with traction-free circular hole under remote plane stress.

    Uses the general remote tensor ``(S_xx_far, S_yy_far, S_xy_far)`` (fracturemechanics.org
    “General 2-D Loading” polar formulas). Inside ``r < R`` the evaluation uses
    ``r_eff = max(r, R)`` to avoid the origin singularity.
    """
    if R <= 0:
        raise ValueError(f"Kirsch requires R > 0, got {R}")
    x, y, _ = _cell_grid(L=L, n=n, dtype=dtype)
    xc = 0.5 * L
    yc = 0.5 * L
    xr = x - xc
    yr = y - yc
    r = jnp.sqrt(xr * xr + yr * yr)
    r_eff = jnp.maximum(r, jnp.asarray(float(R), dtype=dtype))
    th = jnp.arctan2(yr, xr)
    t = (float(R) / r_eff) ** 2
    t2 = t * t
    srr_inf = 0.5 * (float(S_xx_far) + float(S_yy_far))
    sdiff = 0.5 * (float(S_xx_far) - float(S_yy_far))
    sxyf = float(S_xy_far)
    srr = srr_inf * (1.0 - t) + (sdiff * jnp.cos(2.0 * th) + sxyf * jnp.sin(2.0 * th)) * (
        1.0 - 4.0 * t + 3.0 * t2
    )
    stt = srr_inf * (1.0 + t) - (sdiff * jnp.cos(2.0 * th) + sxyf * jnp.sin(2.0 * th)) * (
        1.0 + 3.0 * t2
    )
    srt = ((-sdiff) * jnp.sin(2.0 * th) + sxyf * jnp.cos(2.0 * th)) * (1.0 + 2.0 * t - 3.0 * t2)
    return _polar_to_cartesian(srr, stt, srt, th)


def lithostatic(
    *,
    L: float,
    n: int,
    rho_g_dim: float,
    lateral_K: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Lithostatic-style ``σ_yy(y) = -ρg (L-y)``, ``σ_xx = lateral_K σ_yy``, ``σ_xy=0``."""
    if rho_g_dim <= 0:
        raise ValueError(f"lithostatic requires rho_g_dim > 0, got {rho_g_dim}")
    if not (0.0 < float(lateral_K) <= 1.0 + 1e-9):
        raise ValueError(f"lithostatic requires 0 < lateral_K <= 1, got {lateral_K}")
    _, y, _ = _cell_grid(L=L, n=n, dtype=dtype)
    syy = -float(rho_g_dim) * (jnp.asarray(L, dtype=dtype) - y)
    sxx = jnp.asarray(float(lateral_K), dtype=dtype) * syy
    z = jnp.zeros_like(sxx)
    return sxx, syy, z


def tectonic_far_field(
    *,
    L: float,
    n: int,
    S_H: float,
    S_h: float,
    S_V: float,
    theta_SH: float = 0.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Uniform remote Anderson-style horizontal principal stresses rotated by ``theta_SH``.

    The in-plane Cauchy tensor is ``diag(S_H, S_h)`` in principal axes, rotated into ``(x,y)``.
    ``S_V`` is accepted for YAML documentation / regime labelling only; it does **not** enter the
    2D plane-stress tensor (see spec).
    """
    _ = L
    _ = S_V
    c = jnp.cos(jnp.asarray(float(theta_SH), dtype=dtype))
    s = jnp.sin(jnp.asarray(float(theta_SH), dtype=dtype))
    sxx = jnp.full((n, n), float(S_H) * c * c + float(S_h) * s * s, dtype=dtype)
    syy = jnp.full((n, n), float(S_H) * s * s + float(S_h) * c * c, dtype=dtype)
    sxy = jnp.full((n, n), (float(S_H) - float(S_h)) * float(c) * float(s), dtype=dtype)
    return sxx, syy, sxy


def inglis(
    *,
    L: float,
    n: int,
    a: float,
    b: float,
    theta: float = 0.0,
    S_xx_far: float = 0.0,
    S_yy_far: float = 0.0,
    S_xy_far: float = 0.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Inglis (1913) / elliptic-hole stress around a traction-free ellipse.

    When ``a ≈ b`` (within a relative tolerance), delegates to :func:`kirsch` with
    ``R = (a+b)/2`` so the circular limit is exact.

    For ``a ≠ b``, uses a **circular Kirsch surrogate** with ``R = sqrt(a·b)`` in the
    ellipse-aligned frame (``# TODO``: replace with full Muskhelishvili / elliptic-coordinate
    field for a true Inglis ellipse). Remote loading uses ``(S_xx_far, S_yy_far, S_xy_far)`` in
    the **lab** frame; rotate internally by ``theta`` only when the full field is implemented.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"inglis requires a,b > 0, got a={a}, b={b}")
    aa, bb = float(a), float(b)
    _ = theta  # ellipse orientation reserved for future full Inglis field
    if abs(aa - bb) <= 1e-5 * max(aa, bb):
        return kirsch(
            L=L,
            R=0.5 * (aa + bb),
            n=n,
            S_xx_far=S_xx_far,
            S_yy_far=S_yy_far,
            S_xy_far=S_xy_far,
            dtype=dtype,
        )
    # TODO(Package3): full Inglis / Muskhelishvili σ field for a ≠ b (currently Kirsch surrogate).
    R_eff = float((aa * bb) ** 0.5)
    return kirsch(
        L=L,
        R=R_eff,
        n=n,
        S_xx_far=S_xx_far,
        S_yy_far=S_yy_far,
        S_xy_far=S_xy_far,
        dtype=dtype,
    )


def pore_pressure_field(
    *,
    L: float,
    n: int,
    field: str,
    p0: float,
    dtype: DTypeLike = jnp.float32,
) -> Array:
    """Scalar pore pressure grid for ``uniform`` or ``hydrostatic`` (linear in ``y``)."""
    _, y, _ = _cell_grid(L=L, n=n, dtype=dtype)
    if field == "uniform":
        return jnp.full((n, n), float(p0), dtype=dtype)
    if field == "hydrostatic":
        Ld = jnp.asarray(L, dtype=dtype)
        return jnp.asarray(float(p0), dtype=dtype) * (Ld - y) / Ld
    raise ValueError(f"pore_pressure.field must be 'uniform' or 'hydrostatic', got {field!r}")


def apply_pore_pressure(
    sigma_xx: Array,
    sigma_yy: Array,
    sigma_xy: Array,
    *,
    p_pore: Array | float,
    biot_alpha: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Terzaghi/Biot effective normal stresses: ``σ' = σ - α p I`` (shear unchanged)."""
    p = jnp.asarray(p_pore, dtype=sigma_xx.dtype)
    ba = jnp.asarray(float(biot_alpha), dtype=sigma_xx.dtype)
    return sigma_xx - ba * p, sigma_yy - ba * p, sigma_xy


STRESS_BUILDERS: dict[str, Callable[..., tuple[Array, Array, Array]]] = {
    "none": none,
    "uniform_uniaxial": uniform_uniaxial,
    "uniform_biaxial": uniform_biaxial,
    "pure_shear": pure_shear,
    "flamant_two_point": flamant_two_point,
    "pressure_gradient": pressure_gradient,
    "kirsch": kirsch,
    "lithostatic": lithostatic,
    "tectonic_far_field": tectonic_far_field,
    "inglis": inglis,
}


# ---------------------------------------------------------------------------
# ψ-split coupling (PHYSICS §5)
# ---------------------------------------------------------------------------


def _mu_stress_real(
    psi: ArrayLike,
    sigma_xx: ArrayLike,
    sigma_xy: ArrayLike,
    sigma_yy: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
    B: float | ArrayLike,
) -> Array:
    """Real-space ``μ_stress = -B ∇·(σ ∇ψ)`` (pseudospectral)."""
    gx, gy = grad_real(psi, kx_wave, ky_wave)
    sxx = jnp.asarray(sigma_xx)
    sxy = jnp.asarray(sigma_xy)
    syy = jnp.asarray(sigma_yy)
    fx = sxx * gx + sxy * gy
    fy = sxy * gx + syy * gy
    div = divergence_real(fx, fy, kx_wave, ky_wave)
    B_arr = jnp.asarray(B, dtype=div.dtype)
    return -B_arr * div


def stress_mu_hat(
    psi: ArrayLike,
    sigma_xx: ArrayLike,
    sigma_xy: ArrayLike,
    sigma_yy: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
    B: float | ArrayLike,
) -> Array:
    """``fft2(μ_stress)`` with ``μ_stress = -B ∇·(σ ∇ψ)`` (PHYSICS §5)."""
    mu = _mu_stress_real(psi, sigma_xx, sigma_xy, sigma_yy, kx_wave, ky_wave, B)
    return jnp.fft.fft2(mu)


def mu_stress_real(
    psi: ArrayLike,
    sigma_xx: ArrayLike,
    sigma_xy: ArrayLike,
    sigma_yy: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
    B: float | ArrayLike,
) -> Array:
    """Real-space ``μ_stress = -B ∇·(σ ∇ψ)`` for a prescribed ``ψ`` field (PHYSICS §5)."""
    return _mu_stress_real(psi, sigma_xx, sigma_xy, sigma_yy, kx_wave, ky_wave, B)


def stress_contribution_to_mu(
    phi_m: ArrayLike,
    phi_c: ArrayLike,
    sigma_xx: ArrayLike,
    sigma_xy: ArrayLike,
    sigma_yy: ArrayLike,
    kx_wave: ArrayLike,
    ky_wave: ArrayLike,
    B: float | ArrayLike,
) -> tuple[Array, Array]:
    """Return ``(δμ_m, δμ_c)`` in real space with ``±½`` ψ-split (PHYSICS §5).

    Legacy helper: ``ψ = φ_m - φ_c``. Prefer :func:`mu_stress_real` plus per-phase
    ``½·(ψ_sign_α)·μ_stress`` when more than two order parameters contribute to ``ψ``.
    """
    psi = jnp.asarray(phi_m) - jnp.asarray(phi_c)
    mu = mu_stress_real(psi, sigma_xx, sigma_xy, sigma_yy, kx_wave, ky_wave, B)
    half = jnp.asarray(0.5, dtype=mu.dtype)
    return half * mu, -half * mu
