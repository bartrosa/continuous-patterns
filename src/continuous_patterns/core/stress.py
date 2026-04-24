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


def kirsch(
    *, L: float, R: float, n: int, sigma_0: float, **kwargs: Any
) -> tuple[Array, Array, Array]:
    """Reserved Kirsch / inclusion theory path (PHYSICS §7.6) — not implemented."""
    raise NotImplementedError(
        "Kirsch / inclusion-theory stress is reserved for future work (PHYSICS §7.6)."
    )


STRESS_BUILDERS: dict[str, Callable[..., tuple[Array, Array, Array]]] = {
    "none": none,
    "uniform_uniaxial": uniform_uniaxial,
    "uniform_biaxial": uniform_biaxial,
    "pure_shear": pure_shear,
    "flamant_two_point": flamant_two_point,
    "pressure_gradient": pressure_gradient,
    "kirsch": kirsch,
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
    """Return ``(δμ_m, δμ_c)`` in real space with ``±½`` ψ-split (PHYSICS §5)."""
    psi = jnp.asarray(phi_m) - jnp.asarray(phi_c)
    mu = _mu_stress_real(psi, sigma_xx, sigma_xy, sigma_yy, kx_wave, ky_wave, B)
    half = jnp.asarray(0.5, dtype=mu.dtype)
    return half * mu, -half * mu
