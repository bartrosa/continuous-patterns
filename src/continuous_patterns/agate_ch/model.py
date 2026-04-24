"""Masks, cavity geometry, reaction, double-well + barrier, ratchet selectors."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Geometry(NamedTuple):
    chi: jnp.ndarray
    ring: jnp.ndarray
    ring_accounting: jnp.ndarray  # thin annulus R−2dx ≤ r < R (accounting only)
    rv: jnp.ndarray
    k_sq: jnp.ndarray
    k_four: jnp.ndarray
    #: component wavenumber squares (``(2π kx_fftfreq)²`` etc.) for anisotropic κ.
    kx_sq: jnp.ndarray
    ky_sq: jnp.ndarray
    #: Full angular wavenumbers ``2π k_fftfreq`` for pseudo-spectral derivatives.
    kx_wave: jnp.ndarray
    ky_wave: jnp.ndarray
    #: Spatially varying Cauchy stress (Experiment 6); zeros when inactive.
    sigma_xx: jnp.ndarray
    sigma_yy: jnp.ndarray
    sigma_xy: jnp.ndarray
    dx: float
    xc: float
    yc: float
    R: float


def xy_grid(L: float, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = L / n
    x = (jnp.arange(n) + 0.5) * dx
    y = (jnp.arange(n) + 0.5) * dx
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def build_geometry(
    L: float,
    R: float,
    n: int,
    eps_scale: float = 2.0,
    *,
    stress_mode: str = "none",
    sigma_0: float = 0.0,
    stress_eps_factor: float = 3.0,
) -> Geometry:
    dx = L / n
    xc = yc = L / 2
    X, Y = xy_grid(L, n)
    rv = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    chi = 0.5 * (1.0 - jnp.tanh((rv - R) / (eps_scale * dx)))
    ring = jnp.exp(-0.5 * ((rv - R) / (2.5 * dx)) ** 2)
    ring = jnp.where(ring > 0.35, 1.0, 0.0)
    ring_accounting = (rv < R) & (rv >= R - 2.0 * dx)

    fx = jnp.fft.fftfreq(n, d=dx)
    ky = jnp.fft.fftfreq(n, d=dx)
    kx = 2 * jnp.pi * fx[:, None]
    kk = 2 * jnp.pi * ky[None, :]
    k_sq = kx**2 + kk**2
    k_four = k_sq**2
    kx_sq = kx**2
    ky_sq = kk**2

    sigma_xx = jnp.zeros((n, n), dtype=jnp.float32)
    sigma_yy = jnp.zeros((n, n), dtype=jnp.float32)
    sigma_xy = jnp.zeros((n, n), dtype=jnp.float32)
    if stress_mode == "none":
        pass
    elif stress_mode == "flamant_two_point":
        if sigma_0 > 0.0:
            from continuous_patterns.agate_ch.stress_fields import flamant_two_point

            eps = stress_eps_factor * dx
            sigma_xx, sigma_yy, sigma_xy = flamant_two_point(
                float(L), float(R), int(n), float(sigma_0), float(eps)
            )
    elif stress_mode == "uniform_uniaxial":
        if float(sigma_0) != 0.0:
            from continuous_patterns.agate_ch.stress_fields import uniform_uniaxial_field

            sigma_xx, sigma_yy, sigma_xy = uniform_uniaxial_field(float(L), int(n), float(sigma_0))
    elif stress_mode == "pure_shear":
        if float(sigma_0) != 0.0:
            from continuous_patterns.agate_ch.stress_fields import pure_shear_field

            sigma_xx, sigma_yy, sigma_xy = pure_shear_field(float(L), int(n), float(sigma_0))
    elif stress_mode == "uniform_biaxial":
        if float(sigma_0) != 0.0:
            from continuous_patterns.agate_ch.stress_fields import uniform_biaxial_field

            sigma_xx, sigma_yy, sigma_xy = uniform_biaxial_field(float(L), int(n), float(sigma_0))
    elif stress_mode == "pressure_gradient":
        if float(sigma_0) != 0.0:
            from continuous_patterns.agate_ch.stress_fields import pressure_gradient_field

            sigma_xx, sigma_yy, sigma_xy = pressure_gradient_field(float(L), int(n), float(sigma_0))
    elif stress_mode == "kirsch":
        if float(sigma_0) != 0.0:
            from continuous_patterns.agate_ch.stress_fields import kirsch_field

            sigma_xx, sigma_yy, sigma_xy = kirsch_field(float(L), float(R), int(n), float(sigma_0))
    else:
        raise ValueError(f"unknown stress_mode: {stress_mode!r}")

    return Geometry(
        chi=chi,
        ring=ring,
        ring_accounting=ring_accounting,
        rv=rv,
        k_sq=k_sq,
        k_four=k_four,
        kx_sq=kx_sq,
        ky_sq=ky_sq,
        kx_wave=kx,
        ky_wave=kk,
        sigma_xx=sigma_xx,
        sigma_yy=sigma_yy,
        sigma_xy=sigma_xy,
        dx=dx,
        xc=xc,
        yc=yc,
        R=R,
    )


def dfdphi_well(phi: jnp.ndarray, W: float) -> jnp.ndarray:
    return 2 * W * phi * (1 - phi) * (1 - 2 * phi)


def dfdphi_barrier(phi: jnp.ndarray, lam: float) -> jnp.ndarray:
    neg = jnp.maximum(-phi, 0.0)
    pos = jnp.maximum(phi - 1.0, 0.0)
    return -2.0 * lam * neg + 2.0 * lam * pos


def dfdphi_total(phi: jnp.ndarray, W: float, lam: float) -> jnp.ndarray:
    return dfdphi_well(phi, W) + dfdphi_barrier(phi, lam)


def smoothstep(lo: float, hi: float, x: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(hi - lo, 1e-9)
    t = jnp.clip((x - lo) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gamma_sigma(
    c: jnp.ndarray, c_ostwald: float, w_ostwald: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    psi_m = jax.nn.sigmoid((c - c_ostwald) / w_ostwald)
    return psi_m, 1.0 - psi_m


def gamma_sigma_ratchet(
    c: jnp.ndarray,
    phi_m: jnp.ndarray,
    c_ostwald: float,
    w_ostwald: float,
    lo: float,
    hi: float,
    ratchet_strength: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    psi_m0 = jax.nn.sigmoid((c - c_ostwald) / w_ostwald)
    sm = smoothstep(lo, hi, phi_m)
    psi_m = psi_m0 + ratchet_strength * (1.0 - psi_m0) * sm
    psi_m = jnp.clip(psi_m, 0.0, 1.0)
    return psi_m, 1.0 - psi_m


def precipitation(
    c: jnp.ndarray,
    phi_m: jnp.ndarray,
    phi_c: jnp.ndarray,
    *,
    k_reaction: float,
    c_sat: float,
) -> jnp.ndarray:
    phi_tot = phi_m + phi_c
    return k_reaction * jnp.maximum(c - c_sat, 0.0) * jnp.maximum(1.0 - phi_tot, 0.0)
