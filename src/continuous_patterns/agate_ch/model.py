"""Masks, cavity geometry, reaction, double-well + barrier, ratchet selectors."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Geometry(NamedTuple):
    chi: jnp.ndarray
    ring: jnp.ndarray
    k_sq: jnp.ndarray
    k_four: jnp.ndarray
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


def build_geometry(L: float, R: float, n: int, eps_scale: float = 2.0) -> Geometry:
    dx = L / n
    xc = yc = L / 2
    X, Y = xy_grid(L, n)
    rv = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    chi = 0.5 * (1.0 - jnp.tanh((rv - R) / (eps_scale * dx)))
    ring = jnp.exp(-0.5 * ((rv - R) / (2.5 * dx)) ** 2)
    ring = jnp.where(ring > 0.35, 1.0, 0.0)

    fx = jnp.fft.fftfreq(n, d=dx)
    ky = jnp.fft.fftfreq(n, d=dx)
    kx = 2 * jnp.pi * fx[:, None]
    kk = 2 * jnp.pi * ky[None, :]
    k_sq = kx**2 + kk**2
    k_four = k_sq**2
    return Geometry(
        chi=chi, ring=ring, k_sq=k_sq, k_four=k_four, dx=dx, xc=xc, yc=yc, R=R
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
