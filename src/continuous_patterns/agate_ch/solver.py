"""Pseudo-spectral IMEX Model C step + chunked scan for long runs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from continuous_patterns.agate_ch.model import (
    Geometry,
    build_geometry,
    dfdphi,
    gamma_sigma,
    precipitation,
)


class SimParams(NamedTuple):
    W: float
    gamma: float
    kappa: float
    D_c: float
    k_reaction: float
    M_m: float
    M_c: float
    c_sat: float
    c_0: float
    c_ostwald: float
    w_ostwald: float
    uniform_supersaturation: bool


def cfg_to_sim_params(cfg: dict[str, Any]) -> SimParams:
    return SimParams(
        W=float(cfg["W"]),
        gamma=float(cfg["gamma"]),
        kappa=float(cfg["kappa"]),
        D_c=float(cfg["D_c"]),
        k_reaction=float(cfg["k_reaction"]),
        M_m=float(cfg["M_m"]),
        M_c=float(cfg["M_c"]),
        c_sat=float(cfg["c_sat"]),
        c_0=float(cfg["c_0"]),
        c_ostwald=float(cfg["c_ostwald"]),
        w_ostwald=float(cfg["w_ostwald"]),
        uniform_supersaturation=bool(cfg.get("uniform_supersaturation", False)),
    )


def laplacian(u: jnp.ndarray, k_sq: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(-k_sq * jnp.fft.fft2(u)).real


def imex_step(
    state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    _: Any,
    geom: Geometry,
    prm: SimParams,
    dt: float,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
    c, phim, phic = state
    Gm = precipitation(c, phim, phic, k_reaction=prm.k_reaction, c_sat=prm.c_sat)
    psi_m, psi_c = gamma_sigma(c, prm.c_ostwald, prm.w_ostwald)

    fm = dfdphi(phim, prm.W)
    fc = dfdphi(phic, prm.W)
    lap_mu_m = laplacian(fm + prm.gamma * phic, geom.k_sq)
    lap_mu_c = laplacian(fc + prm.gamma * phim, geom.k_sq)

    rhs_m = prm.M_m * lap_mu_m + psi_m * Gm
    rhs_c = prm.M_c * lap_mu_c + psi_c * Gm

    cm = jnp.fft.fft2(c)
    pm = jnp.fft.fft2(phim)
    pc = jnp.fft.fft2(phic)

    c_hat_new = (cm - dt * jnp.fft.fft2(Gm)) / (1.0 + dt * prm.D_c * geom.k_sq)
    denom_m = 1.0 + dt * prm.M_m * prm.kappa * geom.k_four
    denom_c = 1.0 + dt * prm.M_c * prm.kappa * geom.k_four
    phim_hat_new = (pm + dt * jnp.fft.fft2(rhs_m)) / denom_m
    phic_hat_new = (pc + dt * jnp.fft.fft2(rhs_c)) / denom_c

    c_new = jnp.fft.ifft2(c_hat_new).real
    phim_new = jnp.fft.ifft2(phim_hat_new).real
    phic_new = jnp.fft.ifft2(phic_hat_new).real

    chi = geom.chi
    c_new = c_new * chi
    phim_new = phim_new * chi
    phic_new = phic_new * chi

    if prm.uniform_supersaturation:
        c_new = prm.c_0 * chi
    else:
        c_new = jnp.where(geom.ring > 0.5, prm.c_0, c_new)

    return (c_new, phim_new, phic_new), None


def make_scan_fn(
    geom: Geometry, prm: SimParams, dt: float
) -> Callable[..., tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]]:
    def body(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: Any):
        return imex_step(state, None, geom, prm, dt)

    return body


def initial_state(
    geom: Geometry,
    key: jax.Array,
    *,
    c_sat: float,
    c_0: float,
    noise: float,
    uniform_supersaturation: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = geom.chi.shape[0]
    k1, k2 = jax.random.split(key)
    rm = jax.random.normal(k1, (n, n)) * noise
    rc = jax.random.normal(k2, (n, n)) * noise
    chi = geom.chi
    inside = chi > 0.5
    phim = jnp.where(inside, jnp.clip(rm, -0.05, 0.05), 0.0) * chi
    phic = jnp.where(inside, jnp.clip(rc, -0.05, 0.05), 0.0) * chi
    if uniform_supersaturation:
        c = jnp.where(inside, c_0, 0.0) * chi
    else:
        c = jnp.where(inside, c_sat, 0.0) * chi
        c = jnp.where(geom.ring > 0.5, c_0, c)
    return c, phim, phic


def integrate_chunks(
    cfg: dict[str, Any],
    chunk_size: int,
    on_snapshot: (
        Callable[[int, jnp.ndarray, jnp.ndarray, jnp.ndarray], None] | None
    ) = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Run simulation; optional snapshot callback (step index, c, phim, phic)."""
    L = float(cfg["L"])
    R = float(cfg["R"])
    n = int(cfg["grid"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    snap_every = int(cfg["snapshot_every"])
    seed = int(cfg.get("seed", 0))
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(L, R, n)
    key = jax.random.PRNGKey(seed)
    state = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )

    n_steps = max(1, int(round(T / dt)))
    body = make_scan_fn(geom, prm, dt)

    @partial(jax.jit, static_argnames=("length",))
    def advance(s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], length: int):
        return lax.scan(body, s, xs=None, length=length)[0]

    mass0 = float(jnp.sum((state[0] + state[1] + state[2]) * geom.chi) * geom.dx**2)
    step_count = 0
    remaining = n_steps
    while remaining > 0:
        take = min(chunk_size, remaining)
        state = advance(state, take)
        step_count += take
        remaining -= take
        if on_snapshot and step_count % snap_every == 0:
            on_snapshot(step_count, *state)

    mass1 = float(jnp.sum((state[0] + state[1] + state[2]) * geom.chi) * geom.dx**2)
    meta = {"mass_initial": mass0, "mass_final": mass1, "geom": geom, "prm": prm}
    return (*state, meta)


def simulate_to_host(
    cfg: dict[str, Any],
    chunk_size: int = 2000,
) -> tuple[Any, Any, Any, dict[str, Any], list[tuple[int, Any, Any, Any]]]:
    """Returns final fields, meta, and list of (step, c, phim, phic) snapshots."""
    snaps: list[tuple[int, Any, Any, Any]] = []

    def cb(step: int, c: Any, pm: Any, pc: Any) -> None:
        snaps.append(
            (
                step,
                jax.device_get(c),
                jax.device_get(pm),
                jax.device_get(pc),
            )
        )

    c, pm, pc, meta = integrate_chunks(cfg, chunk_size, on_snapshot=cb)
    return (
        jax.device_get(c),
        jax.device_get(pm),
        jax.device_get(pc),
        meta,
        snaps,
    )
