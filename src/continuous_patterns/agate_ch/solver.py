"""Pseudo-spectral IMEX Model C step + chunked scan for long runs."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from continuous_patterns.agate_ch.model import (
    Geometry,
    build_geometry,
    dfdphi_total,
    gamma_sigma_ratchet,
    precipitation,
)


class SimParams(NamedTuple):
    W: float
    gamma: float
    kappa: float
    lambda_barrier: float
    D_c: float
    k_reaction: float
    M_m: float
    M_c: float
    c_sat: float
    c_0: float
    c_ostwald: float
    w_ostwald: float
    uniform_supersaturation: bool
    phi_m_ratchet_low: float
    phi_m_ratchet_high: float
    ratchet_enabled: float
    rho_m: float
    rho_c: float


def cfg_to_sim_params(cfg: dict[str, Any]) -> SimParams:
    W = float(cfg["W"])
    return SimParams(
        W=W,
        gamma=float(cfg["gamma"]),
        kappa=float(cfg["kappa"]),
        lambda_barrier=float(cfg.get("lambda_barrier", 10.0)),
        D_c=float(cfg["D_c"]),
        k_reaction=float(cfg["k_reaction"]),
        M_m=float(cfg["M_m"]),
        M_c=float(cfg["M_c"]),
        c_sat=float(cfg["c_sat"]),
        c_0=float(cfg["c_0"]),
        c_ostwald=float(cfg["c_ostwald"]),
        w_ostwald=float(cfg["w_ostwald"]),
        uniform_supersaturation=bool(cfg.get("uniform_supersaturation", False)),
        phi_m_ratchet_low=float(cfg.get("phi_m_ratchet_low", 0.3)),
        phi_m_ratchet_high=float(cfg.get("phi_m_ratchet_high", 0.5)),
        ratchet_enabled=1.0 if bool(cfg.get("use_ratchet", True)) else 0.0,
        rho_m=float(cfg.get("rho_m", 1.0)),
        rho_c=float(cfg.get("rho_c", 1.0)),
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
    if prm.uniform_supersaturation:
        c = prm.c_0 * geom.chi
    else:
        c = jnp.where(geom.ring > 0.5, prm.c_0, c)

    Gm = precipitation(c, phim, phic, k_reaction=prm.k_reaction, c_sat=prm.c_sat)
    rk = jnp.asarray(prm.ratchet_enabled)
    psi_m, psi_c = gamma_sigma_ratchet(
        c,
        phim,
        prm.c_ostwald,
        prm.w_ostwald,
        prm.phi_m_ratchet_low,
        prm.phi_m_ratchet_high,
        rk,
    )

    fm = dfdphi_total(phim, prm.W, prm.lambda_barrier)
    fc = dfdphi_total(phic, prm.W, prm.lambda_barrier)
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

    c_new = jnp.maximum(jnp.fft.ifft2(c_hat_new).real, 0.0)
    phim_new = jnp.clip(jnp.fft.ifft2(phim_hat_new).real, -0.05, 1.05)
    phic_new = jnp.clip(jnp.fft.ifft2(phic_hat_new).real, -0.05, 1.05)

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


def total_silica_jnp(
    c: jnp.ndarray,
    phim: jnp.ndarray,
    phic: jnp.ndarray,
    chi: jnp.ndarray,
    dx: float,
    rho_m: float,
    rho_c: float,
) -> jnp.ndarray:
    return jnp.sum((c + rho_m * phim + rho_c * phic) * chi) * dx**2


def _stderr_progress(cfg: dict[str, Any]) -> bool:
    if "progress" in cfg:
        return bool(cfg["progress"])
    return sys.stderr.isatty()


def integrate_chunks(
    cfg: dict[str, Any],
    chunk_size: int,
    on_snapshot: (
        Callable[[int, jnp.ndarray, jnp.ndarray, jnp.ndarray], None] | None
    ) = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Run simulation; optional snapshot callback (step index, c, phim, phic)."""
    from continuous_patterns.agate_ch.diagnostics import boundary_flux_mass_rate

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

    silica0 = float(
        total_silica_jnp(
            state[0], state[1], state[2], geom.chi, geom.dx, prm.rho_m, prm.rho_c
        )
    )

    n_steps = max(1, int(round(T / dt)))
    body = make_scan_fn(geom, prm, dt)

    @partial(jax.jit, static_argnames=("length",))
    def advance(s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], length: int):
        return lax.scan(body, s, xs=None, length=length)[0]

    flux_cumulative = 0.0
    step_count = 0
    remaining = n_steps
    want_bar = _stderr_progress(cfg)
    bar = None
    if want_bar:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=n_steps,
                unit="step",
                desc="agate-ch",
                file=sys.stderr,
                mininterval=0.3,
            )
        except ImportError:
            bar = None
    while remaining > 0:
        take = min(chunk_size, remaining)
        state = advance(state, take)
        c_np = np.asarray(jax.device_get(state[0]))
        flux_cumulative += boundary_flux_mass_rate(c_np, L, R, prm.D_c, prm.c_0) * (
            take * dt
        )
        step_count += take
        remaining -= take
        if bar is not None:
            bar.update(take)
        if on_snapshot and step_count % snap_every == 0:
            on_snapshot(step_count, *state)
    if bar is not None:
        bar.close()

    silica1 = float(
        total_silica_jnp(
            state[0], state[1], state[2], geom.chi, geom.dx, prm.rho_m, prm.rho_c
        )
    )
    mb_err = abs(silica1 - silica0 - flux_cumulative) / max(abs(silica0), 1e-15) * 100.0
    meta = {
        "silica_initial": silica0,
        "silica_final": silica1,
        "cumulative_boundary_flux_mass": flux_cumulative,
        "mass_balance_percent": mb_err,
        "geom": geom,
        "prm": prm,
        "mass_initial": silica0,
        "mass_final": silica1,
    }
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
