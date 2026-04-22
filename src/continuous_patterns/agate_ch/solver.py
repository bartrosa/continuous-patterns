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

from continuous_patterns.agate_ch.diagnostics import (
    compute_influx_rate_physical,
    compute_influx_rate_physical_jnp,
    dissolved_silica_mass_inside_disk_numpy,
    flux_validation_milestone_line,
    radial_shell_widths_from_dx,
    silica_mass_inside_disk_numpy,
)
from continuous_patterns.agate_ch.model import (
    Geometry,
    build_geometry,
    dfdphi_total,
    gamma_sigma_ratchet,
    precipitation,
    xy_grid,
)


def print_ring_mask_sanity(
    geom: Geometry,
    *,
    L: float,
    R: float,
    stream: Any = None,
) -> dict[str, float]:
    """Host-side sanity check on enforcement ring vs thin accounting annulus."""
    out = sys.stderr if stream is None else stream
    dx = float(geom.dx)
    rv_np = np.asarray(jax.device_get(geom.rv))
    ring_wide_np = np.asarray(jax.device_get(geom.ring > 0.5)).astype(np.float64)
    thin_np = np.asarray(jax.device_get(geom.ring_accounting)).astype(np.float64)

    ring_area_cells = int(np.sum(ring_wide_np))
    ring_area_physical = ring_area_cells * dx * dx
    cavity_mask = rv_np <= R + 1e-9
    cavity_cells = int(np.sum(cavity_mask))
    cavity_area = cavity_cells * dx * dx
    cavity_perimeter = float(2.0 * np.pi * R)
    implied_ring_thickness = (
        ring_area_physical / cavity_perimeter if cavity_perimeter > 0 else 0.0
    )

    thin_cells = int(np.sum(thin_np))
    thin_area_physical = thin_cells * dx * dx

    frac_cavity = ring_area_physical / cavity_area if cavity_area > 0 else 1.0

    print("RING MASK SANITY:", file=out)
    print(f"  cells in enforcement ring (>0.5): {ring_area_cells}", file=out)
    print(f"  physical area (wide ring): {ring_area_physical:.2f}", file=out)
    print(f"  cavity perimeter: {cavity_perimeter:.2f}", file=out)
    print(
        f"  implied ring thickness (wide): {implied_ring_thickness / dx:.3f} dx="
        f"{dx:.3f}",
        file=out,
    )
    print(f"  cells in thin accounting ring [R−2dx,R): {thin_cells}", file=out)
    print(f"  thin ring physical area: {thin_area_physical:.4f}", file=out)
    print(f"  wide ring / cavity area: {frac_cavity:.4f}", file=out)

    stats: dict[str, float] = {
        "wide_ring_cells": float(ring_area_cells),
        "thin_ring_cells": float(thin_cells),
        "implied_wide_thickness_dx": float(implied_ring_thickness / dx),
        "wide_ring_frac_of_cavity_area": float(frac_cavity),
        "thin_ring_area_physical": float(thin_area_physical),
        "wide_ring_area_physical": float(ring_area_physical),
    }
    return stats


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
    disable_dirichlet: bool


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
        disable_dirichlet=bool(cfg.get("disable_dirichlet", False)),
    )


def laplacian(u: jnp.ndarray, k_sq: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(-k_sq * jnp.fft.fft2(u)).real


def imex_step(
    state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    _: Any,
    geom: Geometry,
    prm: SimParams,
    dt: float,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    c, phim, phic = state
    dx = jnp.asarray(geom.dx, dtype=jnp.float32)
    if not prm.disable_dirichlet:
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

    if not prm.disable_dirichlet:
        chi = geom.chi
        c_new = c_new * chi
        phim_new = phim_new * chi
        phic_new = phic_new * chi

    dx_f64 = jnp.asarray(dx, jnp.float64)

    if prm.disable_dirichlet:
        delta_pair = jnp.zeros(2, dtype=jnp.float64)
    elif prm.uniform_supersaturation:
        c_before_bc = c_new
        c_new = prm.c_0 * chi
        delta_full = jnp.sum((c_new - c_before_bc).astype(jnp.float64)) * dx_f64**2
        delta_pair = jnp.stack([delta_full, delta_full])
    else:
        c_before_bc = c_new
        ring_bc = geom.ring > 0.5
        c_new = jnp.where(ring_bc, prm.c_0, c_new)
        dc = c_new.astype(jnp.float64) - c_before_bc.astype(jnp.float64)
        thin_m = jnp.asarray(geom.ring_accounting, dtype=jnp.float64)
        delta_thin = jnp.sum(dc * thin_m) * dx_f64**2
        delta_full = jnp.sum(dc) * dx_f64**2
        delta_pair = jnp.stack([delta_thin, delta_full])

    return (c_new, phim_new, phic_new), delta_pair


def make_scan_fn(
    geom: Geometry, prm: SimParams, dt: float
) -> Callable[..., tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]]:
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


def initial_state_blob(
    geom: Geometry,
    *,
    L: float,
    c_sat: float,
    c_0: float,
    blob_x_frac: float = 0.6,
    blob_y_frac: float = 0.6,
    sigma: float = 10.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gaussian supersaturation bump (χ-windowed); φ phases zero."""
    n = geom.chi.shape[0]
    X, Y = xy_grid(L, n)
    xc_b = blob_x_frac * L
    yc_b = blob_y_frac * L
    chi = geom.chi
    rv2 = (X - xc_b) ** 2 + (Y - yc_b) ** 2
    gauss = jnp.exp(-rv2 / (2.0 * sigma**2))
    c = chi * (c_sat + (c_0 - c_sat) * gauss)
    phim = jnp.zeros_like(c)
    phic = jnp.zeros_like(c)
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


def total_silica_full_domain_jnp(
    c: jnp.ndarray,
    phim: jnp.ndarray,
    phic: jnp.ndarray,
    dx: float,
    rho_m: float,
    rho_c: float,
) -> jnp.ndarray:
    """Integral of total silica over the full periodic grid (no cavity mask)."""
    return jnp.sum(c + rho_m * phim + rho_c * phic) * dx**2


def _stderr_progress(cfg: dict[str, Any]) -> bool:
    if "progress" in cfg:
        return bool(cfg["progress"])
    return sys.stderr.isatty()


def _integrate_flux_diagnosis(
    cfg: dict[str, Any],
    chunk_size: int,
    on_snapshot: (Callable[[int, jnp.ndarray, jnp.ndarray, jnp.ndarray], None] | None),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Single lax.scan run with per-step influx and stdout boundary diagnostics."""
    L = float(cfg["L"])
    R = float(cfg["R"])
    n = int(cfg["grid"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    snap_every = int(cfg["snapshot_every"])
    seed = int(cfg.get("seed", 0))
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(L, R, n)
    dx = float(geom.dx)
    key = jax.random.PRNGKey(seed)
    ring_sanity: dict[str, float] | None = None
    if cfg.get("print_ring_mask_sanity", True):
        ring_sanity = print_ring_mask_sanity(geom, L=L, R=R)
    state0 = initial_state(
        geom,
        key,
        c_sat=prm.c_sat,
        c_0=prm.c_0,
        noise=0.01,
        uniform_supersaturation=prm.uniform_supersaturation,
    )

    silica0 = float(
        total_silica_jnp(
            state0[0], state0[1], state0[2], geom.chi, geom.dx, prm.rho_m, prm.rho_c
        )
    )
    _dxdiag = float(geom.dx)
    _odx = float(cfg.get("physical_flux_outer_dx", 3.0))
    _idx = float(cfg.get("physical_flux_inner_dx", 5.0))
    _r_outer_disk, _ = radial_shell_widths_from_dx(
        _dxdiag, R, outer_dx=_odx, inner_dx=_idx
    )
    c0_np_init = np.asarray(jax.device_get(state0[0]))
    pm0_np_init = np.asarray(jax.device_get(state0[1]))
    pc0_np_init = np.asarray(jax.device_get(state0[2]))
    silica_disk_0 = silica_mass_inside_disk_numpy(
        c0_np_init,
        pm0_np_init,
        pc0_np_init,
        L=L,
        r_disk=_r_outer_disk,
        rho_m=float(prm.rho_m),
        rho_c=float(prm.rho_c),
    )
    n_steps = max(1, int(round(T / dt)))

    def scan_body(
        s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: Any
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        new_s, _ = imex_step(s, None, geom, prm, dt)
        infl = compute_influx_rate_physical_jnp(
            new_s[0],
            L=L,
            R=R,
            D_c=float(prm.D_c),
            dx=dx,
            outer_dx=_odx,
            inner_dx=_idx,
        )
        return new_s, infl

    final_state, aux_arr = lax.scan(scan_body, state0, xs=None, length=n_steps)

    influx_arr_np = np.asarray(aux_arr)
    influx_total_dense = float(np.trapezoid(influx_arr_np, dx=dt))

    snap_steps = [s for s in range(1, n_steps + 1) if s % snap_every == 0]
    if snap_steps:
        times_sn = np.array([float(s) * dt for s in snap_steps], dtype=np.float64)
        rates_sn = np.array([influx_arr_np[s - 1] for s in snap_steps])
        influx_total_snapshots = (
            float(np.trapezoid(rates_sn, times_sn))
            if times_sn.size >= 2
            else float(rates_sn[0]) * T
        )
    else:
        influx_total_snapshots = float("nan")

    silica1 = float(
        total_silica_jnp(
            final_state[0],
            final_state[1],
            final_state[2],
            geom.chi,
            geom.dx,
            prm.rho_m,
            prm.rho_c,
        )
    )
    c1_np_fin = np.asarray(jax.device_get(final_state[0]))
    pm1_np_fin = np.asarray(jax.device_get(final_state[1]))
    pc1_np_fin = np.asarray(jax.device_get(final_state[2]))
    silica_disk_1 = silica_mass_inside_disk_numpy(
        c1_np_fin,
        pm1_np_fin,
        pc1_np_fin,
        L=L,
        r_disk=_r_outer_disk,
        rho_m=float(prm.rho_m),
        rho_c=float(prm.rho_c),
    )
    c_disk_0 = dissolved_silica_mass_inside_disk_numpy(
        c0_np_init, L=L, r_disk=_r_outer_disk
    )
    c_disk_1 = dissolved_silica_mass_inside_disk_numpy(
        c1_np_fin, L=L, r_disk=_r_outer_disk
    )
    delta_c_disk = c_disk_1 - c_disk_0
    delta_silica_disk = silica_disk_1 - silica_disk_0

    milestones = sorted({0, 1, 2, 5, 10, 50, 100, 200}.union({n_steps}))
    milestones = [m for m in milestones if m <= n_steps]

    @partial(jax.jit, static_argnames=("length",))
    def advance(
        s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], length: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def body_carry(st: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: Any):
            ns, _ = imex_step(st, None, geom, prm, dt)
            return ns, None

        out_s, _ = lax.scan(body_carry, s, xs=None, length=length)
        return out_s

    dx_j = float(geom.dx)
    silica_full_initial_f = float(
        jax.device_get(
            total_silica_full_domain_jnp(
                state0[0],
                state0[1],
                state0[2],
                dx_j,
                prm.rho_m,
                prm.rho_c,
            )
        )
    )

    print("[flux diagnosis] influx vs disk silica rate (stdout)")
    walk = state0
    walk_at = 0
    prev_step: int | None = None
    prev_c_inside: float | None = None
    for tgt in milestones:
        if tgt > walk_at:
            walk = advance(walk, tgt - walk_at)
            walk_at = tgt
        if tgt == n_steps:
            st_show = final_state
        else:
            st_show = walk
        c_np = np.asarray(jax.device_get(st_show[0]))
        influx_rate = compute_influx_rate_physical(
            c_np,
            L=L,
            R=R,
            D_c=float(prm.D_c),
            c_0=float(prm.c_0),
            outer_dx=_odx,
            inner_dx=_idx,
        )
        s_c_inside = dissolved_silica_mass_inside_disk_numpy(
            c_np, L=L, r_disk=_r_outer_disk
        )
        if prev_step is None:
            d_c_dt = float("nan")
        else:
            d_c_dt = (s_c_inside - prev_c_inside) / (float(tgt - prev_step) * dt)
        line = flux_validation_milestone_line(
            t=float(tgt) * dt,
            influx_rate=influx_rate,
            d_c_dt_disk=d_c_dt,
        )
        print(line)
        prev_step = tgt
        prev_c_inside = s_c_inside

    full_delta = silica1 - silica0
    flux_closure_ratio_full = (
        influx_total_dense / full_delta if abs(full_delta) > 1e-20 else float("nan")
    )
    flux_closure_ratio_c_disk = (
        influx_total_dense / delta_c_disk if abs(delta_c_disk) > 1e-20 else float("nan")
    )
    print(
        f"influx integrated with dt spacing: {influx_total_dense:.2f}\n"
        f"influx integrated with snapshot spacing: {influx_total_snapshots:.2f}\n"
        f"silica gain (full cavity): {full_delta:.2f}\n"
        f"silica gain (disk r<R-{_odx:g}dx): {delta_silica_disk:.2f}\n"
        f"dissolved c gain (disk r<R-{_odx:g}dx): {delta_c_disk:.2f}\n"
        "flux_closure_ratio (∫flux dt / full silica gain): "
        f"{flux_closure_ratio_full:.4f}\n"
        "flux_closure_ratio (∫flux dt / dissolved gain in disk): "
        f"{flux_closure_ratio_c_disk:.4f}"
    )

    influx_tracked = silica1 - silica0
    denom = max(abs(silica0), abs(silica1), abs(influx_tracked))
    mb_direct = abs(silica1 - silica0 - influx_tracked) / max(denom, 1e-30) * 100.0

    if cfg.get("print_mass_balance", True):
        print(
            f"silica_initial: {silica0:.3f}\n"
            f"silica_final:   {silica1:.3f}\n"
            f"influx_direct:  {influx_tracked:.3f}  (final − initial silica)\n"
            f"mass_balance_direct: {mb_direct:.6f}% (tautological)",
            file=sys.stderr,
        )

    silica_full_final_f = float(
        jax.device_get(
            total_silica_full_domain_jnp(
                final_state[0],
                final_state[1],
                final_state[2],
                dx_j,
                prm.rho_m,
                prm.rho_c,
            )
        )
    )

    walk_snap = state0
    prev_s = 0
    if on_snapshot:
        for s in snap_steps:
            walk_snap = advance(walk_snap, s - prev_s)
            prev_s = s
            on_snapshot(s, walk_snap[0], walk_snap[1], walk_snap[2])

    meta = {
        "silica_initial": silica0,
        "silica_final": silica1,
        "cumulative_boundary_flux_mass": influx_tracked,
        "mass_balance_percent_direct": mb_direct,
        "mass_balance_method_direct": "silica_delta",
        "geom": geom,
        "prm": prm,
        "mass_initial": silica0,
        "mass_final": silica1,
        "influx_total_dense_trapz": influx_total_dense,
        "influx_total_snapshots_trapz": influx_total_snapshots,
        "flux_closure_ratio_dense": flux_closure_ratio_full,
        "flux_closure_ratio_dissolved_disk": flux_closure_ratio_c_disk,
        "silica_disk_initial": silica_disk_0,
        "silica_disk_final": silica_disk_1,
        "delta_silica_disk": delta_silica_disk,
        "dissolved_c_disk_initial": c_disk_0,
        "dissolved_c_disk_final": c_disk_1,
        "delta_dissolved_c_disk": delta_c_disk,
        "silica_full_domain_initial": silica_full_initial_f,
        "silica_full_domain_final": silica_full_final_f,
        "ring_mask_sanity": ring_sanity,
    }
    return (*final_state, meta)


def integrate_chunks(
    cfg: dict[str, Any],
    chunk_size: int,
    on_snapshot: (
        Callable[[int, jnp.ndarray, jnp.ndarray, jnp.ndarray], None] | None
    ) = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Run simulation; optional snapshot callback (step index, c, phim, phic)."""
    if cfg.get("diagnose_flux_detail"):
        return _integrate_flux_diagnosis(cfg, chunk_size, on_snapshot)

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
    ring_sanity: dict[str, float] | None = None
    if cfg.get("print_ring_mask_sanity", True):
        ring_sanity = print_ring_mask_sanity(geom, L=L, R=R)
    ic_mode = str(cfg.get("initial_condition", "default"))
    if ic_mode == "blob":
        state = initial_state_blob(
            geom,
            L=L,
            c_sat=prm.c_sat,
            c_0=prm.c_0,
            blob_x_frac=float(cfg.get("blob_x_frac", 0.6)),
            blob_y_frac=float(cfg.get("blob_y_frac", 0.6)),
            sigma=float(cfg.get("blob_sigma", 10.0)),
        )
    else:
        state = initial_state(
            geom,
            key,
            c_sat=prm.c_sat,
            c_0=prm.c_0,
            noise=0.01,
            uniform_supersaturation=prm.uniform_supersaturation,
        )
    ic_state = state

    silica0 = float(
        total_silica_jnp(
            state[0], state[1], state[2], geom.chi, geom.dx, prm.rho_m, prm.rho_c
        )
    )

    n_steps = max(1, int(round(T / dt)))

    @partial(jax.jit, static_argnames=("length",))
    def advance(
        s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], length: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def body_carry(st: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: Any):
            ns, _ = imex_step(st, None, geom, prm, dt)
            return ns, None

        out_s, _ = lax.scan(body_carry, s, xs=None, length=length)
        return out_s

    def _silica_total(st: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> float:
        return float(
            total_silica_jnp(
                st[0], st[1], st[2], geom.chi, geom.dx, prm.rho_m, prm.rho_c
            )
        )

    def _print_diag(n: int, st: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
        pm = np.asarray(jax.device_get(st[1]))
        pc = np.asarray(jax.device_get(st[2]))
        fm = float(np.mean(np.abs(pm) > 1.05))
        fc = float(np.mean(np.abs(pc) > 1.05))
        print(
            f"step {n}: |phi_m|>1.05 fraction = {fm:.4f}, "
            f"|phi_c|>1.05 fraction = {fc:.4f}",
            file=sys.stderr,
        )

    influx_tracked = 0.0
    step_count = 0
    remaining = n_steps
    milestones = [1000, 10000, 100000] if cfg.get("diagnose_overshoot", False) else []
    mi = 0

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

    if cfg.get("diagnose_overshoot", False):
        _print_diag(0, state)

    while remaining > 0:
        if mi < len(milestones) and step_count < milestones[mi]:
            need = milestones[mi] - step_count
            if 0 < need <= remaining:
                s0 = _silica_total(state)
                state = advance(state, need)
                s1 = _silica_total(state)
                influx_tracked += s1 - s0
                step_count += need
                remaining -= need
                _print_diag(step_count, state)
                mi += 1
                if bar is not None:
                    bar.update(need)
                if on_snapshot and step_count % snap_every == 0:
                    on_snapshot(step_count, *state)
                continue

        take = min(chunk_size, remaining)
        s0 = _silica_total(state)
        state = advance(state, take)
        s1 = _silica_total(state)
        influx_tracked += s1 - s0
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
    dx_j = float(geom.dx)
    silica_full_initial = float(
        jax.device_get(
            total_silica_full_domain_jnp(
                ic_state[0],
                ic_state[1],
                ic_state[2],
                dx_j,
                prm.rho_m,
                prm.rho_c,
            )
        )
    )
    silica_full_final = float(
        jax.device_get(
            total_silica_full_domain_jnp(
                state[0],
                state[1],
                state[2],
                dx_j,
                prm.rho_m,
                prm.rho_c,
            )
        )
    )

    denom = max(abs(silica0), abs(silica1), abs(influx_tracked))
    mb_direct = abs(silica1 - silica0 - influx_tracked) / max(denom, 1e-30) * 100.0

    if cfg.get("print_mass_balance", True):
        print(
            f"silica_initial: {silica0:.3f}\n"
            f"silica_final:   {silica1:.3f}\n"
            f"influx_direct:  {influx_tracked:.3f}  (sum of Δ silica per chunk)\n"
            f"raw residual:   {silica1 - silica0 - influx_tracked:.3f}\n"
            f"denominator:    {denom:.3f}\n"
            f"mass_balance_direct: {mb_direct:.6f}% (tautological)",
            file=sys.stderr,
        )

    meta = {
        "silica_initial": silica0,
        "silica_final": silica1,
        "cumulative_boundary_flux_mass": influx_tracked,
        "mass_balance_percent_direct": mb_direct,
        "mass_balance_method_direct": "chunk_silica_deltas",
        "geom": geom,
        "prm": prm,
        "mass_initial": silica0,
        "mass_final": silica1,
        "silica_full_domain_initial": silica_full_initial,
        "silica_full_domain_final": silica_full_final,
        "ring_mask_sanity": ring_sanity,
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


def simulate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Short diagnostic run: periodic spectral mass series (see NOTES.md)."""
    L = float(cfg["L"])
    R = float(cfg["R"])
    n = int(cfg["grid"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    snap_every = max(1, int(cfg.get("snapshot_every", 10)))
    prm = cfg_to_sim_params(cfg)
    geom = build_geometry(L, R, n)
    ic_mode = str(cfg.get("initial_condition", "default"))
    if ic_mode == "blob":
        state = initial_state_blob(
            geom,
            L=L,
            c_sat=prm.c_sat,
            c_0=prm.c_0,
            blob_x_frac=float(cfg.get("blob_x_frac", 0.6)),
            blob_y_frac=float(cfg.get("blob_y_frac", 0.6)),
            sigma=float(cfg.get("blob_sigma", 10.0)),
        )
    else:
        key = jax.random.PRNGKey(int(cfg.get("seed", 0)))
        state = initial_state(
            geom,
            key,
            c_sat=prm.c_sat,
            c_0=prm.c_0,
            noise=float(cfg.get("noise_ic", 0.01)),
            uniform_supersaturation=prm.uniform_supersaturation,
        )

    n_steps = max(1, int(round(T / dt)))
    dx_j = float(geom.dx)

    @jax.jit
    def advance_one(s: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        ns, _ = imex_step(s, None, geom, prm, dt)
        return ns

    def mass_of(st: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> float:
        return float(
            jax.device_get(
                total_silica_full_domain_jnp(
                    st[0], st[1], st[2], dx_j, prm.rho_m, prm.rho_c
                )
            )
        )

    total_mass_series: list[float] = []
    snapshot_times: list[float] = []
    st = state
    total_mass_series.append(mass_of(st))
    snapshot_times.append(0.0)
    for step_i in range(1, n_steps + 1):
        st = advance_one(st)
        if step_i % snap_every == 0:
            total_mass_series.append(mass_of(st))
            snapshot_times.append(float(step_i * dt))

    return {
        "total_mass_series": total_mass_series,
        "snapshot_times": snapshot_times,
    }


def spectral_mass_conservation_diagnostic(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fixed-horizon blob IC; no Dirichlet / no χ projection; returns leak %."""
    dcfg = dict(cfg)
    dcfg["disable_dirichlet"] = True
    dcfg["initial_condition"] = "blob"
    dcfg["T"] = float(cfg.get("spectral_mass_T", cfg.get("option_D_T", 1.0)))
    dcfg["dt"] = float(
        cfg.get("spectral_mass_dt", cfg.get("option_D_dt", cfg.get("dt", 0.01)))
    )
    dcfg["snapshot_every"] = max(
        1,
        int(
            cfg.get(
                "spectral_mass_snapshot_every",
                cfg.get("option_D_snapshot_every", 10),
            )
        ),
    )
    dcfg["progress"] = False
    raw = simulate(dcfg)
    m0 = raw["total_mass_series"][0]
    m1 = raw["total_mass_series"][-1]
    drift = m1 - m0
    leak_pct = 100.0 * drift / max(abs(m0), 1e-30)
    return {
        "leak_pct": leak_pct,
        "total_mass_initial": m0,
        "total_mass_final": m1,
        "drift": drift,
        "snapshot_times": raw["snapshot_times"],
        "total_mass_series": raw["total_mass_series"],
        "note": (
            "Periodic IMEX+FFT without Dirichlet enforcement or χ projection; "
            "mass integrated on the full periodic grid."
        ),
    }
