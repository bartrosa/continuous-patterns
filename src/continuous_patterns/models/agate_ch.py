"""Stage I — reaction-coupled CH with cavity masks and rim Dirichlet.

Assembles ``Geometry``, ``SimParams``, and calls :mod:`continuous_patterns.core.imex`
for time integration (``docs/ARCHITECTURE.md`` §4.1, ``docs/PHYSICS.md`` §6.1).
"""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from continuous_patterns.core.diagnostics_stage1 import (
    chi_weighted_silica_integral,
    count_bands_multislice,
    fft_psi_anisotropy_ratio,
    hard_disk_mask,
    jab_metrics_canonical_slice,
    option_b_residual_pct,
    pixel_noise_rms,
)
from continuous_patterns.core.imex import Geometry, SimParams
from continuous_patterns.core.masks import MASK_BUILDERS
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import STRESS_BUILDERS
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models._integrate import make_chunk_runner


def _require(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key {key!r} in {where}")
    return mapping[key]


def build_geometry(cfg: dict[str, Any]) -> Geometry:
    """Build cavity ``Geometry`` from nested ``cfg['geometry']`` + ``cfg['stress']``."""
    gcfg = _require(cfg, "geometry", where="config")
    gtype = _require(gcfg, "type", where="config.geometry")
    if gtype not in MASK_BUILDERS:
        raise ValueError(f"Unknown geometry.type {gtype!r}; allowed: {sorted(MASK_BUILDERS)}")

    L = float(_require(gcfg, "L", where="config.geometry"))
    R = float(_require(gcfg, "R", where="config.geometry"))
    n = int(_require(gcfg, "n", where="config.geometry"))
    eps_scale = float(gcfg.get("eps_scale", 2.0))

    dtype = jnp.float64 if cfg.get("precision") == "float64" else jnp.float32

    builder = MASK_BUILDERS[gtype]
    m = builder(L=L, R=R, n=n, eps_scale=eps_scale, dtype=dtype)

    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)

    st = _require(cfg, "stress", where="config")
    smode = _require(st, "mode", where="config.stress")
    if smode not in STRESS_BUILDERS:
        raise ValueError(f"Unknown stress.mode {smode!r}; allowed: {sorted(STRESS_BUILDERS)}")
    skwargs: dict[str, Any] = {"L": L, "n": n, "dtype": dtype}
    if smode in ("flamant_two_point", "kirsch"):
        skwargs["R"] = R
    # ``stress_eps_factor``: validated on the stress block; only Flamant builder uses it.
    _skip = frozenset({"mode", "stress_coupling_B", "stress_eps_factor", "dtype"})
    for k, v in st.items():
        if k in _skip:
            continue
        skwargs.setdefault(k, v)
    if smode == "flamant_two_point":
        skwargs["stress_eps_factor"] = float(st.get("stress_eps_factor", 3.0))
    sxx, syy, sxy = STRESS_BUILDERS[smode](**skwargs)

    def _to(x: Any) -> Array:
        return jnp.asarray(x, dtype=dtype)

    return Geometry(
        chi=_to(m["chi"]),
        ring=_to(m["ring"]),
        ring_accounting=_to(m["ring_accounting"]),
        sigma_xx=_to(sxx),
        sigma_yy=_to(syy),
        sigma_xy=_to(sxy),
        k_sq=jnp.asarray(k_sq, dtype=dtype),
        kx_sq=jnp.asarray(kx_sq, dtype=dtype),
        ky_sq=jnp.asarray(ky_sq, dtype=dtype),
        kx_wave=jnp.asarray(kx_wave, dtype=dtype),
        ky_wave=jnp.asarray(ky_wave, dtype=dtype),
        k_four=jnp.asarray(k_four, dtype=dtype),
        rv=_to(m["rv"]),
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def build_sim_params(cfg: dict[str, Any]) -> SimParams:
    """Build ``SimParams`` from ``cfg['physics']`` and ``cfg['stress']`` (Stage I defaults)."""
    ph = _require(cfg, "physics", where="config")
    st = _require(cfg, "stress", where="config")

    if "kappa_x" in ph:
        kappa_x = float(ph["kappa_x"])
    elif "kappa" in ph:
        kappa_x = float(ph["kappa"])
    else:
        raise KeyError("physics must include kappa_x (or isotropic alias kappa)")
    if "kappa_y" in ph:
        kappa_y = float(ph["kappa_y"])
    elif "kappa" in ph:
        kappa_y = float(ph["kappa"])
    else:
        kappa_y = kappa_x

    lambda_bar = ph.get("lambda_bar", ph.get("lambda_barrier"))
    if lambda_bar is None:
        raise KeyError("physics must include lambda_bar (or legacy alias lambda_barrier)")

    reaction_active = bool(ph.get("reaction_active", True))
    dirichlet_active = bool(ph.get("dirichlet_active", True))

    return SimParams(
        reaction_active=reaction_active,
        dirichlet_active=dirichlet_active,
        D_c=float(_require(ph, "D_c", where="physics")),
        M_m=float(_require(ph, "M_m", where="physics")),
        M_c=float(_require(ph, "M_c", where="physics")),
        W=float(_require(ph, "W", where="physics")),
        gamma=float(_require(ph, "gamma", where="physics")),
        kappa_x=kappa_x,
        kappa_y=kappa_y,
        stress_coupling_B=float(st.get("stress_coupling_B", 0.0)),
        k_rxn=float(_require(ph, "k_rxn", where="physics")),
        c_sat=float(_require(ph, "c_sat", where="physics")),
        rho_m=float(ph.get("rho_m", 1.0)),
        rho_c=float(ph.get("rho_c", 1.0)),
        c0=float(_require(ph, "c_0", where="physics")),
        lambda_bar=float(lambda_bar),
        c_ostwald=float(_require(ph, "c_ostwald", where="physics")),
        w_ostwald=float(_require(ph, "w_ostwald", where="physics")),
        use_ratchet=bool(ph.get("use_ratchet", True)),
        phi_m_ratchet_low=float(ph.get("phi_m_ratchet_low", 0.3)),
        phi_m_ratchet_high=float(ph.get("phi_m_ratchet_high", 0.5)),
    )


def build_initial_state(
    cfg: dict[str, Any],
    geom: Geometry,
    prm: SimParams,
    key: Array,
) -> tuple[Array, Array, Array]:
    """Random ICs in cavity (``χ``-masked phases); uniform ``c`` from ``physics.c_0``."""
    n = geom.n
    dtype = geom.chi.dtype
    ph = cfg["physics"]
    ic = cfg.get("initial", {})

    phi_m0 = float(ic.get("phi_m_init", 0.0))
    phi_c0 = float(ic.get("phi_c_init", 0.0))
    sig_m = float(ic.get("phi_m_noise", 0.01))
    sig_c = float(ic.get("phi_c_noise", 0.01))

    k1, k2 = jax.random.split(key)
    chi = geom.chi
    phi_m = (phi_m0 + sig_m * jax.random.normal(k1, (n, n), dtype=dtype)) * chi
    phi_c = (phi_c0 + sig_c * jax.random.normal(k2, (n, n), dtype=dtype)) * chi
    c0 = float(_require(ph, "c_0", where="physics"))
    c = jnp.full((n, n), c0, dtype=dtype)
    return phi_m, phi_c, c


def _append_flux_sample(
    state: tuple[Array, Array, Array],
    geom: Geometry,
    flux: dict[str, list[float]],
    *,
    t: float,
    r_fix_frac: float,
) -> None:
    phi_m, phi_c, c = state
    pm = np.asarray(phi_m)
    pc = np.asarray(phi_c)
    cc = np.asarray(c)
    rv = np.asarray(geom.rv)
    ring_acc = np.asarray(geom.ring_accounting)
    dx = float(geom.dx)
    r_fix = float(r_fix_frac) * float(geom.R)
    mask_inner = rv < r_fix
    dissolved_inner = float(np.sum(cc * mask_inner) * dx * dx)
    w = ring_acc > 0.5
    c_ring = float(np.sum(cc * w) / (np.sum(w) + 1e-30))
    band = np.abs(rv - r_fix) < dx
    phi_pack = float(np.sum((pm + pc) * band) / (np.sum(band) + 1e-30))

    flux["times"].append(float(t))
    flux["dissolved_inner_mass"].append(dissolved_inner)
    flux["c_ring_mean"].append(c_ring)
    flux["phi_pack_rfix"].append(phi_pack)


def _option_b_leak_from_flux_samples(
    flux: dict[str, list[float]],
    *,
    D_c: float,
    r_fix: float,
    dx: float,
    inner_area: float,
) -> float:
    """Crude dense-path residual (PHYSICS §10.1) from recorded samples."""
    times = np.asarray(flux["times"], dtype=np.float64)
    M = np.asarray(flux["dissolved_inner_mass"], dtype=np.float64)
    cr = np.asarray(flux["c_ring_mean"], dtype=np.float64)
    if times.size < 2 or inner_area <= 0.0:
        return 0.0
    dM = float(M[-1] - M[0])
    inner_c = M / inner_area
    dr_scale = max(dx, float(r_fix) * 0.25)
    flux_rate = D_c * 2.0 * np.pi * float(r_fix) * (cr - inner_c) / dr_scale
    Ftrap = float(np.trapezoid(flux_rate, x=times))
    return float(option_b_residual_pct(dM, Ftrap))


def _assemble_diagnostics(
    state: tuple[Array, Array, Array],
    geom: Geometry,
    prm: SimParams,
    meta: dict[str, Any],
) -> dict[str, Any]:
    pm = np.asarray(state[0])
    pc = np.asarray(state[1])
    cc = np.asarray(state[2])
    chi = np.asarray(geom.chi)
    dx = float(geom.dx)
    L = float(geom.L)
    R = float(geom.R)

    disk = hard_disk_mask(L=L, n=int(geom.n), cavity_R=R)
    out: dict[str, Any] = {
        "chi_weighted_silica_final": chi_weighted_silica_integral(
            cc, pm, pc, chi, rho_m=float(prm.rho_m), rho_c=float(prm.rho_c), dx=dx
        ),
        "jab_canonical": jab_metrics_canonical_slice(pm, pc, L=L, R=R),
        "bands_multislice": count_bands_multislice(pm, pc, L=L, R=R),
        "psi_fft_anisotropy": fft_psi_anisotropy_ratio(pm, pc, L=L, cavity_R=R),
        "pixel_noise_rms_phi_m": pixel_noise_rms(pm, disk, periodic=True),
    }
    flux = meta.get("flux_samples")
    if isinstance(flux, dict) and flux.get("times"):
        r_fix_frac = float(meta.get("option_b_r_fix_frac", 0.75))
        r_fix = r_fix_frac * R
        mask_inner = np.asarray(geom.rv) < r_fix
        inner_area = float(np.sum(mask_inner) * dx * dx)
        out["option_b_leak_pct"] = _option_b_leak_from_flux_samples(
            flux, D_c=float(prm.D_c), r_fix=r_fix, dx=dx, inner_area=inner_area
        )
    return out


def simulate(cfg: dict[str, Any], *, chunk_size: int = 2000) -> SimResult:
    """Run Stage I to ``time.T`` with chunked ``fori_loop`` integration."""
    cfg_resolved = copy.deepcopy(cfg)
    geom = build_geometry(cfg)
    prm = build_sim_params(cfg)
    tcfg = _require(cfg, "time", where="config")
    dt = float(_require(tcfg, "dt", where="time"))
    T = float(_require(tcfg, "T", where="time"))
    if dt <= 0 or T < 0:
        raise ValueError("time.dt must be positive and time.T non-negative")

    n_total = int(round(T / dt))
    if n_total <= 0:
        raise ValueError("n_total_steps = T/dt must be positive")

    seed = int(cfg.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, k_ic = jax.random.split(key)
    state = build_initial_state(cfg, geom, prm, k_ic)

    outcfg = cfg.get("output", {})
    flux_dt = float(outcfg.get("flux_sample_dt", 2.0))
    flux_every = max(1, int(round(flux_dt / dt)))
    r_fix_frac = float(outcfg.get("option_b_r_fix_frac", 0.75))

    meta: dict[str, Any] = {
        "flux_samples": {
            "times": [],
            "dissolved_inner_mass": [],
            "c_ring_mean": [],
            "phi_pack_rfix": [],
        },
        "option_b_r_fix_frac": r_fix_frac,
        "flux_sample_dt": flux_dt,
        "chunk_size": chunk_size,
        "n_steps": n_total,
        "snapshots": [],
    }

    snap_every = int(tcfg.get("snapshot_every", 10**9))
    if snap_every < 1:
        snap_every = 1

    run_chunk = make_chunk_runner(geom, prm, dt)

    current_step = 0
    next_flux_step = flux_every
    next_snap_step = snap_every

    while current_step < n_total:
        target = n_total
        if next_flux_step <= n_total:
            target = min(target, next_flux_step)
        if next_snap_step <= n_total:
            target = min(target, next_snap_step)
        n_run = min(chunk_size, target - current_step)
        if n_run <= 0:
            n_run = min(chunk_size, n_total - current_step)
        state = run_chunk(state, n_run)
        current_step += n_run

        while next_flux_step <= current_step and next_flux_step <= n_total:
            _append_flux_sample(
                state,
                geom,
                meta["flux_samples"],
                t=float(next_flux_step * dt),
                r_fix_frac=r_fix_frac,
            )
            next_flux_step += flux_every

        while next_snap_step <= current_step and next_snap_step <= n_total:
            meta["snapshots"].append({"step": int(next_snap_step), "t": float(next_snap_step * dt)})
            next_snap_step += snap_every

    t_final = float(n_total * dt)
    state_final = SimState(phi_m=state[0], phi_c=state[1], c=state[2], t=t_final)
    diagnostics = _assemble_diagnostics(state, geom, prm, meta)
    return SimResult(
        state_final=state_final,
        meta=meta,
        diagnostics=diagnostics,
        config_resolved=cfg_resolved,
        paths=None,
    )
