"""Stage I — reaction-coupled CH with cavity masks and rim Dirichlet.

Assembles ``Geometry``, ``SimParams``, and calls :mod:`continuous_patterns.core.imex`
for time integration (``docs/ARCHITECTURE.md`` §4.1, ``docs/PHYSICS.md`` §6.1).
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm.auto import tqdm

from continuous_patterns.core.diagnostics_stage1 import (
    azimuthal_mean_at_radius_numpy,
    chi_weighted_silica_integral,
    count_bands_multislice,
    dissolved_mass_disk_numpy,
    fft_psi_anisotropy_ratio,
    hard_disk_mask,
    jab_metrics_canonical_slice,
    pixel_noise_rms,
)
from continuous_patterns.core.imex import Geometry, SimParams
from continuous_patterns.core.masks import MASK_BUILDERS
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import STRESS_BUILDERS
from continuous_patterns.core.stress import none as stress_none
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models._integrate import make_chunk_runner

logger = logging.getLogger(__name__)


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
    """``φ_m``, ``φ_c`` random in cavity (``χ``-masked); ``c`` = ``c_sat`` inside, 0 outside.

    Interior ``c = c_sat`` (or ``initial.c_init``) avoids uniform supersaturation at ``t=0``;
    the rim Dirichlet supplies ``c_0`` each step so a diffusive gradient rim→interior can form.
    Outside ``χ≈0``, ``c=0`` — no periodic-torus silica reservoir (see ``docs/PHYSICS.md`` §6.1).
    """
    _ = prm
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

    c_sat = float(_require(ph, "c_sat", where="physics"))
    _ = float(_require(ph, "c_0", where="physics"))  # must be present for rim Dirichlet
    c_init_interior = float(ic.get("c_init", c_sat))
    c = c_init_interior * chi
    c = jnp.asarray(c, dtype=dtype)
    return phi_m, phi_c, c


def _geometry_bulk_spectral(L: float, n: int, dtype: jnp.dtype) -> Geometry:
    """Periodic torus: ``χ≡1``, ``ring≡0``, for Option D spectral-mass diagnostic."""
    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)
    z = jnp.zeros((n, n), dtype=dtype)
    o = jnp.ones((n, n), dtype=dtype)
    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x_cent = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y_cent = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    rv = jnp.sqrt((x_cent - xc) ** 2 + (y_cent - yc) ** 2)
    sxx, syy, sxy = stress_none(L=L, n=n, dtype=dtype)
    return Geometry(
        chi=o,
        ring=z,
        ring_accounting=z,
        sigma_xx=jnp.asarray(sxx, dtype=dtype),
        sigma_yy=jnp.asarray(syy, dtype=dtype),
        sigma_xy=jnp.asarray(sxy, dtype=dtype),
        k_sq=jnp.asarray(k_sq, dtype=dtype),
        kx_sq=jnp.asarray(kx_sq, dtype=dtype),
        ky_sq=jnp.asarray(ky_sq, dtype=dtype),
        kx_wave=jnp.asarray(kx_wave, dtype=dtype),
        ky_wave=jnp.asarray(ky_wave, dtype=dtype),
        k_four=jnp.asarray(k_four, dtype=dtype),
        rv=rv,
        dx=float(dx),
        L=float(L),
        R=0.0,
        n=int(n),
        xc=float(xc),
        yc=float(yc),
    )


def run_spectral_mass_diagnostic(cfg: dict[str, Any]) -> dict[str, float]:
    """Option D: short periodic CH+diffusion on a Gaussian ``c`` bump (PHYSICS §10.1)."""
    gcfg = _require(cfg, "geometry", where="config")
    L = float(_require(gcfg, "L", where="config.geometry"))
    n = int(_require(gcfg, "n", where="config.geometry"))
    out = cfg.get("output", {})
    T_dm = float(out.get("spectral_mass_T", 1.0))
    dt_dm = float(out.get("spectral_mass_dt", 0.01))
    if dt_dm <= 0 or T_dm < 0:
        raise ValueError("spectral_mass_dt must be positive and spectral_mass_T non-negative")
    n_steps = int(round(T_dm / dt_dm))
    if n_steps <= 0:
        raise ValueError("spectral_mass diagnostic: n_steps must be positive")

    dtype = jnp.float64 if cfg.get("precision") == "float64" else jnp.float32
    geom = _geometry_bulk_spectral(L=L, n=n, dtype=dtype)

    cfg_dm = copy.deepcopy(cfg)
    cfg_dm.setdefault("physics", {})
    cfg_dm["physics"]["reaction_active"] = False
    cfg_dm["physics"]["dirichlet_active"] = False
    prm = build_sim_params(cfg_dm)

    ph = cfg.get("physics", {})
    rho_m = float(ph.get("rho_m", 1.0))
    rho_c = float(ph.get("rho_c", 1.0))

    dx = L / n
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    xg = (ii + 0.5) * dx
    yg = (jj + 0.5) * dx
    x0 = 0.35 * L
    y0 = 0.5 * L
    sigma = L / 15.0
    bump = jnp.exp(-((xg - x0) ** 2 + (yg - y0) ** 2) / (2.0 * sigma**2))
    c_ic = jnp.asarray(0.5, dtype=dtype) * bump
    phi_m = jnp.zeros((n, n), dtype=dtype)
    phi_c = jnp.zeros((n, n), dtype=dtype)
    state = (phi_m, phi_c, c_ic)

    def _total_mass(s: tuple[Array, Array, Array]) -> float:
        pm, pc, cc = s
        ch = np.asarray(jax.device_get(geom.chi))
        return float(
            np.sum(
                ch
                * (
                    np.asarray(jax.device_get(cc))
                    + rho_m * np.asarray(jax.device_get(pm))
                    + rho_c * np.asarray(jax.device_get(pc))
                )
            )
            * dx
            * dx
        )

    m0 = _total_mass(state)
    run_chunk = make_chunk_runner(geom, prm, dt_dm)
    state, _inj = run_chunk(state, n_steps)
    m1 = _total_mass(state)
    denom = max(abs(m0), 1e-30)
    leak_pct = 100.0 * abs(m1 - m0) / denom
    return {
        "leak_pct": float(leak_pct),
        "M_initial": float(m0),
        "M_final": float(m1),
        "steps": int(n_steps),
    }


def _append_flux_sample(
    state: tuple[Array, Array, Array],
    geom: Geometry,
    flux: dict[str, list[float]],
    *,
    t: float,
    r_fix_frac: float,
    D_c: float,
) -> None:
    """v1-style flux samples: bilinear circle means + ``2·dx`` central difference for ``∂c/∂r``."""
    phi_m, phi_c, c = state
    pm = np.asarray(phi_m)
    pc = np.asarray(phi_c)
    cc = np.asarray(c)
    L = float(geom.L)
    R = float(geom.R)
    dx = float(geom.dx)
    r_fix = float(r_fix_frac) * R

    r_out = r_fix + dx
    r_in = max(r_fix - dx, 1e-6)
    c_out = azimuthal_mean_at_radius_numpy(cc, L=L, r_abs=r_out)
    c_in = azimuthal_mean_at_radius_numpy(cc, L=L, r_abs=r_in)
    dc_dr = (c_out - c_in) / (2.0 * dx)
    perimeter = 2.0 * np.pi * r_fix
    flux_rate = float(D_c) * dc_dr * perimeter

    m_dissolved = dissolved_mass_disk_numpy(cc, L=L, r_disk=r_fix)
    phi_sum = pm + pc
    phi_pack = azimuthal_mean_at_radius_numpy(phi_sum, L=L, r_abs=r_fix)

    flux["times"].append(float(t))
    flux["M_dissolved"].append(float(m_dissolved))
    flux["flux_rate"].append(float(flux_rate))
    flux["phi_pack_rfix"].append(float(phi_pack))
    flux["c_in_circle"].append(float(c_in))
    flux["c_out_circle"].append(float(c_out))


def _compute_surface_flux_balance(
    flux: dict[str, list[float]],
    *,
    front_threshold: float = 0.3,
) -> dict[str, Any]:
    """v1 ``surface_flux_budget``: time-bounded integration, signed ``leak_pct``."""
    times = np.asarray(flux["times"], dtype=np.float64)
    M = np.asarray(flux["M_dissolved"], dtype=np.float64)
    f_rate = np.asarray(flux["flux_rate"], dtype=np.float64)
    phi_pack = np.asarray(flux["phi_pack_rfix"], dtype=np.float64)

    if times.size < 2:
        return {
            "leak_pct": 0.0,
            "n_samples": int(times.size),
            "front_reached": False,
            "front_arrival_t": float("nan"),
            "front_arrival_idx": -1,
            "dissolved_change": 0.0,
            "flux_integrated": 0.0,
            "residual": 0.0,
            "dissolved_initial": float("nan"),
            "dissolved_at_stop": float("nan"),
            "front_threshold": float(front_threshold),
        }

    crossed = np.where(phi_pack > float(front_threshold))[0]
    if crossed.size > 0:
        front_idx = int(crossed[0])
        front_t = float(times[front_idx])
        mask = times <= front_t + 1e-9
    else:
        front_idx = -1
        front_t = float("nan")
        mask = np.ones(times.shape[0], dtype=bool)

    times_f = times[mask]
    rates_f = f_rate[mask]
    diss_f = M[mask]

    if times_f.size >= 2:
        flux_integrated = float(np.trapezoid(rates_f, x=times_f))
    elif times_f.size == 1:
        flux_integrated = float(rates_f[0] * times_f[0])
    else:
        flux_integrated = 0.0

    if diss_f.size >= 2:
        dissolved_initial = float(diss_f[0])
        dissolved_at_stop = float(diss_f[-1])
        dissolved_change = dissolved_at_stop - dissolved_initial
    else:
        dissolved_initial = float("nan")
        dissolved_at_stop = float("nan")
        dissolved_change = float("nan")

    def _abs_or_zero(x: float) -> float:
        return 0.0 if x != x else abs(x)

    residual = (
        float(dissolved_change - flux_integrated)
        if dissolved_change == dissolved_change
        else float("nan")
    )
    denom = max(
        _abs_or_zero(dissolved_initial),
        _abs_or_zero(dissolved_at_stop),
        abs(flux_integrated),
        1e-30,
    )
    leak_pct = 100.0 * residual / denom if residual == residual else float("nan")

    return {
        "leak_pct": float(leak_pct),
        "residual": float(residual) if residual == residual else float("nan"),
        "dissolved_change": float(dissolved_change)
        if dissolved_change == dissolved_change
        else float("nan"),
        "flux_integrated": float(flux_integrated),
        "dissolved_initial": dissolved_initial,
        "dissolved_at_stop": dissolved_at_stop,
        "n_samples": int(len(times_f)),
        "front_reached": bool(crossed.size > 0),
        "front_arrival_t": front_t,
        "front_threshold": float(front_threshold),
        "front_arrival_idx": int(front_idx) if crossed.size > 0 else -1,
    }


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
    if "M_total_initial" in meta and "cumulative_dirichlet_injection" in meta:
        mi = float(meta["M_total_initial"])
        mf = float(meta["M_total_final"])
        inj = float(meta["cumulative_dirichlet_injection"])
        d_m = mf - mi
        resid = d_m - inj
        # Scale by total mass so near-closed runs (inj≈0) do not report 100% on fp drift alone.
        denom = max(abs(mi), abs(mf), abs(d_m), abs(inj), 1e-30)
        out["dirichlet_mass_balance"] = {
            "M_total_initial": mi,
            "M_total_final": mf,
            "cumulative_injection": inj,
            "mass_change": d_m,
            "residual": resid,
            "residual_pct": float(100.0 * abs(resid) / denom),
            "ratio": float(d_m / inj) if abs(inj) > 1e-30 else float("nan"),
        }

    flux = meta.get("flux_samples")
    if isinstance(flux, dict) and flux.get("times"):
        sfb = _compute_surface_flux_balance(flux)
        rff = float(meta.get("option_b_r_fix_frac", 0.75))
        sfb["r_fix"] = rff * float(geom.R)
        out["surface_flux_balance"] = sfb

    smd = meta.get("spectral_mass_drift")
    if isinstance(smd, dict):
        out["spectral_mass_drift"] = dict(smd)

    return out


def simulate(
    cfg: dict[str, Any], *, chunk_size: int = 2000, show_progress: bool = True
) -> SimResult:
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

    exp = cfg["experiment"]
    gcfg = cfg["geometry"]
    logger.info(
        "Simulation start: model=%s, n=%d, T=%.1f, dt=%.4f, total_steps=%d",
        exp["model"],
        int(gcfg["n"]),
        T,
        dt,
        n_total,
    )

    seed = int(cfg.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, k_ic = jax.random.split(key)
    state = build_initial_state(cfg, geom, prm, k_ic)

    chi_np = np.asarray(jax.device_get(geom.chi))
    dx_np = float(geom.dx)
    pm0 = np.asarray(jax.device_get(state[0]))
    pc0 = np.asarray(jax.device_get(state[1]))
    c0_arr = np.asarray(jax.device_get(state[2]))
    m_total_initial = float(
        np.sum(chi_np * (c0_arr + float(prm.rho_m) * pm0 + float(prm.rho_c) * pc0)) * dx_np * dx_np
    )
    cumulative_injection = 0.0

    outcfg = cfg.get("output", {})
    flux_dt = float(outcfg.get("flux_sample_dt", 2.0))
    flux_every = max(1, int(round(flux_dt / dt)))
    r_fix_frac = float(outcfg.get("option_b_r_fix_frac", 0.75))

    meta: dict[str, Any] = {
        "flux_samples": {
            "times": [],
            "M_dissolved": [],
            "flux_rate": [],
            "phi_pack_rfix": [],
            "c_in_circle": [],
            "c_out_circle": [],
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

    wall_t0 = time.perf_counter()
    pbar = tqdm(
        total=n_total,
        desc=str(exp.get("name", "run")),
        unit="step",
        disable=not show_progress,
        leave=True,
    )
    try:
        while current_step < n_total:
            target = n_total
            if next_flux_step <= n_total:
                target = min(target, next_flux_step)
            if next_snap_step <= n_total:
                target = min(target, next_snap_step)
            n_run = min(chunk_size, target - current_step)
            if n_run <= 0:
                n_run = min(chunk_size, n_total - current_step)
            state, chunk_inj = run_chunk(state, n_run)
            cumulative_injection += float(np.asarray(jax.device_get(chunk_inj)))
            current_step += n_run
            pbar.update(n_run)
            pbar.set_postfix_str(f"t={current_step * dt:.3f}")

            logger.debug(
                "Chunk complete: steps %d/%d (t=%.2f)",
                current_step,
                n_total,
                current_step * dt,
            )

            while next_flux_step <= current_step and next_flux_step <= n_total:
                _append_flux_sample(
                    state,
                    geom,
                    meta["flux_samples"],
                    t=float(next_flux_step * dt),
                    r_fix_frac=r_fix_frac,
                    D_c=float(prm.D_c),
                )
                next_flux_step += flux_every

            while next_snap_step <= current_step and next_snap_step <= n_total:
                meta["snapshots"].append(
                    {"step": int(next_snap_step), "t": float(next_snap_step * dt)}
                )
                logger.info(
                    "Snapshot at t=%.1f (step %d)",
                    float(next_snap_step * dt),
                    int(next_snap_step),
                )
                next_snap_step += snap_every
    finally:
        pbar.close()

    t_final = float(n_total * dt)
    wall_time = time.perf_counter() - wall_t0
    logger.info(
        "Simulation complete: wall_time=%.1fs, final_t=%.1f",
        wall_time,
        t_final,
    )
    pm_f = np.asarray(jax.device_get(state[0]))
    pc_f = np.asarray(jax.device_get(state[1]))
    c_f = np.asarray(jax.device_get(state[2]))
    m_total_final = float(
        np.sum(chi_np * (c_f + float(prm.rho_m) * pm_f + float(prm.rho_c) * pc_f)) * dx_np * dx_np
    )
    meta["M_total_initial"] = m_total_initial
    meta["M_total_final"] = m_total_final
    meta["cumulative_dirichlet_injection"] = cumulative_injection

    if bool(outcfg.get("record_spectral_mass_diagnostic", False)):
        try:
            meta["spectral_mass_drift"] = run_spectral_mass_diagnostic(cfg_resolved)
        except Exception as exc:
            logger.warning("spectral mass diagnostic failed: %s", exc)

    state_final = SimState(phi_m=state[0], phi_c=state[1], c=state[2], t=t_final)
    diagnostics = _assemble_diagnostics(state, geom, prm, meta)
    return SimResult(
        state_final=state_final,
        meta=meta,
        diagnostics=diagnostics,
        config_resolved=cfg_resolved,
        paths=None,
    )
