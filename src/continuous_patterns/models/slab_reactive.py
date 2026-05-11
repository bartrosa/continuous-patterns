"""Stage I reactive Cahn--Hilliard in slab geometry (mixed BC in x/y)."""

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

from continuous_patterns.core.diagnostics_slab import compute_slab_axial_diagnostics
from continuous_patterns.core.imex import Geometry, SimParams
from continuous_patterns.core.masks import MASK_BUILDERS
from continuous_patterns.core.spectral_ops import NeumannPeriodicOps
from continuous_patterns.core.stress import (
    STRESS_BUILDERS,
    apply_pore_pressure,
    pore_pressure_field,
)
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models._integrate import make_chunk_runner
from continuous_patterns.models.cavity_reactive import (
    _append_flux_sample,
    _append_host_fields_snapshot,
    _assemble_diagnostics,
    _require,
    build_sim_params,
    run_spectral_mass_diagnostic,
)

logger = logging.getLogger(__name__)


def build_geometry(cfg: dict[str, Any]) -> Geometry:
    """Build slab ``Geometry`` from nested ``cfg['geometry']`` + ``cfg['stress']``."""
    gcfg = _require(cfg, "geometry", where="config")
    gtype = _require(gcfg, "type", where="config.geometry")
    if gtype != "slab":
        raise ValueError(f"slab_reactive requires geometry.type='slab', got {gtype!r}")

    L = float(_require(gcfg, "L", where="config.geometry"))
    n = int(_require(gcfg, "n", where="config.geometry"))
    eps_scale = float(gcfg.get("eps_scale", 2.0))
    rim_width_px = int(gcfg.get("rim_width_px", 2))
    rim_left_width_px = int(gcfg.get("rim_left_width_px", 0))

    dtype = jnp.float64 if cfg.get("precision") == "float64" else jnp.float32
    builder = MASK_BUILDERS["slab"]
    m = builder(
        L=L,
        n=n,
        rim_width_px=rim_width_px,
        rim_left_width_px=rim_left_width_px,
        eps_scale=eps_scale,
        dtype=dtype,
    )

    spectral_ops = NeumannPeriodicOps(n=n, L=L)

    st = _require(cfg, "stress", where="config")
    smode = _require(st, "mode", where="config.stress")
    if smode not in STRESS_BUILDERS:
        raise ValueError(f"Unknown stress.mode {smode!r}; allowed: {sorted(STRESS_BUILDERS)}")
    if smode in ("flamant_two_point", "kirsch", "inglis"):
        raise ValueError(f"stress.mode={smode!r} not supported for slab geometry")

    skwargs: dict[str, Any] = {"L": L, "n": n, "dtype": dtype}
    _skip = frozenset(
        {
            "mode",
            "stress_coupling_B",
            "stress_eps_factor",
            "dtype",
            "pore_pressure",
        }
    )
    _modes_use_sigma0 = frozenset(
        {
            "uniform_uniaxial",
            "uniform_biaxial",
            "pure_shear",
            "flamant_two_point",
            "pressure_gradient",
        }
    )
    for k, v in st.items():
        if k in _skip or v is None:
            continue
        if k == "sigma_0" and smode not in _modes_use_sigma0:
            continue
        skwargs.setdefault(k, v)
    if smode == "tectonic_far_field" and "theta_SH" not in skwargs:
        skwargs["theta_SH"] = float(st.get("theta_SH", 0.0))
    sxx, syy, sxy = STRESS_BUILDERS[smode](**skwargs)
    pp = st.get("pore_pressure")
    if isinstance(pp, dict) and pp:
        psp = pp.get("field", "uniform")
        p0 = float(pp["p0"])
        bi = float(pp.get("biot_alpha", 1.0))
        p_arr = pore_pressure_field(L=L, n=n, field=str(psp), p0=p0, dtype=dtype)
        sxx, syy, sxy = apply_pore_pressure(sxx, syy, sxy, p_pore=p_arr, biot_alpha=bi)

    def _to(x: Any) -> Array:
        return jnp.asarray(x, dtype=dtype)

    return Geometry(
        chi=_to(m["chi"]),
        # Slab uses a hard Dirichlet strip (right wall), so IMEX rim enforcement
        # should use the hard accounting mask rather than the smooth Gaussian.
        ring=_to(m["ring_accounting"]),
        ring_accounting=_to(m["ring_accounting"]),
        ring_left=_to(m["ring_left"]),
        sigma_xx=_to(sxx),
        sigma_yy=_to(syy),
        sigma_xy=_to(sxy),
        spectral_ops=spectral_ops,
        rv=_to(m["rv"]),
        dx=float(m["dx"]),
        L=float(m["L"]),
        R=float(m["R"]),
        n=int(m["n"]),
        xc=float(m["xc"]),
        yc=float(m["yc"]),
    )


def build_initial_state(
    cfg: dict[str, Any],
    geom: Geometry,
    prm: SimParams,
    key: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Slab IC with optional left-wall moganite nucleation seed."""
    n = geom.n
    dtype = geom.chi.dtype
    ph = cfg["physics"]
    ic = cfg.get("initial", {})

    phi_m0 = float(ic.get("phi_m_init", 0.0))
    phi_c0 = float(ic.get("phi_c_init", 0.0))
    phi_q0 = float(ic.get("phi_q_init", 0.0))
    phi_imp0 = float(ic.get("phi_imp_init", 0.0))
    noise_default = float(ic.get("phi_noise_amplitude", 0.01))
    sig_m = float(ic.get("phi_m_noise", noise_default))
    sig_c = float(ic.get("phi_c_noise", noise_default))
    sig_q = float(ic.get("phi_q_noise", noise_default))
    sig_imp = float(ic.get("phi_imp_noise", noise_default))

    k_m, k_c, k_q, k_i, _ = jax.random.split(key, 5)
    chi = geom.chi
    phi_m = (phi_m0 + sig_m * jax.random.normal(k_m, (n, n), dtype=dtype)) * chi
    phi_c = (phi_c0 + sig_c * jax.random.normal(k_c, (n, n), dtype=dtype)) * chi
    if prm.phi_q_potential.active:
        phi_q = (phi_q0 + sig_q * jax.random.normal(k_q, (n, n), dtype=dtype)) * chi
    else:
        phi_q = jnp.zeros((n, n), dtype=dtype)
    if prm.phi_imp_potential.active:
        phi_imp = (phi_imp0 + sig_imp * jax.random.normal(k_i, (n, n), dtype=dtype)) * chi
    else:
        phi_imp = jnp.zeros((n, n), dtype=dtype)

    phi_m_wall_layer = float(ic.get("phi_m_wall_layer", 0.0))
    phi_m_wall_width_px = int(ic.get("phi_m_wall_width_px", 0))
    if phi_m_wall_layer > 0.0 and phi_m_wall_width_px > 0:
        seed_mask = jnp.zeros((n, n), dtype=dtype)
        seed_mask = seed_mask.at[:phi_m_wall_width_px, :].set(1.0)
        phi_m = jnp.where(seed_mask > 0.5, jnp.asarray(phi_m_wall_layer, dtype=dtype), phi_m)

    c_sat = float(_require(ph, "c_sat", where="physics"))
    _ = float(_require(ph, "c_0", where="physics"))
    if "c_init" in ic and ic.get("c_init_factor") is not None:
        raise ValueError("initial: set at most one of c_init and c_init_factor")
    if ic.get("c_init_factor") is not None:
        c_init_interior = c_sat * float(ic["c_init_factor"])
    elif "c_init" in ic and ic["c_init"] is None:
        c_init_interior = c_sat
    elif "c_init" in ic:
        c_init_interior = float(ic["c_init"])
    else:
        c_init_interior = c_sat
    c = c_init_interior * chi
    c = jnp.asarray(c, dtype=dtype)
    return phi_m, phi_c, phi_q, phi_imp, c


def simulate(
    cfg: dict[str, Any], *, chunk_size: int = 2000, show_progress: bool = True
) -> SimResult:
    """Run slab Stage I to ``time.T`` with chunked ``fori_loop`` integration."""
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

    exp_seed = cfg.get("experiment", {})
    if isinstance(exp_seed, dict) and "seed" in exp_seed:
        seed = int(exp_seed["seed"])
    else:
        seed = int(cfg.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, k_ic = jax.random.split(key)
    state = build_initial_state(cfg, geom, prm, k_ic)

    chi_np = np.asarray(jax.device_get(geom.chi))
    dx_np = float(geom.dx)
    pm0 = np.asarray(jax.device_get(state[0]))
    pc0 = np.asarray(jax.device_get(state[1]))
    pq0 = np.asarray(jax.device_get(state[2]))
    pim0 = np.asarray(jax.device_get(state[3]))
    c0_arr = np.asarray(jax.device_get(state[4]))
    m_total_initial = float(
        np.sum(
            chi_np
            * (
                c0_arr
                + float(prm.phi_m_potential.rho) * pm0
                + float(prm.phi_c_potential.rho) * pc0
                + float(prm.phi_q_potential.rho) * pq0
                + float(prm.phi_imp_potential.rho) * pim0
            )
        )
        * dx_np
        * dx_np
    )
    cumulative_injection = 0.0

    outcfg = cfg.get("output", {})
    flux_dt = float(outcfg.get("flux_sample_dt", 2.0))
    flux_every = max(1, int(round(flux_dt / dt)))
    r_fix_frac = float(outcfg.get("option_b_r_fix_frac", 0.75))

    save_h5 = bool(outcfg.get("save_snapshots_h5", False))
    record_gif = bool(outcfg.get("record_evolution_gif", False))
    need_snapshots = save_h5 or record_gif

    h5_snapshots: list[dict[str, Any]] = []
    gif_snapshots: list[tuple[float, np.ndarray]] = []

    meta: dict[str, Any] = {
        "effective_cavity_R": float(geom.R),
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
    if save_h5:
        meta["h5_snapshots"] = h5_snapshots
    if record_gif:
        meta["gif_snapshots"] = gif_snapshots

    snap_every = int(tcfg.get("snapshot_every", 10**9))
    if snap_every < 1:
        snap_every = 1

    eff_chunk = min(chunk_size, snap_every) if need_snapshots else chunk_size
    if need_snapshots and eff_chunk < chunk_size:
        n_expect = n_total // snap_every + 1 + (1 if n_total % snap_every != 0 else 0)
        logger.info(
            "Snapshots enabled: sub-chunk size reduced from %d to %d (snapshot_every=%d). "
            "Roughly %d snapshot times over %d total steps.",
            chunk_size,
            eff_chunk,
            snap_every,
            n_expect,
            n_total,
        )

    run_chunk = make_chunk_runner(geom, prm, dt)

    current_step = 0
    next_flux_step = flux_every
    next_snap_step = snap_every

    if need_snapshots:
        _append_host_fields_snapshot(
            state,
            step=0,
            t=0.0,
            save_h5=save_h5,
            record_gif=record_gif,
            h5_list=h5_snapshots if save_h5 else None,
            gif_list=gif_snapshots if record_gif else None,
        )
        meta["snapshots"].append({"step": 0, "t": 0.0})

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
            if not need_snapshots and next_snap_step <= n_total:
                target = min(target, next_snap_step)
            n_run = min(eff_chunk, target - current_step)
            if n_run <= 0:
                n_run = min(eff_chunk, n_total - current_step)
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

            if need_snapshots and current_step > 0:
                if current_step % snap_every == 0 or current_step == n_total:
                    _append_host_fields_snapshot(
                        state,
                        step=current_step,
                        t=float(current_step * dt),
                        save_h5=save_h5,
                        record_gif=record_gif,
                        h5_list=h5_snapshots if save_h5 else None,
                        gif_list=gif_snapshots if record_gif else None,
                    )
                    meta["snapshots"].append(
                        {"step": int(current_step), "t": float(current_step * dt)}
                    )
                    logger.info(
                        "Snapshot at t=%.1f (step %d)",
                        float(current_step * dt),
                        int(current_step),
                    )
            else:
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
    pq_f = np.asarray(jax.device_get(state[2]))
    pim_f = np.asarray(jax.device_get(state[3]))
    c_f = np.asarray(jax.device_get(state[4]))
    m_total_final = float(
        np.sum(
            chi_np
            * (
                c_f
                + float(prm.phi_m_potential.rho) * pm_f
                + float(prm.phi_c_potential.rho) * pc_f
                + float(prm.phi_q_potential.rho) * pq_f
                + float(prm.phi_imp_potential.rho) * pim_f
            )
        )
        * dx_np
        * dx_np
    )
    meta["M_total_initial"] = m_total_initial
    meta["M_total_final"] = m_total_final
    meta["cumulative_dirichlet_injection"] = cumulative_injection

    if bool(outcfg.get("record_spectral_mass_diagnostic", False)):
        try:
            meta["spectral_mass_drift"] = run_spectral_mass_diagnostic(cfg_resolved)
        except Exception as exc:
            logger.warning("spectral mass diagnostic failed: %s", exc)

    state_final = SimState(
        phi_m=state[0],
        phi_c=state[1],
        phi_q=state[2],
        phi_imp=state[3],
        c=state[4],
        t=t_final,
    )
    diagnostics = _assemble_diagnostics(state, geom, prm, meta)
    diagnostics["slab_axial"] = compute_slab_axial_diagnostics(
        np.asarray(state[0]),
        np.asarray(state[1]),
        L=float(geom.L),
    )
    return SimResult(
        state_final=state_final,
        meta=meta,
        diagnostics=diagnostics,
        config_resolved=cfg_resolved,
        paths=None,
    )
