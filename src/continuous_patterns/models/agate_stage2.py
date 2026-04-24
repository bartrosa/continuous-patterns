"""Stage II — bulk periodic CH relaxation (no reaction, no rim Dirichlet).

Uses the same :func:`continuous_patterns.core.imex.imex_step` with
``reaction_active=False`` and ``dirichlet_active=False``. Post-processing uses
``diagnostics_stage2`` only (``docs/ARCHITECTURE.md`` §4.2).

Differences from Stage I (:mod:`continuous_patterns.models.agate_ch`):

- **No** ``flux_samples`` in ``meta`` (no rim / Option B dense path).
- **Diagnostics** are bulk-only: structure factor, coarsening / interface metrics —
  no ``option_b_leak_pct``, Jabłczyński, or cavity-only scalars at top level.
- **Initial phases** are **not** ``χ``-masked (full torus).
- ``c`` is inert when ``G=0`` but still evolved by diffusion if ``D_c>0``; default
  IC is uniform zeros or ``physics.c_0``.
"""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from continuous_patterns.core.diagnostics_stage2 import (
    bulk_scalar_stats,
    coarsening_metrics,
    interface_density,
    structure_factor_radial_average,
)
from continuous_patterns.core.imex import Geometry, SimParams
from continuous_patterns.core.spectral import k_vectors
from continuous_patterns.core.stress import STRESS_BUILDERS
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models._integrate import make_chunk_runner


def _require(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key {key!r} in {where}")
    return mapping[key]


def _bulk_geometry_arrays(
    *,
    L: float,
    n: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """``χ≡1``, zero rim masks, ``rv`` from domain centre (for API consistency)."""
    dx = L / n
    xc = 0.5 * L
    yc = 0.5 * L
    ii = jnp.arange(n, dtype=dtype)[:, None]
    jj = jnp.arange(n, dtype=dtype)[None, :]
    x_cent = jnp.broadcast_to((ii + 0.5) * dx, (n, n))
    y_cent = jnp.broadcast_to((jj + 0.5) * dx, (n, n))
    rv = jnp.sqrt((x_cent - xc) ** 2 + (y_cent - yc) ** 2)
    chi = jnp.ones((n, n), dtype=dtype)
    z = jnp.zeros((n, n), dtype=dtype)
    return chi, z, z, rv


def build_geometry(cfg: dict[str, Any]) -> Geometry:
    """Bulk periodic ``Geometry``: ``χ≡1``, no rim, ``R=0``, stress from config."""
    gcfg = _require(cfg, "geometry", where="config")
    L = float(_require(gcfg, "L", where="config.geometry"))
    n = int(_require(gcfg, "n", where="config.geometry"))
    # ``geometry.type`` / ``R`` are ignored for bulk Stage II (no cavity semantics).

    dtype = jnp.float64 if cfg.get("precision") == "float64" else jnp.float32
    chi, ring, ring_accounting, rv = _bulk_geometry_arrays(L=L, n=n, dtype=dtype)

    k_sq, kx_sq, ky_sq, kx_wave, ky_wave, k_four = k_vectors(L=L, n=n)

    st = _require(cfg, "stress", where="config")
    smode = _require(st, "mode", where="config.stress")
    if smode not in STRESS_BUILDERS:
        raise ValueError(f"Unknown stress.mode {smode!r}; allowed: {sorted(STRESS_BUILDERS)}")
    skwargs: dict[str, Any] = {"L": L, "n": n, "dtype": dtype}
    # ``stress_eps_factor`` only used by ``flamant_two_point`` (see agate_ch ``build_geometry``).
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

    xc = 0.5 * L
    yc = 0.5 * L
    return Geometry(
        chi=chi,
        ring=ring,
        ring_accounting=ring_accounting,
        sigma_xx=_to(sxx),
        sigma_yy=_to(syy),
        sigma_xy=_to(sxy),
        k_sq=jnp.asarray(k_sq, dtype=dtype),
        kx_sq=jnp.asarray(kx_sq, dtype=dtype),
        ky_sq=jnp.asarray(ky_sq, dtype=dtype),
        kx_wave=jnp.asarray(kx_wave, dtype=dtype),
        ky_wave=jnp.asarray(ky_wave, dtype=dtype),
        k_four=jnp.asarray(k_four, dtype=dtype),
        rv=rv,
        dx=L / n,
        L=L,
        R=0.0,
        n=n,
        xc=float(xc),
        yc=float(yc),
    )


def build_sim_params(cfg: dict[str, Any]) -> SimParams:
    """Stage II: ``reaction_active`` and ``dirichlet_active`` are forced ``False``."""
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

    return SimParams(
        reaction_active=False,
        dirichlet_active=False,
        D_c=float(_require(ph, "D_c", where="physics")),
        M_m=float(_require(ph, "M_m", where="physics")),
        M_c=float(_require(ph, "M_c", where="physics")),
        W=float(_require(ph, "W", where="physics")),
        gamma=float(_require(ph, "gamma", where="physics")),
        kappa_x=kappa_x,
        kappa_y=kappa_y,
        stress_coupling_B=float(st.get("stress_coupling_B", 0.0)),
        k_rxn=float(ph.get("k_rxn", 0.0)),
        c_sat=float(ph.get("c_sat", 0.0)),
        rho_m=float(ph.get("rho_m", 1.0)),
        rho_c=float(ph.get("rho_c", 1.0)),
        c0=float(_require(ph, "c_0", where="physics")),
        lambda_bar=float(lambda_bar),
        c_ostwald=float(_require(ph, "c_ostwald", where="physics")),
        w_ostwald=float(_require(ph, "w_ostwald", where="physics")),
        use_ratchet=bool(ph.get("use_ratchet", False)),
        phi_m_ratchet_low=float(ph.get("phi_m_ratchet_low", 0.3)),
        phi_m_ratchet_high=float(ph.get("phi_m_ratchet_high", 0.5)),
    )


def build_initial_state(
    cfg: dict[str, Any],
    geom: Geometry,
    prm: SimParams,
    key: Array,
) -> SimState:
    """Bulk IC: mixed phases on full torus; ``c`` uniform (default zeros).

    Future: optional ``cfg['initial']['from_npz']`` to continue from Stage I
    ``final_state.npz`` — not implemented.
    """
    _ = prm
    if cfg.get("initial", {}).get("from_npz"):
        raise NotImplementedError("initial.from_npz for Stage II is not implemented yet.")

    n = geom.n
    dtype = geom.chi.dtype
    ph = cfg["physics"]
    ic = cfg.get("initial", {})

    phi_m0 = float(ic.get("phi_m_init", 0.5))
    phi_c0 = float(ic.get("phi_c_init", 0.5))
    sig_m = float(ic.get("phi_m_noise", 0.01))
    sig_c = float(ic.get("phi_c_noise", 0.01))

    k1, k2 = jax.random.split(key)
    phi_m = phi_m0 + sig_m * jax.random.normal(k1, (n, n), dtype=dtype)
    phi_c = phi_c0 + sig_c * jax.random.normal(k2, (n, n), dtype=dtype)
    c0 = float(ic.get("c_init", ph.get("c_0", 0.0)))
    c = jnp.full((n, n), c0, dtype=dtype)
    return SimState(phi_m=phi_m, phi_c=phi_c, c=c, t=0.0)


def _append_snapshot_bulk(
    meta_snapshots: list[dict[str, Any]],
    state: tuple[Array, Array, Array],
    *,
    step: int,
    t: float,
    L: float,
) -> None:
    pm = np.asarray(state[0])
    pc = np.asarray(state[1])
    meta_snapshots.append({"step": step, "t": t, "bulk_stats": bulk_scalar_stats(pm, pc)})


def _assemble_diagnostics_s2(
    state: tuple[Array, Array, Array],
    geom: Geometry,
) -> dict[str, Any]:
    pm = np.asarray(state[0])
    pc = np.asarray(state[1])
    L = float(geom.L)
    psi = pm - pc
    return {
        "structure_factor": structure_factor_radial_average(psi, L=L),
        "bulk_stats_final": bulk_scalar_stats(pm, pc),
        "interface_density": interface_density(pm, pc, L=L),
        "coarsening_metrics": coarsening_metrics(pm, pc, L=L),
    }


def simulate(cfg: dict[str, Any], *, chunk_size: int = 2000) -> SimResult:
    """Run Stage II bulk relaxation to ``time.T`` (no rim flux bookkeeping)."""
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
    ic = build_initial_state(cfg, geom, prm, k_ic)
    state = (ic.phi_m, ic.phi_c, ic.c)

    meta: dict[str, Any] = {
        "chunk_size": chunk_size,
        "n_steps": n_total,
        "snapshots": [],
        "stage": "agate_stage2",
    }

    snap_every = int(tcfg.get("snapshot_every", 10**9))
    if snap_every < 1:
        snap_every = 1

    run_chunk = make_chunk_runner(geom, prm, dt)

    current_step = 0
    next_snap_step = snap_every

    while current_step < n_total:
        target = n_total
        if next_snap_step <= n_total:
            target = min(target, next_snap_step)
        n_run = min(chunk_size, target - current_step)
        if n_run <= 0:
            n_run = min(chunk_size, n_total - current_step)
        state = run_chunk(state, n_run)
        current_step += n_run

        while next_snap_step <= current_step and next_snap_step <= n_total:
            _append_snapshot_bulk(
                meta["snapshots"],
                state,
                step=int(next_snap_step),
                t=float(next_snap_step * dt),
                L=float(geom.L),
            )
            next_snap_step += snap_every

    t_final = float(n_total * dt)
    state_final = SimState(phi_m=state[0], phi_c=state[1], c=state[2], t=t_final)
    diagnostics = _assemble_diagnostics_s2(state, geom)
    return SimResult(
        state_final=state_final,
        meta=meta,
        diagnostics=diagnostics,
        config_resolved=cfg_resolved,
        paths=None,
    )
