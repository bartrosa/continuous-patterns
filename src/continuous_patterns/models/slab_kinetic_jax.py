"""JAX backend for the 1D slab kinetic model (batched ``lax.scan`` + ``vmap``).

Physics, discretization, and diagnostics definitions match
``continuous_patterns.models.slab_kinetic`` (NumPy reference). This module is
imported only when ``solver.backend='jax'`` is requested.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from continuous_patterns.core.diagnostics_kinetic import (
    compute_mean_dwell_period,
    compute_temporal_spectrum,
)
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models.slab_kinetic import (
    KineticParams,
    build_geometry,
    build_initial_state,
    build_params,
    build_time_params,
)

logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)

TARGET_HISTORY_POINTS = 50_000
DEFAULT_MAX_TRANSITIONS = 200_000


@dataclass(frozen=True)
class SolverJAXOptions:
    """JAX-specific solver knobs from ``cfg['solver']``."""

    history_every: int
    max_transitions: int


@dataclass(frozen=True)
class BatchParams:
    """Batched physical parameters for ``simulate_batch``."""

    Da: np.ndarray
    kappa: np.ndarray
    c1: np.ndarray
    c2: np.ndarray


def _solver_options(cfg: dict[str, Any], *, n_steps: int) -> SolverJAXOptions:
    solver = cfg.get("solver") if isinstance(cfg.get("solver"), dict) else {}
    hist_raw = solver.get("history_every")
    if hist_raw is None:
        history_every = max(1, int(n_steps) // TARGET_HISTORY_POINTS)
    else:
        history_every = max(1, int(hist_raw))
    max_trans = int(solver.get("max_transitions", DEFAULT_MAX_TRANSITIONS))
    if max_trans < 1:
        raise ValueError("solver.max_transitions must be >= 1")
    return SolverJAXOptions(history_every=history_every, max_transitions=max_trans)


def _integration_grid(cfg: dict[str, Any]) -> tuple[KineticParams, float, float, int, np.ndarray]:
    geom = build_geometry(cfg)
    prm = build_params(cfg)
    tm = build_time_params(cfg)
    dx = float(geom.dx)
    dt = float(tm.dt_safety) * dx * dx / 2.0
    n_steps = int(np.floor(float(tm.T) / dt + 1e-15))
    if n_steps < 1:
        n_steps = 1
    x = np.asarray(geom.x, dtype=np.float64)
    return prm, dt, float(tm.T), n_steps, x


def _transitions_from_buffers(
    trans_times: np.ndarray,
    trans_dirs: np.ndarray,
    trans_count: int,
) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    n = int(min(trans_count, trans_times.size))
    for i in range(n):
        tv = float(trans_times[i])
        code = int(trans_dirs[i])
        kind = "L→H" if code == 0 else "H→L"
        out.append((tv, kind))
    return out


def _dwell_stats_from_accumulators(
    *,
    sum_L: float,
    sumsq_L: float,
    count_L: int,
    sum_H: float,
    sumsq_H: float,
    count_H: int,
    n_transitions: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    if count_L > 0:
        dwell_L_mean = float(sum_L / count_L)
        dwell_L_std = float(np.sqrt(max(sumsq_L / count_L - dwell_L_mean**2, 0.0)))
        dwell_L = np.full(count_L, dwell_L_mean, dtype=np.float64)
    else:
        dwell_L_mean = float("nan")
        dwell_L_std = float("nan")
        dwell_L = np.asarray([], dtype=np.float64)

    if count_H > 0:
        dwell_H_mean = float(sum_H / count_H)
        dwell_H_std = float(np.sqrt(max(sumsq_H / count_H - dwell_H_mean**2, 0.0)))
        dwell_H = np.full(count_H, dwell_H_mean, dtype=np.float64)
    else:
        dwell_H_mean = float("nan")
        dwell_H_std = float("nan")
        dwell_H = np.asarray([], dtype=np.float64)

    mean_period = compute_mean_dwell_period(dwell_L, dwell_H, n_transitions)  # noqa: F841
    return dwell_L_mean, dwell_L_std, dwell_H_mean, dwell_H_std, dwell_L, dwell_H


def _make_batched_runner(
    *,
    n: int,
    dx: float,
    dt: float,
    n_steps: int,
    history_every: int,
    max_transitions: int,
    x_init: np.ndarray,
):
    """Build and return a JIT-compiled ``vmap`` runner."""

    def _single(Da, kappa, c1, c2, c_init):
        n_chunks = max(1, n_steps // history_every)
        remainder = n_steps - n_chunks * history_every

        def _apply_dwell_close(carry, *, seg_dt, did_trans):
            (
                dwell_state,
                _dwell_start_t,
                sum_L,
                sumsq_L,
                count_L,
                sum_H,
                sumsq_H,
                count_H,
            ) = carry
            in_L = dwell_state == 0
            in_H = dwell_state == 1
            sum_L = sum_L + jnp.where(did_trans & in_L, seg_dt, 0.0)
            sumsq_L = sumsq_L + jnp.where(did_trans & in_L, seg_dt * seg_dt, 0.0)
            count_L = count_L + jnp.where(did_trans & in_L, 1, 0)
            sum_H = sum_H + jnp.where(did_trans & in_H, seg_dt, 0.0)
            sumsq_H = sumsq_H + jnp.where(did_trans & in_H, seg_dt * seg_dt, 0.0)
            count_H = count_H + jnp.where(did_trans & in_H, 1, 0)
            return (sum_L, sumsq_L, count_L, sum_H, sumsq_H, count_H)

        def _integration_step(carry, _unused):
            (
                c,
                state,
                t,
                thickness,
                trans_count,
                trans_times,
                trans_dirs,
                overflow,
                dwell_state,
                dwell_start_t,
                sum_L,
                sumsq_L,
                count_L,
                sum_H,
                sumsq_H,
                count_H,
            ) = carry

            r_robin = jnp.where(state == 1, kappa, 1.0)
            lap = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (dx * dx)
            c_new = c.at[1:-1].set(c[1:-1] + dt * lap)
            c_new = c_new.at[-1].set(1.0)
            c_new = c_new.at[0].set(c_new[1] / (1.0 + r_robin * Da * dx))

            c0 = c_new[0]
            to_H = (state == 0) & (c0 > c1)
            to_L = (state == 1) & (c0 < c2)
            did_trans = to_H | to_L
            new_state = jnp.where(to_H, 1, jnp.where(to_L, 0, state))

            t_new = t + dt
            can_record = did_trans & (trans_count < max_transitions)
            would_overflow = did_trans & (trans_count >= max_transitions)
            new_overflow = overflow | would_overflow

            dir_code = jnp.where(to_H, 0, jnp.where(to_L, 1, -1))
            new_trans_times = trans_times.at[trans_count].set(t_new)
            new_trans_times = jnp.where(can_record, new_trans_times, trans_times)
            new_trans_dirs = trans_dirs.at[trans_count].set(dir_code)
            new_trans_dirs = jnp.where(can_record, new_trans_dirs, trans_dirs)
            new_trans_count = trans_count + jnp.where(can_record, 1, 0)

            seg_dt = t_new - dwell_start_t
            sum_L, sumsq_L, count_L, sum_H, sumsq_H, count_H = _apply_dwell_close(
                (dwell_state, dwell_start_t, sum_L, sumsq_L, count_L, sum_H, sumsq_H, count_H),
                seg_dt=seg_dt,
                did_trans=did_trans,
            )
            new_dwell_state = jnp.where(did_trans, 1 - dwell_state, dwell_state)
            new_dwell_start_t = jnp.where(did_trans, t_new, dwell_start_t)

            r_thick = jnp.where(new_state == 1, kappa, 1.0)
            new_thickness = thickness + dt * r_thick * c0

            return (
                c_new,
                new_state,
                t_new,
                new_thickness,
                new_trans_count,
                new_trans_times,
                new_trans_dirs,
                new_overflow,
                new_dwell_state,
                new_dwell_start_t,
                sum_L,
                sumsq_L,
                count_L,
                sum_H,
                sumsq_H,
                count_H,
            ), None

        def _chunk_step(carry, _unused):
            carry, _ = jax.lax.scan(_integration_step, carry, None, length=history_every)
            emit = (carry[2], carry[0][0], carry[1], carry[3])
            return carry, emit

        init_carry = (
            c_init,
            jnp.int32(0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.int32(0),
            jnp.zeros(max_transitions, dtype=jnp.float64),
            jnp.full(max_transitions, -1, dtype=jnp.int32),
            jnp.bool_(False),
            jnp.int32(0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.int32(0),
            jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.int32(0),
        )

        carry, hist = jax.lax.scan(_chunk_step, init_carry, None, length=n_chunks)
        carry = jax.lax.cond(
            remainder > 0,
            lambda c: jax.lax.scan(_integration_step, c, None, length=remainder)[0],
            lambda c: c,
            carry,
        )

        (
            c_final,
            state_final,
            t_final,
            thickness_final,
            trans_count,
            trans_times,
            trans_dirs,
            overflow,
            dwell_state,
            dwell_start_t,
            sum_L,
            sumsq_L,
            count_L,
            sum_H,
            sumsq_H,
            count_H,
        ) = carry

        tail = t_final - dwell_start_t
        in_L = dwell_state == 0
        in_H = dwell_state == 1
        sum_L = sum_L + jnp.where(in_L, tail, 0.0)
        sumsq_L = sumsq_L + jnp.where(in_L, tail * tail, 0.0)
        count_L = count_L + jnp.where(in_L, 1, 0)
        sum_H = sum_H + jnp.where(in_H, tail, 0.0)
        sumsq_H = sumsq_H + jnp.where(in_H, tail * tail, 0.0)
        count_H = count_H + jnp.where(in_H, 1, 0)

        return {
            "c_final": c_final,
            "state_final": state_final,
            "t_final": t_final,
            "thickness_final": thickness_final,
            "trans_count": trans_count,
            "trans_times": trans_times,
            "trans_dirs": trans_dirs,
            "overflow": overflow,
            "sum_L": sum_L,
            "sumsq_L": sumsq_L,
            "count_L": count_L,
            "sum_H": sum_H,
            "sumsq_H": sumsq_H,
            "count_H": count_H,
            "hist_t": hist[0],
            "hist_c0": hist[1],
            "hist_state": hist[2],
            "hist_thickness": hist[3],
        }

    batched = jax.vmap(_single, in_axes=(0, 0, 0, 0, None))
    return jax.jit(batched)


def _pack_sim_result(
    cfg: dict[str, Any],
    *,
    prm: KineticParams,
    dt: float,
    n_steps: int,
    opts: SolverJAXOptions,
    raw: dict[str, np.ndarray | float | bool],
    wall_s: float,
) -> SimResult:
    geom = build_geometry(cfg)
    tm = build_time_params(cfg)

    c_final = np.asarray(raw["c_final"], dtype=np.float64)
    t_final = float(raw["t_final"])
    thickness_final = float(raw["thickness_final"])
    trans_count = int(raw["trans_count"])
    overflow = bool(raw["overflow"])

    trans_times = np.asarray(raw["trans_times"], dtype=np.float64)
    trans_dirs = np.asarray(raw["trans_dirs"], dtype=np.int32)
    transitions = _transitions_from_buffers(trans_times, trans_dirs, trans_count)

    dwell_L_mean, dwell_L_std, dwell_H_mean, dwell_H_std, dwell_L, dwell_H = (
        _dwell_stats_from_accumulators(
            sum_L=float(raw["sum_L"]),
            sumsq_L=float(raw["sumsq_L"]),
            count_L=int(raw["count_L"]),
            sum_H=float(raw["sum_H"]),
            sumsq_H=float(raw["sumsq_H"]),
            count_H=int(raw["count_H"]),
            n_transitions=trans_count,
        )
    )

    hist_t = np.asarray(raw["hist_t"], dtype=np.float64).ravel()
    hist_c0 = np.asarray(raw["hist_c0"], dtype=np.float64).ravel()
    hist_state = np.asarray(raw["hist_state"], dtype=np.int32).ravel()
    hist_th = np.asarray(raw["hist_thickness"], dtype=np.float64).ravel()

    spec = compute_temporal_spectrum(hist_c0, dt * opts.history_every)
    mean_period = compute_mean_dwell_period(dwell_L, dwell_H, trans_count)

    state_final_str = "H" if int(raw["state_final"]) == 1 else "L"
    transitions_json = [{"t": float(tv), "kind": str(k)} for tv, k in transitions]

    diagnostics: dict[str, Any] = {
        "n_transitions": trans_count,
        "dwell_L_mean": dwell_L_mean,
        "dwell_L_std": dwell_L_std,
        "dwell_H_mean": dwell_H_mean,
        "dwell_H_std": dwell_H_std,
        "peak_period": float(spec["peak_period"]),
        "peak_frequency": float(spec["peak_freq"]),
        "mean_dwell_period": float(mean_period["mean_dwell_period"]),
        "mean_dwell_freq": float(mean_period["mean_dwell_freq"]),
        "thickness_final": thickness_final,
        "wall_time_s": float(wall_s),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "T": float(tm.T),
        "kinetic_transitions": transitions_json,
        "dwell_L_samples": dwell_L.tolist(),
        "dwell_H_samples": dwell_H.tolist(),
        "jax_backend": True,
        "jax_transition_overflow": overflow,
        "jax_history_every": int(opts.history_every),
    }

    zeros2 = jnp.zeros((1, 1))
    state_final = SimState(
        phi_m=zeros2,
        phi_c=zeros2,
        phi_q=zeros2,
        phi_imp=zeros2,
        c=jnp.asarray(c_final[:, None]),
        t=t_final,
    )

    meta_block: dict[str, Any] = {
        "geometry": {"type": "kinetic_1d", "n": int(geom.n)},
        "params": {
            "Da": float(prm.Da),
            "kappa": float(prm.kappa),
            "c1": float(prm.c1),
            "c2": float(prm.c2),
        },
        "time": {"T": float(tm.T), "dt": float(dt), "dt_safety": float(tm.dt_safety)},
        "t_history": hist_t,
        "c_at_x0_history": hist_c0,
        "state_history": hist_state,
        "thickness_history": hist_th,
        "snapshots_t": np.asarray([], dtype=np.float64),
        "snapshots_c": np.zeros((0, geom.n), dtype=np.float64),
        "transitions": transitions,
        "temporal_spectrum": {k: v for k, v in spec.items()},
        "state_final_str": state_final_str,
        "thickness_final": thickness_final,
        "t_final": t_final,
        "jax_transition_overflow": overflow,
    }

    return SimResult(
        state_final=state_final,
        meta={"slab_kinetic": meta_block},
        diagnostics=diagnostics,
        config_resolved=dict(cfg),
    )


def _raw_to_numpy(raw_batched: dict[str, Any], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in raw_batched.items():
        arr = np.asarray(val)
        if arr.ndim == 0:
            out[key] = arr.item()
        else:
            out[key] = arr[index]
    return out


def simulate_jax(
    cfg: dict[str, Any],
    *,
    show_progress: bool = False,  # noqa: ARG001 — API parity with NumPy backend
) -> SimResult:
    """Run a single ``slab_kinetic`` simulation on the JAX backend."""
    prm, dt, _T, n_steps, x = _integration_grid(cfg)
    opts = _solver_options(cfg, n_steps=n_steps)
    ic = build_initial_state(cfg, build_geometry(cfg), prm)
    c_init = jnp.asarray(ic.c, dtype=jnp.float64)

    runner = _make_batched_runner(
        n=int(build_geometry(cfg).n),
        dx=float(build_geometry(cfg).dx),
        dt=float(dt),
        n_steps=int(n_steps),
        history_every=int(opts.history_every),
        max_transitions=int(opts.max_transitions),
        x_init=x,
    )

    t0 = time.perf_counter()
    raw = runner(
        jnp.asarray([prm.Da]),
        jnp.asarray([prm.kappa]),
        jnp.asarray([prm.c1]),
        jnp.asarray([prm.c2]),
        c_init,
    )
    raw = jax.block_until_ready(raw)
    wall_s = time.perf_counter() - t0

    single = _raw_to_numpy(raw, 0)
    return _pack_sim_result(
        cfg,
        prm=prm,
        dt=dt,
        n_steps=n_steps,
        opts=opts,
        raw=single,
        wall_s=wall_s,
    )


def simulate_batch(
    cfg: dict[str, Any],
    batch: BatchParams,
    *,
    compile_only: bool = False,
) -> tuple[list[SimResult], dict[str, float]]:
    """Run a batched simulation; returns results and ``{compile_s, exec_s}`` timing."""
    prm0, dt, _T, n_steps, x = _integration_grid(cfg)
    opts = _solver_options(cfg, n_steps=n_steps)
    geom = build_geometry(cfg)
    ic = build_initial_state(cfg, geom, prm0)
    c_init = jnp.asarray(ic.c, dtype=jnp.float64)

    runner = _make_batched_runner(
        n=int(geom.n),
        dx=float(geom.dx),
        dt=float(dt),
        n_steps=int(n_steps),
        history_every=int(opts.history_every),
        max_transitions=int(opts.max_transitions),
        x_init=x,
    )

    Da = jnp.asarray(batch.Da, dtype=jnp.float64)
    kappa = jnp.asarray(batch.kappa, dtype=jnp.float64)
    c1 = jnp.asarray(batch.c1, dtype=jnp.float64)
    c2 = jnp.asarray(batch.c2, dtype=jnp.float64)

    t_compile0 = time.perf_counter()
    _ = runner.lower(Da, kappa, c1, c2, c_init).compile()
    compile_s = time.perf_counter() - t_compile0

    if compile_only:
        return [], {"compile_s": compile_s, "exec_s": 0.0}

    t_exec0 = time.perf_counter()
    raw = runner(Da, kappa, c1, c2, c_init)
    raw = jax.block_until_ready(raw)
    exec_s = time.perf_counter() - t_exec0

    results: list[SimResult] = []
    b = int(batch.Da.size)
    for i in range(b):
        prm_i = KineticParams(
            Da=float(batch.Da[i]),
            kappa=float(batch.kappa[i]),
            c1=float(batch.c1[i]),
            c2=float(batch.c2[i]),
        )
        cfg_i = dict(cfg)
        cfg_i.setdefault("physics", {})
        cfg_i["physics"] = {
            **cfg_i["physics"],
            "Da": prm_i.Da,
            "kappa": prm_i.kappa,
            "c1": prm_i.c1,
            "c2": prm_i.c2,
        }
        single = _raw_to_numpy(raw, i)
        results.append(
            _pack_sim_result(
                cfg_i,
                prm=prm_i,
                dt=dt,
                n_steps=n_steps,
                opts=opts,
                raw=single,
                wall_s=exec_s,
            )
        )

    logger.info(
        "slab_kinetic_jax batch: B=%d compile=%.2fs exec=%.2fs history_every=%d",
        b,
        compile_s,
        exec_s,
        opts.history_every,
    )
    return results, {"compile_s": compile_s, "exec_s": exec_s}
