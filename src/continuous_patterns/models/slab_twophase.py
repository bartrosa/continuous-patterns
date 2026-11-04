"""1D two-phase slab model — verified reference kernel integration (v6).

Growth, transport, and demixing numerics match ``twophase_kernel_reference.py``
verbatim (equations, stencils, gating, mobility, spectral CH dt, update order).
Only plumbing differs: Pydantic config, ``SimResult`` outputs, evolution frames, tqdm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from continuous_patterns.core.banding_metrics import compute_clip_rate
from continuous_patterns.core.types import SimResult, SimState
from continuous_patterns.models.twophase_kernel_d1 import (
    c_dep_at_active_front,
    run_driven,
    run_hysteresis,
)
from continuous_patterns.models.twophase_kernel_reference import run as _kernel_run

logger = logging.getLogger(__name__)

DemixingMode = Literal["ac", "ch"]

# Exchange channel only uses gamma·φ² cross terms (W=0); not user-configurable.
KERNEL_W = 0.0
DEFAULT_PHI_GATE = 0.02


@dataclass(frozen=True)
class TwophaseGeometry:
    L: float
    n: int
    dx: float
    x: np.ndarray
    seed_width_px: int


@dataclass(frozen=True)
class TwophaseParams:
    demixing: DemixingMode
    gamma: float
    D_c: float
    k_rxn: float
    c_sat: float
    c_0: float
    c_ostwald: float
    w_ostwald: float
    bc_right: str
    rho: float
    phi_gate: float
    kappa_s: float
    L_s: float
    kappa: float
    M0: float


@dataclass(frozen=True)
class TwophaseTimeParams:
    T: float
    dt_safety: float
    snapshot_every: int


def ch_spinodal_wavelength(*, gamma: float, kappa: float) -> float:
    """Fastest-growing CH wavelength for ``gamma·φ_m²·φ_c²`` exchange about φ_m ≈ φ_c ≈ 0.5."""
    if gamma <= 0.0:
        return float("nan")
    return float(2.0 * np.pi * np.sqrt(2.0 * kappa / gamma))


def build_geometry(cfg: dict[str, Any]) -> TwophaseGeometry:
    g = cfg.get("geometry") or {}
    L = float(g.get("L", 100.0))
    n = int(g["n"])
    if n < 3:
        raise ValueError("slab_twophase requires geometry.n >= 3")
    dx = L / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    seed_width_px = int(g.get("seed_width_px", 2))
    return TwophaseGeometry(L=L, n=n, dx=dx, x=x, seed_width_px=seed_width_px)


def build_params(cfg: dict[str, Any]) -> TwophaseParams:
    p = cfg.get("physics") or {}
    bc = str(p.get("bc_right", "dirichlet")).lower()
    if bc not in ("dirichlet", "noflux"):
        raise ValueError("physics.bc_right must be 'dirichlet' or 'noflux'")
    demixing = str(p.get("demixing", "ac")).lower()
    if demixing not in ("ac", "ch"):
        raise ValueError("physics.demixing must be 'ac' or 'ch'")
    phi_gate = float(p.get("phi_gate", DEFAULT_PHI_GATE))
    base = dict(
        demixing=demixing,  # type: ignore[arg-type]
        gamma=float(p["gamma"]),
        D_c=float(p["D_c"]),
        k_rxn=float(p["k_rxn"]),
        c_sat=float(p["c_sat"]),
        c_0=float(p["c_0"]),
        c_ostwald=float(p["c_ostwald"]),
        w_ostwald=float(p["w_ostwald"]),
        bc_right=bc,
        rho=float(p.get("rho", 1.0)),
        phi_gate=phi_gate,
    )
    if demixing == "ac":
        return TwophaseParams(
            **base,
            kappa_s=float(p["kappa_s"]),
            L_s=float(p["L_s"]),
            kappa=0.0,
            M0=0.0,
        )
    return TwophaseParams(
        **base,
        kappa_s=0.0,
        L_s=0.0,
        kappa=float(p["kappa"]),
        M0=float(p["M0"]),
    )


def build_time_params(cfg: dict[str, Any]) -> TwophaseTimeParams:
    t = cfg.get("time") or {}
    return TwophaseTimeParams(
        T=float(t["T"]),
        dt_safety=float(t["dt_safety"]),
        snapshot_every=int(t.get("snapshot_every", 500)),
    )


def _c_init(cfg: dict[str, Any]) -> float:
    ini = cfg.get("initial") or {}
    return float(ini.get("c_init", 0.3))


def _physics_mode(cfg: dict[str, Any]) -> tuple[str, str]:
    p = cfg.get("physics") or {}
    run_mode = str(p.get("run_mode", "reactive")).lower()
    precip = str(p.get("precipitation", "ostwald")).lower()
    return run_mode, precip


def estimate_integration_steps(cfg: dict[str, Any]) -> tuple[float, int]:
    """Match reference-kernel dt limits (spectral CH when ``demixing: ch``)."""
    geom = build_geometry(cfg)
    prm = build_params(cfg)
    tm = build_time_params(cfg)
    c_init = _c_init(cfg)
    run_mode, precip = _physics_mode(cfg)
    dx = geom.dx
    if run_mode == "driven":
        if prm.demixing != "ch" or prm.M0 <= 0.0:
            raise ValueError("driven mode requires demixing='ch' and M0>0")
        qm = np.pi / dx
        fpp_abs = 4.0 * KERNEL_W + 4.0 * prm.gamma
        lims = [2.0 / (prm.M0 * qm * qm * (fpp_abs + prm.kappa * qm * qm))]
    else:
        lims = [dx * dx / (2.0 * prm.D_c), 1.0 / (prm.k_rxn * max(prm.c_0, c_init) + 1e-12)]
        if precip != "hysteresis" and prm.demixing == "ch" and prm.M0 > 0.0:
            qm = np.pi / dx
            fpp_abs = 4.0 * KERNEL_W + 4.0 * prm.gamma
            lims.append(2.0 / (prm.M0 * qm * qm * (fpp_abs + prm.kappa * qm * qm)))
    dt = tm.dt_safety * min(lims)
    n_steps = int(np.ceil(tm.T / dt))
    return float(dt), n_steps


def _front_position(phi_tot: np.ndarray, x: np.ndarray) -> float:
    idx = np.where(phi_tot > 0.5)[0]
    return float(x[idx[-1]]) if idx.size else 0.0


def _c_front_ahead(c: np.ndarray, phi_tot: np.ndarray) -> float:
    idx = np.where(phi_tot > 0.5)[0]
    if not idx.size:
        return float(c[0])
    j = int(min(int(idx[-1]) + 1, c.size - 1))
    return float(c[j])


def _c_at_front_cell(c: np.ndarray, phi_tot: np.ndarray) -> float:
    return c_dep_at_active_front(c, phi_tot)


def _frames_to_evolution(
    frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    x: np.ndarray,
    *,
    evolution_dt: float,
    t_final: float,
) -> tuple[list[float], list[np.ndarray], list[float], list[float], list[float], list[float]]:
    if not frames or evolution_dt <= 0.0:
        return [], [], [], [], [], []
    targets = np.arange(0.0, t_final + 0.5 * evolution_dt, evolution_dt, dtype=np.float64)
    snap_t: list[float] = []
    snap_solid: list[np.ndarray] = []
    snap_c0: list[float] = []
    snap_c_front: list[float] = []
    snap_xfront: list[float] = []
    snap_c_dep: list[float] = []
    frame_times = np.array([f[0] for f in frames], dtype=np.float64)
    j = 0
    for t_target in targets:
        while j + 1 < frame_times.size and frame_times[j + 1] <= t_target:
            j += 1
        t_f, pt, _ps, cc = frames[j]
        snap_t.append(float(t_target if t_target <= t_final else t_f))
        snap_solid.append(np.asarray(pt, dtype=np.float64, copy=True))
        snap_c0.append(float(cc[0]))
        snap_c_front.append(_c_front_ahead(cc, pt))
        snap_xfront.append(_front_position(pt, x))
        snap_c_dep.append(_c_at_front_cell(cc, pt))
    return snap_t, snap_solid, snap_c0, snap_c_front, snap_xfront, snap_c_dep


def _resample_trace(
    t_src: np.ndarray,
    values: np.ndarray | list[Any],
    *,
    t_final: float,
    evolution_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick nearest-kernel samples on the evolution_dt grid."""
    t_src = np.asarray(t_src, dtype=np.float64).ravel()
    if t_src.size == 0 or evolution_dt <= 0.0:
        return np.array([], dtype=np.float64), np.asarray(values).ravel()[:0]
    vals = np.asarray(values).ravel()
    if vals.size != t_src.size:
        raise ValueError(f"trace length mismatch: t={t_src.size} values={vals.size}")
    targets = np.arange(0.0, t_final + 0.5 * evolution_dt, evolution_dt, dtype=np.float64)
    out_t: list[float] = []
    out_v: list[Any] = []
    j = 0
    for t_target in targets:
        while j + 1 < t_src.size and t_src[j + 1] <= t_target:
            j += 1
        t_pick = float(t_target if t_target <= t_final else t_src[j])
        out_t.append(t_pick)
        out_v.append(vals[j])
    return np.asarray(out_t, dtype=np.float64), np.asarray(
        out_v, dtype=object if vals.dtype == object else vals.dtype
    )


def _hysteresis_evolution_snapshots(
    kr: dict[str, Any],
    *,
    t_final: float,
    evolution_dt: float,
) -> dict[str, np.ndarray]:
    """Resample kernel hysteresis traces to evolution_dt."""
    t_src = np.asarray(kr.get("snapshots_t_state", []), dtype=np.float64)
    if t_src.size == 0 or evolution_dt <= 0.0:
        return {}
    out: dict[str, np.ndarray] = {}
    snap_t, _ = _resample_trace(t_src, t_src, t_final=t_final, evolution_dt=evolution_dt)
    out["snapshots_t"] = snap_t
    for key, src in (
        ("snapshots_c_dep", kr.get("snapshots_c_dep", [])),
        ("snapshots_S", list(kr.get("snapshots_S", []))),
        ("snapshots_rate_total", kr.get("snapshots_rate_total", [])),
        ("snapshots_supply_front", kr.get("snapshots_supply_front", [])),
        ("snapshots_consume_front", kr.get("snapshots_consume_front", [])),
        ("snapshots_k_eff", kr.get("snapshots_k_eff", [])),
        ("snapshots_gap_width", kr.get("snapshots_gap_width", [])),
        ("snapshots_Da", kr.get("snapshots_Da", [])),
    ):
        if not src:
            continue
        _, vals = _resample_trace(t_src, src, t_final=t_final, evolution_dt=evolution_dt)
        out[key] = vals
    return out


def _front_frozen_from_snapshots(snap_xfront: list[float], L: float) -> bool:
    if len(snap_xfront) < 2:
        return False
    arr = np.asarray(snap_xfront, dtype=np.float64)
    return bool(np.max(arr) - np.min(arr) < 0.01 * L)


def _dispatch_kernel(
    cfg: dict[str, Any],
    *,
    geom: TwophaseGeometry,
    prm: TwophaseParams,
    tm: TwophaseTimeParams,
    c_init: float,
    n_frames: int,
    seed: int,
    progress: Any,
) -> dict[str, Any]:
    """Route to D1 wrappers or the frozen reference kernel."""
    phys = cfg.get("physics") or {}
    run_mode, precip = _physics_mode(cfg)
    if run_mode == "driven":
        return run_driven(
            L=geom.L,
            n=geom.n,
            T=tm.T,
            v_front=float(phys["v_front"]),
            f_chalcedony=float(phys["f_chalcedony"]),
            gamma=prm.gamma,
            kappa=prm.kappa,
            M0=prm.M0,
            phi_gate=prm.phi_gate,
            seed_px=geom.seed_width_px,
            dt_safety=tm.dt_safety,
            n_frames=n_frames,
            noise_amp=float(phys.get("noise_amp", 1e-3)),
            rng_seed=seed,
            progress=progress,
        )
    if precip == "hysteresis":
        return run_hysteresis(
            L=geom.L,
            n=geom.n,
            T=tm.T,
            D=prm.D_c,
            k_rxn=prm.k_rxn,
            c_sat=prm.c_sat,
            c0=prm.c_0,
            c_init=c_init,
            c1=float(phys["c1"]),
            c2=float(phys["c2"]),
            kappa_rate=float(phys["kappa_rate"]),
            k_L_scale=float(phys.get("k_L_scale", 1.0)),
            bc_right=prm.bc_right,
            phi_gate=prm.phi_gate,
            seed_px=geom.seed_width_px,
            dt_safety=tm.dt_safety,
            n_frames=n_frames,
            progress=progress,
        )
    return _kernel_run(
        mode=prm.demixing,
        bc_right=prm.bc_right,
        L=geom.L,
        n=geom.n,
        T=tm.T,
        D=prm.D_c,
        k_rxn=prm.k_rxn,
        c_sat=prm.c_sat,
        c0=prm.c_0,
        c_init=c_init,
        c_ostwald=prm.c_ostwald,
        w_ostwald=prm.w_ostwald,
        gamma=prm.gamma,
        kappa=prm.kappa,
        M0=prm.M0,
        W=KERNEL_W,
        phi_gate=prm.phi_gate,
        seed_px=geom.seed_width_px,
        dt_safety=tm.dt_safety,
        n_frames=n_frames,
        rng_seed=seed,
        progress=progress,
    )


def simulate(
    cfg: dict[str, Any],
    *,
    chunk_size: int = 2000,  # noqa: ARG001
    show_progress: bool = True,
) -> SimResult:
    """Run the verified reference-kernel integrator (``ac`` or ``ch``)."""
    del chunk_size
    t_wall0 = time.perf_counter()

    geom = build_geometry(cfg)
    prm = build_params(cfg)
    tm = build_time_params(cfg)
    c_init = _c_init(cfg)
    out = cfg.get("output") or {}
    save_evolution = bool(out.get("save_evolution", True))
    evolution_dt = float(out.get("evolution_dt", 50.0))

    run_mode, precip = _physics_mode(cfg)
    mode_tag = (
        run_mode if run_mode == "driven" else precip if precip == "hysteresis" else prm.demixing
    )
    dt_est, n_steps_est = estimate_integration_steps(cfg)
    n_frames = max(120, int(tm.T / max(evolution_dt, dt_est)) + 2) if save_evolution else 120

    pbar = tqdm(
        total=n_steps_est,
        desc=f"slab_twophase[{mode_tag}]",
        unit="step",
        leave=True,
        disable=not show_progress,
        miniters=max(1, n_steps_est // 200),
    )

    seed = int((cfg.get("experiment") or {}).get("seed", 42))
    kr = _dispatch_kernel(
        cfg,
        geom=geom,
        prm=prm,
        tm=tm,
        c_init=c_init,
        n_frames=n_frames,
        seed=seed,
        progress=pbar,
    )
    pbar.close()

    phi_m = np.asarray(kr["phi_m"], dtype=np.float64)
    phi_c = np.asarray(kr["phi_c"], dtype=np.float64)
    c = np.asarray(kr["c"], dtype=np.float64)
    phi_tot = phi_m + phi_c
    phi_w = np.maximum(1.0 - phi_tot, 0.0)
    dx = float(kr["dx"])
    x = np.asarray(kr["x"], dtype=np.float64)

    M_deposit = float(np.sum(phi_tot) * dx * prm.rho)
    u = np.asarray(kr["u"], dtype=np.float64)
    M_solution = float(np.sum(u) * dx)
    M_influx = float(kr["influx"])
    M_tot0 = float(kr["M_tot0"])
    M_tot_f = M_solution + M_deposit
    if run_mode == "driven":
        ledger_residual = float(kr.get("phi_ledger_residual", 0.0))
        mass_residual_pct = 100.0 * ledger_residual
    else:
        ledger_residual = abs(M_tot_f - M_tot0 - M_influx)
        mass_residual_pct = 100.0 * ledger_residual / max(abs(M_tot0) + abs(M_influx), 1e-30)

    clipped_mass = float(kr["clipped_mass"])
    total_silica = max(abs(M_tot0) + abs(M_influx), 1e-30)  # noqa: F841

    n_projection_events = int(kr["nproj"])
    min_phi_w = float(kr["min_phiw"])
    min_phi_w_wetted = float(np.min(phi_w[phi_tot > 1e-8])) if np.any(phi_tot > 1e-8) else min_phi_w
    max_abs_right_flux = float(kr["max_abs_right_flux"])
    t_final = float(kr["nsteps"]) * float(kr["dt"])
    wall_s = time.perf_counter() - t_wall0

    frames = kr["frames"]
    snap_S_hyst: np.ndarray | None = None
    snap_rate_total: list[float] = []
    snap_supply_front: list[float] = []
    snap_consume_front: list[float] = []
    snap_k_eff: list[float] = []
    snap_gap_width: list[float] = []
    snap_Da: list[float] = []
    if save_evolution:
        snap_t, snap_solid, snap_c0, snap_c_front, snap_xfront, snap_c_dep = _frames_to_evolution(
            frames, x, evolution_dt=evolution_dt, t_final=t_final
        )
        if precip == "hysteresis":
            h_snaps = _hysteresis_evolution_snapshots(
                kr, t_final=t_final, evolution_dt=evolution_dt
            )
            if h_snaps.get("snapshots_t", np.array([])).size:
                snap_t = [float(v) for v in h_snaps["snapshots_t"]]
                snap_c_dep = [float(v) for v in h_snaps.get("snapshots_c_dep", [])]
                snap_S_hyst = h_snaps.get("snapshots_S")
                snap_rate_total = [float(v) for v in h_snaps.get("snapshots_rate_total", [])]
                snap_supply_front = [float(v) for v in h_snaps.get("snapshots_supply_front", [])]
                snap_consume_front = [float(v) for v in h_snaps.get("snapshots_consume_front", [])]
                snap_k_eff = [float(v) for v in h_snaps.get("snapshots_k_eff", [])]
                snap_gap_width = [float(v) for v in h_snaps.get("snapshots_gap_width", [])]
                snap_Da = [float(v) for v in h_snaps.get("snapshots_Da", [])]
    else:
        snap_t, snap_solid, snap_c0, snap_c_front, snap_xfront, snap_c_dep = [], [], [], [], [], []

    clip_rate = compute_clip_rate(n_projection_events, int(kr["nsteps"]), geom.n)
    lambda_ch = (
        ch_spinodal_wavelength(gamma=prm.gamma, kappa=prm.kappa) if prm.demixing == "ch" else None
    )
    front_frozen = _front_frozen_from_snapshots(snap_xfront, geom.L)

    diagnostics: dict[str, Any] = {
        "mass_residual_pct": float(mass_residual_pct),
        "min_phi_w": min_phi_w,
        "min_phi_w_wetted": min_phi_w_wetted,
        "n_projection_events": n_projection_events,
        "clipped_mass": clipped_mass,
        "M_solution_final": M_solution,
        "M_deposit_final": M_deposit,
        "M_influx": M_influx,
        "wall_time_s": float(wall_s),
        "n_steps": int(kr["nsteps"]),
        "dt": float(kr["dt"]),
        "T": float(tm.T),
        "bc_right": prm.bc_right,
        "demixing": prm.demixing,
        "phi_gate": prm.phi_gate,
        "max_abs_right_flux": max_abs_right_flux,
        "front_frozen": front_frozen,
        "run_mode": run_mode,
        "precipitation": precip,
        "d1_kernel_mode": str(kr.get("mode", prm.demixing)),
        "clip_rate": clip_rate,
    }
    if run_mode == "driven":
        diagnostics["phi_ledger_residual"] = float(kr.get("phi_ledger_residual", ledger_residual))
        diagnostics["prescribed_v"] = float(kr.get("prescribed_v", float("nan")))
        diagnostics["f_chalcedony"] = float(kr.get("f_chalcedony", float("nan")))
        diagnostics["rng_seed"] = int(kr.get("rng_seed", seed))
    if precip == "hysteresis":
        diagnostics["c1"] = float(kr.get("c1", float("nan")))
        diagnostics["c2"] = float(kr.get("c2", float("nan")))
        diagnostics["kappa_rate"] = float(kr.get("kappa_rate", float("nan")))
        diagnostics["k_L_scale"] = float(kr.get("k_L_scale", float("nan")))
    if lambda_ch is not None:
        diagnostics["lambda_ch"] = float(lambda_ch)

    import jax.numpy as jnp

    zeros1 = jnp.zeros((geom.n, 1))
    state_final = SimState(
        phi_m=jnp.asarray(phi_m[:, None], dtype=jnp.float64),
        phi_c=jnp.asarray(phi_c[:, None], dtype=jnp.float64),
        phi_q=zeros1,
        phi_imp=zeros1,
        c=jnp.asarray(c[:, None], dtype=jnp.float64),
        t=t_final,
    )

    meta_block: dict[str, Any] = {
        "geometry": {"L": geom.L, "n": geom.n, "seed_width_px": geom.seed_width_px, "dx": dx},
        "demixing": prm.demixing,
        "phi_tot_final": np.asarray(phi_tot, dtype=np.float64),
        "snapshots_t": np.asarray(snap_t, dtype=np.float64),
        "snapshots_solid": (
            np.stack(snap_solid, axis=0) if snap_solid else np.zeros((0, geom.n), dtype=np.float64)
        ),
        "snapshots_c0": np.asarray(snap_c0, dtype=np.float64),
        "snapshots_c_front": np.asarray(snap_c_front, dtype=np.float64),
        "snapshots_x_front": np.asarray(snap_xfront, dtype=np.float64),
        "snapshots_c_dep": np.asarray(snap_c_dep, dtype=np.float64),
        "domain_L": geom.L,
        "t_final": t_final,
        "seed_exclude_x": float(3 * geom.seed_width_px * dx),
        "s_final": np.asarray(phi_m - phi_c, dtype=np.float64),
    }
    if precip == "hysteresis":
        meta_block["snapshots_S"] = (
            snap_S_hyst if snap_S_hyst is not None else np.asarray(kr.get("snapshots_S", []))
        )
        meta_block["snapshots_t_state"] = np.asarray(snap_t, dtype=np.float64)
        meta_block["snapshots_c_dep_kernel"] = np.asarray(
            kr.get("snapshots_c_dep", []), dtype=np.float64
        )
        meta_block["snapshots_rate_total"] = np.asarray(snap_rate_total, dtype=np.float64)
        meta_block["snapshots_supply_front"] = np.asarray(snap_supply_front, dtype=np.float64)
        meta_block["snapshots_consume_front"] = np.asarray(snap_consume_front, dtype=np.float64)
        meta_block["snapshots_k_eff"] = np.asarray(snap_k_eff, dtype=np.float64)
        meta_block["snapshots_gap_width"] = np.asarray(snap_gap_width, dtype=np.float64)
        meta_block["snapshots_Da"] = np.asarray(snap_Da, dtype=np.float64)
    if lambda_ch is not None:
        meta_block["lambda_ch"] = float(lambda_ch)

    logger.info(
        "slab_twophase[%s] finished: steps=%d dt=%.3e wall=%.2fs mass_res=%.4f%% "
        "n_proj=%d clipped_frac=%.3e min_phi_w=%.4g",
        prm.demixing,
        int(kr["nsteps"]),
        float(kr["dt"]),
        wall_s,
        mass_residual_pct,
        n_projection_events,
        clip_rate,
        min_phi_w,
    )

    return SimResult(
        state_final=state_final,
        meta={"slab_twophase": meta_block},
        diagnostics=diagnostics,
        config_resolved=dict(cfg),
    )
