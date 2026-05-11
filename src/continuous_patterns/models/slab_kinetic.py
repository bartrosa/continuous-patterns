"""Kinetic 1D agate banding model (Szymczak–Kossacki).

Reproduces the 1D reaction–diffusion model of Kossacki & Szymczak (UW poster).
Single concentration field ``c(x, t)`` on ``x ∈ [0, 1]`` with Dirichlet reservoir
at ``x = 1`` and hysteretic Robin crystallization boundary at ``x = 0``.

References
----------
J. Kossacki, P. Szymczak — "Formation of Rhythmic Band Structures in Agates"
(University of Warsaw, Faculty of Physics poster; based on Heaney & Davis,
Science 269, 1562–1565, 1995).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from continuous_patterns.core.diagnostics_kinetic import (
    compute_dwell_times,
    compute_mean_dwell_period,
    compute_temporal_spectrum,
)
from continuous_patterns.core.types import SimResult, SimState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KineticGeometry:
    """1D grid for the slab kinetic model (local to ``slab_kinetic``)."""

    n: int
    dx: float
    x: np.ndarray  # shape (n,), points in [0, 1]


@dataclass(frozen=True)
class KineticParams:
    """Physical parameters for the slab kinetic model."""

    Da: float  # Damköhler L·r_low/D
    kappa: float  # ratio r_H / r_L
    c1: float  # threshold L→H
    c2: float  # threshold H→L

    def __post_init__(self) -> None:
        if self.c1 <= self.c2:
            raise ValueError(f"c1 must be > c2, got c1={self.c1}, c2={self.c2}")
        if self.kappa <= 1.0:
            raise ValueError(f"kappa must be > 1, got {self.kappa}")


@dataclass(frozen=True)
class KineticTimeParams:
    """Time integration parameters."""

    T: float
    dt_safety: float = 0.4
    snapshot_every: int = 100


@dataclass
class KineticState:
    """Mutable solver state during integration."""

    c: np.ndarray  # shape (n,)
    state: str  # 'L' or 'H'
    t: float


def build_geometry(cfg: dict[str, Any]) -> KineticGeometry:
    """Build a 1D grid from ``cfg['geometry']`` (``type: kinetic_1d``)."""
    g = cfg.get("geometry") or {}
    if str(g.get("type")) != "kinetic_1d":
        raise ValueError("slab_kinetic requires geometry.type='kinetic_1d'")
    n = int(g["n"])
    if n < 3:
        raise ValueError("slab_kinetic requires geometry.n >= 3")
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    dx = float(x[1] - x[0]) if n > 1 else 1.0
    return KineticGeometry(n=n, dx=dx, x=x)


def build_params(cfg: dict[str, Any]) -> KineticParams:
    """Build physical parameters from ``cfg['physics']``."""
    p = cfg.get("physics") or {}
    return KineticParams(
        Da=float(p["Da"]),
        kappa=float(p["kappa"]),
        c1=float(p["c1"]),
        c2=float(p["c2"]),
    )


def build_time_params(cfg: dict[str, Any]) -> KineticTimeParams:
    """Build time-integration parameters from ``cfg['time']``."""
    t = cfg.get("time") or {}
    return KineticTimeParams(
        T=float(t["T"]),
        dt_safety=float(t.get("dt_safety", 0.4)),
        snapshot_every=int(t.get("snapshot_every", 100)),
    )


def build_initial_state(
    cfg: dict[str, Any],
    geom: KineticGeometry,
    prm: KineticParams,  # noqa: ARG001 — reserved for future IC variants
) -> KineticState:
    """Build the initial concentration profile."""
    del prm
    ini = cfg.get("initial") or {}
    profile = str(ini.get("profile", "linear"))
    if profile == "linear":
        c0 = np.asarray(geom.x, dtype=np.float64)
    elif profile == "uniform":
        c0 = np.ones(geom.n, dtype=np.float64)
    else:
        raise ValueError("initial.profile must be 'linear' or 'uniform'")
    return KineticState(c=c0, state="L", t=0.0)


def _robin_r(state: str, kappa: float) -> float:
    return float(kappa) if state == "H" else 1.0


def simulate(
    cfg: dict[str, Any],
    *,
    chunk_size: int = 2000,  # noqa: ARG001 — interface parity with spectral models
    show_progress: bool = True,
) -> SimResult:
    """Run the ``slab_kinetic`` explicit-Euler integrator (NumPy)."""
    del chunk_size
    t_wall0 = time.perf_counter()

    geom = build_geometry(cfg)
    prm = build_params(cfg)
    tm = build_time_params(cfg)
    ic = build_initial_state(cfg, geom, prm)

    dx = geom.dx
    dt = float(tm.dt_safety) * dx * dx / 2.0
    n_steps = int(np.floor(float(tm.T) / dt + 1e-15))
    if n_steps < 1:
        n_steps = 1

    Da = float(prm.Da)
    kappa = float(prm.kappa)
    c1 = float(prm.c1)
    c2 = float(prm.c2)

    c = np.array(ic.c, dtype=np.float64, copy=True)
    state = ic.state
    t = 0.0

    t_hist = np.empty(n_steps, dtype=np.float64)
    c0_hist = np.empty(n_steps, dtype=np.float64)
    st_hist = np.empty(n_steps, dtype=np.int32)
    th_hist = np.empty(n_steps, dtype=np.float64)

    transitions: list[tuple[float, str]] = []
    snap_t: list[float] = []
    snap_c: list[np.ndarray] = []

    thickness = 0.0
    pbar = tqdm(
        total=n_steps,
        desc="slab_kinetic",
        unit="step",
        leave=True,
        disable=not show_progress,
    )

    for step in range(n_steps):
        r_robin = _robin_r(state, kappa)

        lap = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (dx * dx)
        c_new = np.empty_like(c)
        c_new[1:-1] = c[1:-1] + dt * lap
        c_new[-1] = 1.0
        c_new[0] = c_new[1] / (1.0 + r_robin * Da * dx)

        if state == "L" and float(c_new[0]) > c1:
            state = "H"
            transitions.append((float(t + dt), "L→H"))
        elif state == "H" and float(c_new[0]) < c2:
            state = "L"
            transitions.append((float(t + dt), "H→L"))

        r_thick = _robin_r(state, kappa)
        thickness += dt * r_thick * float(c_new[0])

        c[:] = c_new
        t += dt

        t_hist[step] = t
        c0_hist[step] = float(c[0])
        st_hist[step] = 1 if state == "H" else 0
        th_hist[step] = thickness

        if tm.snapshot_every >= 1 and (step % tm.snapshot_every == 0 or step == n_steps - 1):
            snap_t.append(float(t))
            snap_c.append(np.array(c, dtype=np.float64, copy=True))

        pbar.update(1)
        pbar.set_postfix_str(f"t={t:.4g}")

    pbar.close()

    dwell_L, dwell_H = compute_dwell_times(transitions, float(t))
    spec = compute_temporal_spectrum(c0_hist, dt)
    n_tr_ev = int(len(transitions))
    mean_period = compute_mean_dwell_period(dwell_L, dwell_H, n_tr_ev)

    wall_s = time.perf_counter() - t_wall0

    dwell_L_mean = float(np.mean(dwell_L)) if dwell_L.size else float("nan")
    dwell_L_std = float(np.std(dwell_L)) if dwell_L.size else float("nan")
    dwell_H_mean = float(np.mean(dwell_H)) if dwell_H.size else float("nan")
    dwell_H_std = float(np.std(dwell_H)) if dwell_H.size else float("nan")

    transitions_json = [{"t": float(tv), "kind": str(k)} for tv, k in transitions]

    diagnostics: dict[str, Any] = {
        "n_transitions": int(len(transitions)),
        "dwell_L_mean": dwell_L_mean,
        "dwell_L_std": dwell_L_std,
        "dwell_H_mean": dwell_H_mean,
        "dwell_H_std": dwell_H_std,
        "peak_period": float(spec["peak_period"]),
        "peak_frequency": float(spec["peak_freq"]),
        "mean_dwell_period": float(mean_period["mean_dwell_period"]),
        "mean_dwell_freq": float(mean_period["mean_dwell_freq"]),
        "thickness_final": float(thickness),
        "wall_time_s": float(wall_s),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "T": float(tm.T),
        "kinetic_transitions": transitions_json,
        "dwell_L_samples": dwell_L.tolist(),
        "dwell_H_samples": dwell_H.tolist(),
    }

    zeros2 = jnp.zeros((1, 1))
    c_final_j = jnp.asarray(np.asarray(c[:, None], dtype=np.float64))
    state_final = SimState(
        phi_m=zeros2,
        phi_c=zeros2,
        phi_q=zeros2,
        phi_imp=zeros2,
        c=c_final_j,
        t=float(t),
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
        "t_history": t_hist,
        "c_at_x0_history": c0_hist,
        "state_history": st_hist,
        "thickness_history": th_hist,
        "snapshots_t": np.asarray(snap_t, dtype=np.float64),
        "snapshots_c": (
            np.stack(snap_c, axis=0) if snap_c else np.zeros((0, geom.n), dtype=np.float64)
        ),
        "transitions": transitions,
        "temporal_spectrum": {k: v for k, v in spec.items()},
        "state_final_str": state,
        "thickness_final": float(thickness),
        "t_final": float(t),
    }

    logger.info(
        "slab_kinetic finished: steps=%d dt=%.3e wall=%.2fs transitions=%d",
        n_steps,
        dt,
        wall_s,
        len(transitions),
    )

    return SimResult(
        state_final=state_final,
        meta={"slab_kinetic": meta_block},
        diagnostics=diagnostics,
        config_resolved=dict(cfg),
    )
