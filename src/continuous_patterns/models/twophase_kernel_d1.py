"""D1 discrimination kernels — deposition wrappers around frozen CH exchange.

The antisymmetric CH exchange block is copied verbatim from
``twophase_kernel_reference.py`` lines 174–185 (see ``_ch_exchange_step``).
New physics enters only as deposition sources / boundary modes.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from continuous_patterns.models.twophase_kernel_reference import gprime, lap_neumann, smooth3

SurfaceState = Literal["L", "H"]
KERNEL_W = 0.0
PHI_W_FRONT_MIN = 0.05


def c_from_u_guarded(
    u: np.ndarray,
    phi_tot: np.ndarray,
    *,
    c_max: float = 1.0,
    phi_w_min: float = PHI_W_FRONT_MIN,
) -> np.ndarray:
    """Recover c = u/phi_w with guards against phi_w → 0 in deposited cells."""
    phi_w = 1.0 - phi_tot
    c = u / np.maximum(phi_w, 1e-12)
    c[phi_w <= phi_w_min] = 0.0
    np.clip(c, 0.0, c_max, out=c)
    return c


def c_dep_at_active_front(
    c: np.ndarray,
    phi_tot: np.ndarray,
    *,
    phi_w_min: float = PHI_W_FRONT_MIN,
) -> float:
    """Silica concentration at the active front: last cell with phi_w > phi_w_min."""
    phi_w = 1.0 - phi_tot
    wetted = np.where(phi_w > phi_w_min)[0]
    if wetted.size == 0:
        return float(np.clip(c[0], 0.0, 1.0)) if c.size else 0.0
    return float(np.clip(c[int(wetted[-1])], 0.0, 1.0))


def _ch_exchange_step(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    dt: float,
    dx: float,
    gamma: float,
    kappa: float,
    M0: float,
    W: float = KERNEL_W,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim exchange update from twophase_kernel_reference.py:174-185."""
    mu_m = W * gprime(phi_m) + 2 * gamma * phi_m * phi_c**2 - kappa * lap_neumann(phi_m, dx)
    mu_c = W * gprime(phi_c) + 2 * gamma * phi_c * phi_m**2 - kappa * lap_neumann(phi_c, dx)
    dmu = mu_m - mu_c
    Mloc = M0 * phi_m * phi_c
    Mf = np.sqrt(np.maximum(Mloc[:-1] * Mloc[1:], 0.0))
    J = -Mf * (dmu[1:] - dmu[:-1]) / dx
    dv = np.zeros_like(phi_m)
    dv[:-1] -= J / dx
    dv[1:] += J / dx
    return dv.copy(), -dv


def _project_phi(phi_m: np.ndarray, phi_c: np.ndarray, dx: float) -> tuple[int, float]:
    nproj = 0
    clipped = 0.0
    for arr in (phi_m, phi_c):
        neg = arr < 0
        if neg.any():
            nproj += int(neg.sum())
            clipped += float(-arr[neg].sum()) * dx
            arr[neg] = 0.0
    tot = phi_m + phi_c
    over = tot > 1.0
    if over.any():
        nproj += int(over.sum())
        scale = 1.0 / tot[over]
        clipped += float((tot[over] - 1.0).sum()) * dx
        phi_m[over] *= scale
        phi_c[over] *= scale
    return nproj, clipped


def _front_index(phi_tot: np.ndarray) -> int:
    idx = np.where(phi_tot > 0.5)[0]
    return int(idx[-1]) if idx.size else -1


def _front_x_position(phi_tot: np.ndarray, x: np.ndarray) -> float:
    idx = _front_index(phi_tot)
    if idx < 0:
        return 0.0
    return float(x[idx])


def _active_front_cell(phi_tot: np.ndarray, *, phi_w_min: float = PHI_W_FRONT_MIN) -> int:
    phi_w = 1.0 - phi_tot
    wetted = np.where(phi_w > phi_w_min)[0]
    return int(wetted[-1]) if wetted.size else 0


def _c_at_front(c: np.ndarray, phi_tot: np.ndarray) -> float:
    return c_dep_at_active_front(c, phi_tot)


def run_driven(
    *,
    L: float,
    n: int,
    T: float,
    v_front: float,
    f_chalcedony: float,
    gamma: float = 4.0,
    kappa: float = 0.5,
    M0: float = 1.0,
    phi_gate: float = 0.02,
    seed_px: int = 2,
    seed_phi: float = 0.5,
    dt_safety: float = 0.4,
    n_frames: int = 120,
    noise_amp: float = 1e-3,
    rng_seed: int = 42,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Block A: prescribed front x_f = v·t, fixed co-deposit fraction, CH behind front."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    rng = np.random.default_rng(rng_seed)

    phi_m = np.zeros(n, dtype=np.float64)
    phi_c = np.zeros(n, dtype=np.float64)
    phi_m[:seed_px] = seed_phi
    phi_c[:seed_px] = max(0.0, 1.0 - seed_phi)

    qm = np.pi / dx
    fpp_abs = 4 * KERNEL_W + 4 * gamma
    dt = dt_safety * (2.0 / (M0 * qm * qm * (fpp_abs + kappa * qm * qm)))
    nsteps = int(np.ceil(T / dt))

    nproj = 0
    clipped_mass = 0.0
    frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    frame_every = max(1, nsteps // n_frames)
    deposited_cells = int(np.sum((phi_m + phi_c)[:seed_px] > 0.5))
    M_deposited_expected = float(min(v_front * T, L))

    for it in range(nsteps):
        t = it * dt
        x_front = min(v_front * t, L)
        front_i = int(min(np.searchsorted(x, x_front, side="right"), n))
        for i in range(front_i):
            if phi_m[i] + phi_c[i] < 0.5:
                noise_m = noise_amp * rng.standard_normal()
                noise_c = noise_amp * rng.standard_normal()
                pm = max(0.0, (1.0 - f_chalcedony) + noise_m)
                pc = max(0.0, f_chalcedony + noise_c)
                s = pm + pc
                if s > 0:
                    pm /= s
                    pc /= s
                phi_m[i] = pm
                phi_c[i] = pc
                deposited_cells += 1

        dv_m, dv_c = _ch_exchange_step(phi_m, phi_c, dt=dt, dx=dx, gamma=gamma, kappa=kappa, M0=M0)
        phi_m += dt * dv_m
        phi_c += dt * dv_c
        npj, clp = _project_phi(phi_m, phi_c, dx)
        nproj += npj
        clipped_mass += clp

        if it % frame_every == 0:
            pt = phi_m + phi_c
            frames.append((t, pt.copy(), (phi_m - phi_c).copy(), np.zeros(n)))

        if progress is not None:
            progress.update(1)

    phi_tot = phi_m + phi_c
    M_deposited = float(np.sum(phi_tot) * dx)
    resid = abs(M_deposited - M_deposited_expected) / max(M_deposited_expected, 1e-12)

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "nsteps": nsteps,
        "phi_m": phi_m,
        "phi_c": phi_c,
        "u": np.zeros(n),
        "c": np.zeros(n),
        "resid_pct": 100.0 * resid,
        "influx": 0.0,
        "nproj": nproj,
        "clipped_mass": clipped_mass,
        "min_phiw": float(np.min(1.0 - phi_tot)),
        "c0_max": 0.0,
        "max_abs_right_flux": 0.0,
        "frames": frames,
        "M_tot0": 0.0,
        "M_deposited_expected": M_deposited_expected,
        "M_deposited_actual": M_deposited,
        "phi_ledger_residual": resid,
        "prescribed_v": float(v_front),
        "f_chalcedony": float(f_chalcedony),
        "noise_amp": float(noise_amp),
        "rng_seed": int(rng_seed),
        "mode": "driven",
    }


def run_hysteresis(
    *,
    L: float,
    n: int,
    T: float,
    D: float = 1.0,
    k_rxn: float = 0.5,
    c_sat: float = 0.2,
    c0: float = 1.0,
    c_init: float = 0.3,
    c1: float = 0.30,
    c2: float = 0.25,
    kappa_rate: float = 5.0,
    k_L_scale: float = 1.0,
    bc_right: str = "dirichlet",
    phi_gate: float = 0.02,
    seed_px: int = 2,
    seed_phi: float = 0.5,
    dt_safety: float = 0.4,
    n_frames: int = 120,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Block B: hysteretic surface precipitation, transport on, demixing OFF (recorder)."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    eps = 1e-6  # noqa: F841

    phi_m = np.zeros(n, dtype=np.float64)
    phi_c = np.zeros(n, dtype=np.float64)
    phi_m[:seed_px] = seed_phi
    phi_c[:seed_px] = max(0.0, 1.0 - seed_phi)
    c = np.full(n, c_init, dtype=np.float64)
    u = (1.0 - phi_m - phi_c) * c

    k_L = float(k_L_scale) * k_rxn
    k_H = kappa_rate * k_L
    state: SurfaceState = "L"

    lims = [dx * dx / (2 * D), 1.0 / (k_rxn * max(c0, c_init) + 1e-12)]
    dt = dt_safety * min(lims)
    nsteps = int(np.ceil(T / dt))

    influx = 0.0
    nproj = 0
    clipped_mass = 0.0
    max_abs_right_flux = 0.0
    frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    snap_S: list[str] = []
    snap_c_dep: list[float] = []
    snap_t: list[float] = []
    snap_rate_total: list[float] = []
    snap_supply_front: list[float] = []
    snap_consume_front: list[float] = []
    snap_k_eff: list[float] = []
    snap_gap_width: list[float] = []
    snap_Da: list[float] = []
    frame_every = max(1, nsteps // n_frames)
    M_tot0 = (u.sum() + (phi_m + phi_c).sum()) * dx

    for it in range(nsteps):
        phi_tot = phi_m + phi_c
        phi_w = 1.0 - phi_tot
        c_max = max(c0, c_init, c1, c2, c_sat)
        c[:] = c_from_u_guarded(u, phi_tot, c_max=c_max)

        c_dep = c_dep_at_active_front(c, phi_tot)
        if state == "L" and c_dep > c1:
            state = "H"
        elif state == "H" and c_dep < c2:
            state = "L"
        k_S = k_H if state == "H" else k_L

        phis = smooth3(phi_tot)
        G = phis * np.maximum(1.0 - phi_tot, 0.0)
        G[phis < phi_gate] = 0.0
        R = k_S * np.maximum(c - c_sat, 0.0) * G
        # Single-phase deposition: cap R by remaining void (avoids phi_tot > 1 blow-up).
        R = np.minimum(R, np.maximum(1.0 - phi_tot, 0.0) / dt)

        Dw = D * 0.5 * (phi_w[:-1] + phi_w[1:])
        F = Dw * (c[1:] - c[:-1]) / dx
        div = np.zeros(n)
        div[:-1] += F / dx
        div[1:] -= F / dx
        if bc_right == "dirichlet":
            Fr = D * phi_w[-1] * (c0 - c[-1]) / dx
            div[-1] += Fr / dx
            influx += Fr * dt
            max_abs_right_flux = max(max_abs_right_flux, abs(float(Fr)))

        u = u + dt * (div - R)

        if state == "H":
            dphi_m = R
            dphi_c = np.zeros_like(R)
        else:
            dphi_m = np.zeros_like(R)
            dphi_c = R
        phi_m += dt * dphi_m
        phi_c += dt * dphi_c
        npj, clp = _project_phi(phi_m, phi_c, dx)
        nproj += npj
        clipped_mass += clp

        if it % frame_every == 0:
            t = it * dt
            pt = phi_m + phi_c
            front_i = _active_front_cell(pt)
            frames.append((t, pt.copy(), (phi_m - phi_c).copy(), c.copy()))
            snap_t.append(t)
            snap_S.append(state)
            snap_c_dep.append(float(c_dep))
            snap_rate_total.append(float(np.sum(R)))
            snap_supply_front.append(float(div[front_i]))
            snap_consume_front.append(float(R[front_i]))
            snap_k_eff.append(float(k_S))
            L_gap = float(L - _front_x_position(pt, x))
            snap_gap_width.append(L_gap)
            snap_Da.append(float(k_S * L_gap / max(D, 1e-30)))

        if progress is not None:
            progress.update(1)

    phi_tot = phi_m + phi_c
    M_tot = (u.sum() + phi_tot.sum()) * dx
    resid = abs(M_tot - M_tot0 - influx) / max(abs(M_tot0) + abs(influx), 1e-12) * 100

    return {
        "x": x,
        "dx": dx,
        "dt": dt,
        "nsteps": nsteps,
        "phi_m": phi_m,
        "phi_c": phi_c,
        "u": u,
        "c": c,
        "resid_pct": resid,
        "influx": influx,
        "nproj": nproj,
        "clipped_mass": clipped_mass,
        "min_phiw": float(np.min(1.0 - phi_tot)),
        "c0_max": float(c[0]),
        "max_abs_right_flux": max_abs_right_flux,
        "frames": frames,
        "M_tot0": M_tot0,
        "snapshots_S": snap_S,
        "snapshots_c_dep": snap_c_dep,
        "snapshots_t_state": snap_t,
        "snapshots_rate_total": snap_rate_total,
        "snapshots_supply_front": snap_supply_front,
        "snapshots_consume_front": snap_consume_front,
        "snapshots_k_eff": snap_k_eff,
        "snapshots_gap_width": snap_gap_width,
        "snapshots_Da": snap_Da,
        "mode": "hysteresis",
        "c1": c1,
        "c2": c2,
        "kappa_rate": kappa_rate,
        "k_L_scale": k_L_scale,
        "D": D,
        "k_rxn": k_rxn,
        "k_L": k_L,
        "k_H": k_H,
    }
