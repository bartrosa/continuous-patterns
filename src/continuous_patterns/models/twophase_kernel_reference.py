"""Reference kernel for slab_twophase (VERIFIED prototype — integrate VERBATIM).

Verified findings encoded in this file (do not alter the math):
1. Growth channel: dphi_tot = R only; R = k*max(c-c_sat,0)*G with
   G = smooth3(phi_tot)*(1-phi_tot), gated at smooth3(phi_tot) > phi_gate.
   Front advances cell-by-cell via the smoothed gating; bulk can never
   ignite (no noise, gate=0 there). Zero projections by construction
   (local (1-phi_tot) saturation).
2. Transport: u = phi_w*c finite-volume; silica ledger closes to ~1e-12%
   open and closed; solute concentrates as water shrinks (physical).
3. Demixing free energy: gamma*phi_m^2*phi_c^2 ONLY (W wells removed for
   the exchange channel): per-field double wells + sum constraint put the
   minimum at negative phi (root cause of the 78M-projection failure).
4. Demixing dynamics: single antisymmetric EXCHANGE flux driven by
   (mu_m - mu_c), degenerate face mobility M0*phi_m*phi_c (geometric mean
   across faces): local phi_tot untouched by demixing, conservation ~1e-13,
   pure bands lock. Explicit CH dt limit is SPECTRAL:
   dt <= 2*safety / (M0*q^2*(|f''| + kappa*q^2)), q = pi/dx.
   (The dx^4/(8*M*kappa) rule of thumb is ~3x too loose -> grid sawtooth.)
Measured reference values (n, T as in __main__ tests):
   T1 ac-open  n=200 T=4000: front=51.8, resid=1.3e-12%, nproj=0
   T2 ac-closed n=200 T=20000 c_init=0.5: front stall=37.8, deposit=37.8/budget 50
   T3 demix    n=256 T=80 (W=0): lambda=5.88, amp=0.93, nproj=0, resid=0%
   T4a ch-front n=128 T=2500 c_ost=0.60: bands=1 (uniform chalcedony, honest negative)
   T4b ch-front n=128 T=2500 c_ost=0.25 w=0.05: bands=5, m-frac=0.39,
       resid=9.6e-8%, small Gibbs undershoot clips allowed if
       clipped_mass/total < 1e-6.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def psi_ostwald_partition(
    c: np.ndarray,
    *,
    c_ostwald: float,
    w_ostwald: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ostwald co-deposition partition used in the reference kernel (lines ~157-158)."""
    t = (np.asarray(c, dtype=np.float64) - float(c_ostwald)) / max(float(w_ostwald), 1e-12)
    psi_m = sigmoid(t)
    return psi_m, 1.0 - psi_m


def smooth3(a: np.ndarray) -> np.ndarray:
    s = np.empty_like(a)
    s[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3.0
    s[0] = (2 * a[0] + a[1]) / 3.0
    s[-1] = (a[-2] + 2 * a[-1]) / 3.0
    return s


def lap_neumann(a: np.ndarray, dx: float) -> np.ndarray:
    l = np.empty_like(a)  # noqa: E741
    l[1:-1] = (a[:-2] - 2 * a[1:-1] + a[2:]) / dx**2
    l[0] = (a[1] - a[0]) / dx**2
    l[-1] = (a[-2] - a[-1]) / dx**2
    return l


def gprime(x: np.ndarray) -> np.ndarray:
    # double well g = x^2 (1-x)^2
    return 2 * x * (1 - x) * (1 - 2 * x)


def run(
    mode: str = "ac",
    bc_right: str = "dirichlet",
    L: float = 100.0,
    n: int = 200,
    T: float = 3000.0,
    D: float = 1.0,
    k_rxn: float = 0.5,
    c_sat: float = 0.2,
    c0: float = 1.0,
    c_init: float = 0.3,
    c_ostwald: float = 0.6,
    w_ostwald: float = 0.1,
    gamma: float = 4.0,
    kappa: float = 0.5,
    M0: float = 1.0,
    W: float = 0.0,
    phi_gate: float = 0.02,
    seed_px: int = 2,
    seed_phi: float = 0.5,
    dt_safety: float = 0.4,
    n_frames: int = 120,
    rng_seed: int = 42,
    demix_test: bool = False,
    demix_amp: float = 0.0,
    *,
    progress: Any | None = None,
) -> dict[str, Any]:
    dx = L / n
    x = (np.arange(n) + 0.5) * dx

    phi_m = np.zeros(n)
    phi_c = np.zeros(n)
    if demix_test:
        rng = np.random.default_rng(rng_seed)
        pert = demix_amp * rng.standard_normal(n)
        phi_m[:] = 0.475 + pert
        phi_c[:] = 0.475 - pert
        k_rxn = 0.0
    else:
        phi_m[:seed_px] = seed_phi  # moganite seed at left wall

    phi_tot = phi_m + phi_c
    phi_w = 1.0 - phi_tot
    c = np.full(n, c_init)
    u = phi_w * c  # silica per total volume (solution part)

    # dt limits
    lims = [dx * dx / (2 * D), 1.0 / (k_rxn * max(c0, c_init) + 1e-12)]
    if mode == "ch":
        qm = np.pi / dx
        fpp_abs = 4 * W + 4 * gamma
        lims += [2.0 / (M0 * qm * qm * (fpp_abs + kappa * qm * qm))]
    dt = dt_safety * min(lims)
    nsteps = int(np.ceil(T / dt))

    eps = 1e-6
    influx = 0.0
    nproj = 0
    clipped_mass = 0.0
    min_phiw = 1.0
    c0_max = 0.0
    max_abs_right_flux = 0.0
    frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    frame_every = max(1, nsteps // n_frames)
    M_tot0 = (u.sum() + phi_tot.sum()) * dx

    for it in range(nsteps):
        phi_tot = phi_m + phi_c
        phi_w = 1.0 - phi_tot
        c = u / np.maximum(phi_w, eps)
        np.clip(c, 0.0, None, out=c)

        # reaction with smoothed ignition gating + local saturation
        phis = smooth3(phi_tot)
        G = phis * np.maximum(1.0 - phi_tot, 0.0)
        G[phis < phi_gate] = 0.0
        R = k_rxn * np.maximum(c - c_sat, 0.0) * G

        # transport of u = phi_w c (FV, face-averaged phi_w * D)
        Dw = D * 0.5 * (phi_w[:-1] + phi_w[1:])
        F = Dw * (c[1:] - c[:-1]) / dx  # interior faces
        div = np.zeros(n)
        div[:-1] += F / dx
        div[1:] -= F / dx
        if bc_right == "dirichlet":
            Fr = D * phi_w[-1] * (c0 - c[-1]) / dx
            div[-1] += Fr / dx
            influx += Fr * dt
            max_abs_right_flux = max(max_abs_right_flux, abs(float(Fr)))
        # left wall and closed right: zero flux (nothing to add)

        u = u + dt * (div - R)

        # phase update
        psi_m = sigmoid((c - c_ostwald) / w_ostwald)
        psi_c = 1.0 - psi_m
        dphi_m = psi_m * R
        dphi_c = psi_c * R
        if mode == "ch":
            # EXCHANGE formulation: single antisymmetric flux, local phi_tot untouched
            mu_m = W * gprime(phi_m) + 2 * gamma * phi_m * phi_c**2 - kappa * lap_neumann(phi_m, dx)
            mu_c = W * gprime(phi_c) + 2 * gamma * phi_c * phi_m**2 - kappa * lap_neumann(phi_c, dx)
            dmu = mu_m - mu_c
            Mloc = M0 * phi_m * phi_c  # degenerate: dies in pure phases & empty bulk
            Mf = np.sqrt(np.maximum(Mloc[:-1] * Mloc[1:], 0.0))
            J = -Mf * (dmu[1:] - dmu[:-1]) / dx
            dv = np.zeros(n)
            dv[:-1] -= J / dx
            dv[1:] += J / dx
            dphi_m += dv
            dphi_c -= dv
        phi_m = phi_m + dt * dphi_m
        phi_c = phi_c + dt * dphi_c

        # projection bookkeeping (should stay ~0)
        for arr in (phi_m, phi_c):
            neg = arr < 0
            if neg.any():
                nproj += int(neg.sum())
                clipped_mass += float(-arr[neg].sum()) * dx
                arr[neg] = 0.0
        tot = phi_m + phi_c
        over = tot > 1.0
        if over.any():
            nproj += int(over.sum())
            scale = 1.0 / tot[over]
            clipped_mass += float((tot[over] - 1.0).sum()) * dx
            phi_m[over] *= scale
            phi_c[over] *= scale

        min_phiw = min(min_phiw, float((1.0 - (phi_m + phi_c)).min()))
        c0_max = max(c0_max, float(c[0]))

        if it % frame_every == 0:
            frames.append((it * dt, (phi_m + phi_c).copy(), (phi_m - phi_c).copy(), c.copy()))

        if progress is not None:
            progress.update(1)

    phi_tot = phi_m + phi_c
    M_tot = (u.sum() + phi_tot.sum()) * dx
    resid = abs(M_tot - M_tot0 - influx) / max(abs(M_tot0) + abs(influx), 1e-12) * 100

    # front trajectory from frames
    fronts = []
    for t, pt, _ps, _cc in frames:
        idx = np.where(pt > 0.5)[0]
        fronts.append((t, x[idx[-1]] if len(idx) else 0.0))
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
        "min_phiw": min_phiw,
        "c0_max": c0_max,
        "max_abs_right_flux": max_abs_right_flux,
        "frames": frames,
        "fronts": np.array(fronts),
        "M_tot0": M_tot0,
    }


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "t1"

    if which == "t1":
        r = run(mode="ac", bc_right="dirichlet", T=4000.0)
        fr = r["fronts"]
        print("T1 ac-open: dt=%.4g steps=%d" % (r["dt"], r["nsteps"]))  # noqa: UP031
        print(
            "  front: t=500->%.1f  t=2000->%.1f  t=4000->%.1f"  # noqa: UP031
            % (
                np.interp(500, fr[:, 0], fr[:, 1]),
                np.interp(2000, fr[:, 0], fr[:, 1]),
                fr[-1, 1],
            )
        )
        print(
            "  resid=%.3g%%  nproj=%d  min_phiw=%.3g  c0_max=%.3f"  # noqa: UP031
            % (r["resid_pct"], r["nproj"], r["min_phiw"], r["c0_max"])
        )

    if which == "t2":
        r = run(mode="ac", bc_right="noflux", c_init=0.5, T=20000.0)
        fr = r["fronts"]
        budget = 0.5 * 100.0
        dep = (r["phi_m"] + r["phi_c"]).sum() * r["dx"]
        print(
            "T2 ac-closed: front_final=%.1f  deposit=%.1f  budget=%.1f" % (fr[-1, 1], dep, budget)  # noqa: UP031
        )
        print("  resid=%.3g%%  nproj=%d" % (r["resid_pct"], r["nproj"]))  # noqa: UP031

    if which == "t3":
        gamma, kappa = 4.0, 0.5
        r = run(
            mode="ch",
            demix_test=True,
            demix_amp=1e-3,
            T=80.0,
            n=256,
            L=100.0,
            gamma=gamma,
            kappa=kappa,
            c_init=0.0,
            c0=0.0,
        )
        psi = r["phi_m"] - r["phi_c"]
        spec = np.abs(np.fft.rfft(psi - psi.mean())) ** 2
        freqs = np.fft.rfftfreq(len(psi), d=r["dx"])
        kpk = freqs[np.argmax(spec[1:]) + 1]
        lam = 1.0 / kpk if kpk > 0 else np.inf
        phib = 0.475
        W = 0.0  # T3 demix: wells removed (see module docstring)
        fpp = 2 * W * 2 * (1 - 6 * phib + 6 * phib**2) - 4 * gamma * phib**2
        lam_pred = 2 * np.pi * np.sqrt(2 * (2 * kappa) / abs(fpp))
        print(
            "T3 demix: lambda_meas=%.2f  lambda_pred=%.2f  amp(|psi|max)=%.2f"  # noqa: UP031
            % (lam, lam_pred, np.abs(psi).max())
        )
        print(
            "  nproj=%d clipped_mass=%.3g resid=%.3g%%"  # noqa: UP031
            % (r["nproj"], r["clipped_mass"], r["resid_pct"])
        )

    if which == "t4":
        r = run(mode="ch", bc_right="dirichlet", T=2500.0, n=128)
        fr = r["fronts"]
        psi = r["phi_m"] - r["phi_c"]
        sgn = np.sign(psi)
        sgn[np.abs(psi) < 0.05] = 0
        changes = int((np.diff(sgn[sgn != 0]) != 0).sum()) if (sgn != 0).any() else 0
        print("T4 ch-open: dt=%.4g steps=%d front_final=%.1f" % (r["dt"], r["nsteps"], fr[-1, 1]))  # noqa: UP031
        print("  sign changes in psi behind front (band count proxy): %d" % changes)  # noqa: UP031
        print(
            "  resid=%.3g%%  nproj=%d  clipped=%.3g  min_phiw=%.3g"  # noqa: UP031
            % (r["resid_pct"], r["nproj"], r["clipped_mass"], r["min_phiw"])
        )
