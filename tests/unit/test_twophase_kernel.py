"""Golden-value contract tests for the verified twophase reference kernel (V1–V4b)."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.models.twophase_kernel_reference import run


def _front_final(fronts: np.ndarray) -> float:
    return float(fronts[-1, 1]) if fronts.size else 0.0


def _dominant_wavelength(psi: np.ndarray, dx: float) -> float:
    spec = np.abs(np.fft.rfft(psi - psi.mean())) ** 2
    freqs = np.fft.rfftfreq(len(psi), d=dx)
    kpk = freqs[np.argmax(spec[1:]) + 1]
    return float(1.0 / kpk) if kpk > 0 else float("inf")


def _band_count_proxy(psi: np.ndarray) -> int:
    sgn = np.sign(psi)
    sgn[np.abs(psi) < 0.05] = 0
    nz = sgn[sgn != 0]
    if nz.size < 2:
        return 0
    return int((np.diff(nz) != 0).sum())


def _moganite_fraction(phi_m: np.ndarray, phi_c: np.ndarray, dx: float) -> float:
    phi_tot = phi_m + phi_c
    dep = float(np.sum(phi_tot) * dx)
    if dep <= 0.0:
        return 0.0
    return float(np.sum(phi_m) * dx / dep)


def _clipped_mass_frac(r: dict) -> float:
    total = max(abs(float(r["M_tot0"])) + abs(float(r["influx"])), 1e-30)
    return float(r["clipped_mass"]) / total


def test_v1_ac_open() -> None:
    """V1: ac-open n=200 T=4000 — front ~51.8, zero projections."""
    r = run(mode="ac", bc_right="dirichlet", n=200, T=4000.0)
    front = _front_final(r["fronts"])
    assert 49.0 <= front <= 55.0
    assert r["resid_pct"] < 1e-8
    assert r["nproj"] == 0


def test_v2_ac_closed() -> None:
    """V2: ac-closed n=200 T=20000 c_init=0.5 — stall ~37.8, zero right flux."""
    r = run(mode="ac", bc_right="noflux", n=200, c_init=0.5, T=20000.0)
    front = _front_final(r["fronts"])
    assert 35.0 <= front <= 41.0
    budget = 0.5 * 100.0
    dep = float((r["phi_m"] + r["phi_c"]).sum() * r["dx"])
    assert dep <= budget * 1.005
    assert r["resid_pct"] < 1e-8
    assert r["nproj"] == 0
    assert r["max_abs_right_flux"] == pytest.approx(0.0, abs=1e-12)


def _tp_integrity_ok(r: dict) -> bool:
    """Runner rule: zero projections OR negligible clipped mass."""
    total = max(abs(float(r["M_tot0"])) + abs(float(r["influx"])), 1e-30)
    clipped_frac = float(r["clipped_mass"]) / total
    return int(r["nproj"]) == 0 or clipped_frac < 1e-6


def test_v3_demix() -> None:
    """V3: demix n=256 T=80 W=0 — lambda ~5.88, |psi|max > 0.85."""
    r = run(
        mode="ch",
        demix_test=True,
        demix_amp=1e-3,
        T=80.0,
        n=256,
        L=100.0,
        gamma=4.0,
        kappa=0.5,
        W=0.0,
        c_init=0.0,
        c0=0.0,
        rng_seed=42,
    )
    psi = r["phi_m"] - r["phi_c"]
    lam = _dominant_wavelength(psi, r["dx"])
    assert 4.5 <= lam <= 7.5
    assert float(np.abs(psi).max()) > 0.85
    assert r["nproj"] == 0
    assert r["resid_pct"] < 1e-6


def test_v4a_ch_front_baseline() -> None:
    """V4a: ch-front baseline Ostwald — 1 band, moganite fraction < 0.10."""
    r = run(
        mode="ch",
        bc_right="dirichlet",
        n=128,
        T=2500.0,
        c_ostwald=0.60,
        W=1.0,
    )
    psi = r["phi_m"] - r["phi_c"]
    assert _band_count_proxy(psi) == 1
    assert _moganite_fraction(r["phi_m"], r["phi_c"], r["dx"]) < 0.10


def test_v4b_ch_front_matched() -> None:
    """V4b: ch-front matched Ostwald W=0 — >=4 bands, m-frac ~0.39, tiny clips."""
    r = run(
        mode="ch",
        bc_right="dirichlet",
        n=128,
        T=2500.0,
        c_ostwald=0.25,
        w_ostwald=0.05,
        W=0.0,
    )
    psi = r["phi_m"] - r["phi_c"]
    assert _band_count_proxy(psi) >= 4
    m_frac = _moganite_fraction(r["phi_m"], r["phi_c"], r["dx"])
    assert 0.25 <= m_frac <= 0.55
    assert _tp_integrity_ok(r)
    assert _clipped_mass_frac(r) < 1e-6
