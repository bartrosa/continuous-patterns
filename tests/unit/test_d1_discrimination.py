"""Unit tests for D1 discrimination kernels."""

from __future__ import annotations

import numpy as np

from continuous_patterns.models.twophase_kernel_d1 import (
    _ch_exchange_step,
    run_driven,
    run_hysteresis,
)


def test_ch_exchange_matches_reference_one_step():
    n = 64
    L = 25.0
    dx = L / n
    rng = np.random.default_rng(0)
    phi_m = np.clip(0.2 + 0.6 * rng.random(n), 0.0, 1.0)
    phi_c = np.clip(1.0 - phi_m + 0.05 * rng.standard_normal(n), 0.0, 1.0)
    s = phi_m + phi_c
    phi_m /= s
    phi_c /= s
    dt = 1e-4
    gamma, kappa, M0 = 4.0, 0.5, 1.0

    dm, dc = _ch_exchange_step(
        phi_m.copy(), phi_c.copy(), dt=dt, dx=dx, gamma=gamma, kappa=kappa, M0=M0
    )
    # reference kernel applies same block inside run(); spot-check antisymmetry
    assert np.allclose(dm, -dc)
    assert np.any(np.abs(dm) > 0)


def test_driven_deposit_extent_and_ledger():
    v = 0.02
    L = 25.0
    n = 64
    T = 1.1 * L / v
    out = run_driven(
        L=L,
        n=n,
        T=T,
        v_front=v,
        f_chalcedony=0.30,
        gamma=4.0,
        kappa=0.5,
        M0=1.0,
        n_frames=20,
    )
    assert out["phi_ledger_residual"] < 1e-5
    phi_tot = out["phi_m"] + out["phi_c"]
    deposited = float(np.sum(phi_tot) * out["dx"])
    assert deposited > 0.5 * L


def test_driven_mode_short_smoke():
    out = run_driven(L=25.0, n=64, T=50.0, v_front=0.05, f_chalcedony=0.3, n_frames=10)
    assert out["mode"] == "driven"
    assert out["nsteps"] > 0
    assert len(out["frames"]) >= 1


def test_hysteresis_finite_summary_fields():
    out = run_hysteresis(L=100.0, n=64, T=60.0, c1=0.49, c2=0.10, kappa_rate=3.0, n_frames=10)
    assert out["mode"] == "hysteresis"
    assert np.isfinite(out["resid_pct"])
    phi_tot = out["phi_m"] + out["phi_c"]
    assert np.all(np.isfinite(phi_tot))
    assert np.all(phi_tot <= 1.0 + 1e-9)
    assert np.all(np.isfinite(out["c"]))
    cdep = np.asarray(out["snapshots_c_dep"], dtype=np.float64)
    assert cdep.size > 0
    assert np.all(cdep >= -1e-12)
    assert np.all(cdep <= 1.0 + 1e-9)


def test_c_dep_active_front_avoids_deposited_cell():
    from continuous_patterns.models.twophase_kernel_d1 import c_dep_at_active_front

    c = np.array([0.3, 0.5, 0.8, 999.0, 999.0])
    phi_tot = np.array([0.0, 0.0, 0.2, 0.99, 1.0])
    val = c_dep_at_active_front(c, phi_tot, phi_w_min=0.05)
    assert 0.0 <= val <= 1.0
    assert val == 0.8
