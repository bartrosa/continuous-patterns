"""Integration smoke test for slab reactive CH driver."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from continuous_patterns.models.slab_reactive import (
    build_geometry,
    build_initial_state,
    build_sim_params,
    simulate,
)


def test_slab_smoke_short_run() -> None:
    """Short slab run should remain finite with bounded fields and low mass leak."""
    cfg = {
        "experiment": {"name": "slab_smoke", "model": "slab_reactive", "seed": 42},
        "geometry": {"type": "slab", "L": 100.0, "n": 64, "eps_scale": 2.0, "rim_width_px": 2},
        "physics": {
            "W": 1.0,
            "gamma": 4.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.01,
            "M_c": 0.1,
            "D_c": 1.0,
            "k_rxn": 0.5,
            "c_sat": 0.2,
            "c_0": 1.0,
            "c_ostwald": 0.6,
            "w_ostwald": 0.1,
            "lambda_bar": 10.0,
            "use_ratchet": True,
            "phi_m_ratchet_low": 0.3,
            "phi_m_ratchet_high": 0.5,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 5.0, "snapshot_every": 100},
        "output": {"save_final_state": False, "record_spectral_mass_diagnostic": True},
        "initial": {
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_m_noise": 0.0,
            "phi_c_noise": 0.0,
            "c_init": 0.2,
        },
    }

    t0 = time.perf_counter()
    result = simulate(cfg, chunk_size=100, show_progress=False)
    wall_time = time.perf_counter() - t0

    phi_m = result.state_final.phi_m
    phi_c = result.state_final.phi_c
    c = result.state_final.c

    assert phi_m.shape == (64, 64)
    assert phi_c.shape == (64, 64)
    assert c.shape == (64, 64)

    assert jnp.all(jnp.isfinite(phi_m))
    assert jnp.all(jnp.isfinite(phi_c))
    assert jnp.all(jnp.isfinite(c))

    assert float(jnp.min(phi_m)) >= -0.050001
    assert float(jnp.max(phi_m)) <= 1.5
    assert float(jnp.min(phi_c)) >= -0.050001
    assert float(jnp.max(phi_c)) <= 1.5
    assert float(jnp.min(c)) >= 0.0
    assert float(jnp.max(c)) <= 1.5

    geom = build_geometry(cfg)
    prm = build_sim_params(cfg)
    seed = int(cfg["experiment"]["seed"])
    key = jax.random.PRNGKey(seed)
    key, k_ic = jax.random.split(key)
    _pm0, _pc0, _pq0, _pim0, c0 = build_initial_state(cfg, geom, prm, k_ic)
    dx = float(geom.dx)

    c_initial_total = float(jnp.sum(c0) * (dx * dx))
    final_total = float(
        jnp.sum(c + float(prm.phi_m_potential.rho) * phi_m + float(prm.phi_c_potential.rho) * phi_c)
        * (dx * dx)
    )
    injection = float(result.meta.get("cumulative_dirichlet_injection", 0.0))
    leak = abs((c_initial_total + injection) - final_total)
    leak_pct = 100.0 * leak / max(c_initial_total, 1e-30)
    assert leak < 0.05 * max(c_initial_total, 1e-30)

    print(f"wall_time_s={wall_time:.3f}")
    print(f"mass_leak_pct={leak_pct:.6f}")
    print(
        "ranges="
        f"phi_m[{float(jnp.min(phi_m)):.6f},{float(jnp.max(phi_m)):.6f}] "
        f"phi_c[{float(jnp.min(phi_c)):.6f},{float(jnp.max(phi_c)):.6f}] "
        f"c[{float(jnp.min(c)):.6f},{float(jnp.max(c)):.6f}]"
    )
    assert result.state_final.t == pytest.approx(5.0)
