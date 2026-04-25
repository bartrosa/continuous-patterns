"""Integration smoke tests for Stage II bulk driver."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from continuous_patterns.models.bulk_relaxation import simulate


def test_agate_stage2_smoke_short_run() -> None:
    """End-to-end Stage II: mixed IC evolves; structure factor is finite."""
    cfg = {
        "experiment": {"name": "smoke_stage2", "model": "bulk_relaxation"},
        "geometry": {"type": "circular_cavity", "L": 10.0, "R": 0.0, "n": 32},
        "physics": {
            "W": 1.0,
            "gamma": 5.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 1.0,
            "M_c": 1.0,
            "D_c": 0.0,
            "lambda_bar": 10.0,
            "k_rxn": 0.0,
            "c_sat": 0.0,
            "c_0": 0.0,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "use_ratchet": False,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 5.0, "snapshot_every": 100},
        "output": {"save_final_state": True},
        "initial": {
            "phi_m_init": 0.5,
            "phi_m_noise": 0.05,
            "phi_c_init": 0.5,
            "phi_c_noise": 0.05,
        },
    }

    result = simulate(cfg, chunk_size=100, show_progress=False)

    assert result.state_final.phi_m.shape == (32, 32)
    assert result.state_final.t == pytest.approx(5.0)

    assert jnp.all(jnp.isfinite(result.state_final.phi_m))
    assert jnp.all(jnp.isfinite(result.state_final.phi_c))

    var_final = float(jnp.var(result.state_final.phi_m - result.state_final.phi_c))
    assert var_final > 0.01

    assert "structure_factor" in result.diagnostics or "coarsening_metrics" in result.diagnostics
    assert "dirichlet_mass_balance" in result.diagnostics
    dmb = result.diagnostics["dirichlet_mass_balance"]
    assert float(dmb["cumulative_injection"]) == pytest.approx(0.0, abs=1e-20)
    assert float(dmb["residual_pct"]) < 1.0
    assert "n_bands" not in result.diagnostics

    assert result.config_resolved["experiment"]["name"] == "smoke_stage2"
