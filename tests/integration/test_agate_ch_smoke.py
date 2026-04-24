"""Integration smoke tests for Stage I agate CH driver."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from continuous_patterns.models.agate_ch import simulate


def test_agate_ch_smoke_short_run(tmp_path: Path) -> None:
    """End-to-end: nested dict config → simulate → SimResult with finite fields."""
    _ = tmp_path
    cfg = {
        "experiment": {"name": "smoke_test", "model": "agate_ch"},
        "geometry": {"type": "circular_cavity", "L": 10.0, "R": 3.0, "n": 32},
        "physics": {
            "W": 1.0,
            "gamma": 2.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.1,
            "M_c": 1.0,
            "D_c": 0.1,
            "k_rxn": 0.5,
            "c_sat": 0.2,
            "c_0": 0.5,
            "lambda_bar": 10.0,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "use_ratchet": True,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 5.0, "snapshot_every": 100},
        "output": {
            "save_final_state": True,
            "flux_sample_dt": 2.0,
            "record_spectral_mass_diagnostic": False,
        },
    }

    result = simulate(cfg, chunk_size=100)

    assert result.state_final.phi_m.shape == (32, 32)
    assert result.state_final.t == pytest.approx(5.0)

    assert jnp.all(jnp.isfinite(result.state_final.phi_m))
    assert jnp.all(jnp.isfinite(result.state_final.phi_c))
    assert jnp.all(jnp.isfinite(result.state_final.c))

    assert "option_b_leak_pct" in result.diagnostics or "flux_samples" in result.meta

    assert result.config_resolved["experiment"]["name"] == "smoke_test"
