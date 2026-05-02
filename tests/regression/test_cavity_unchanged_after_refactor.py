"""Regression guard for cavity_reactive numerical identity across refactors."""

from __future__ import annotations

import jax.numpy as jnp

from continuous_patterns.experiments.run import run_one

CAVITY_BASELINE_PARAMS = {
    "experiment": {"name": "regression_cavity", "model": "cavity_reactive", "seed": 42},
    "geometry": {"type": "circular_cavity", "L": 100.0, "n": 64, "R": 30.0, "eps_scale": 2.0},
    "physics": {
        "precipitation": {
            "model": "ostwald_basic",
            "params": {"k_rxn": 1.0, "c_sat": 0.2, "c_ostwald": 0.5, "w_ostwald": 0.05},
        },
        "D_c": 1.0,
        "c_0": 1.0,
        "kappa_x": 0.001,
        "kappa_y": 0.001,
        "lambda_bar": 10.0,
        "phases": {
            "moganite": {
                "potential": "double_well",
                "potential_kwargs": {"W": 1.0},
                "mobility": 0.01,
                "rho": 1.0,
                "psi_sign": 1.0,
                "active": True,
            },
            "chalcedony": {
                "potential": "double_well",
                "potential_kwargs": {"W": 1.0},
                "mobility": 0.1,
                "rho": 1.0,
                "psi_sign": -1.0,
                "active": True,
            },
        },
        "gamma": 4.0,
        "use_ratchet": True,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "k_rxn": 1.0,
        "c_sat": 0.2,
        "c_ostwald": 0.5,
        "w_ostwald": 0.05,
    },
    "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
    "initial": {
        "c_init": 0.2,
        "phi_m_init": 0.0,
        "phi_c_init": 0.0,
        "phi_m_noise": 0.01,
        "phi_c_noise": 0.01,
    },
    "time": {"dt": 0.01, "T": 0.05, "snapshot_every": 100},
    "output": {"save_final_state": False, "save_evolution": False},
}

EXPECTED_SUM_PHI_M = 10.867147445678711
EXPECTED_MAX_PHI_M = 0.045514754951000214
EXPECTED_MIN_PHI_M = -0.034236013889312744


def test_cavity_reactive_bitwise_unchanged_after_refactor() -> None:
    """Cavity short run must stay bitwise-identical to frozen baseline scalars."""
    result = run_one(
        CAVITY_BASELINE_PARAMS,
        write_artifacts=False,
        show_progress=False,
        chunk_size=50,
    )
    phi_m_final = result.state_final.phi_m
    sum_phi_m = float(jnp.sum(phi_m_final))
    max_phi_m = float(jnp.max(phi_m_final))
    min_phi_m = float(jnp.min(phi_m_final))
    print(f"sum_phi_m = {sum_phi_m!r}")
    print(f"max_phi_m = {max_phi_m!r}")
    print(f"min_phi_m = {min_phi_m!r}")
    assert sum_phi_m == EXPECTED_SUM_PHI_M
    assert max_phi_m == EXPECTED_MAX_PHI_M
    assert min_phi_m == EXPECTED_MIN_PHI_M
