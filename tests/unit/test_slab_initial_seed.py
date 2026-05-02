"""Unit tests for slab left-wall nucleation seed IC."""

from __future__ import annotations

import jax
import numpy as np

from continuous_patterns.models.slab_reactive import (
    build_geometry,
    build_initial_state,
    build_sim_params,
)


def _base_cfg(initial: dict) -> dict:
    return {
        "experiment": {"name": "slab_seed_test", "model": "slab_reactive", "seed": 42},
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
        "time": {"dt": 0.01, "T": 1.0},
        "output": {"save_final_state": False},
        "initial": initial,
    }


def test_slab_seed_appears_on_left_wall() -> None:
    cfg = _base_cfg(
        {
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_noise_amplitude": 0.01,
            "phi_m_wall_layer": 0.5,
            "phi_m_wall_width_px": 2,
            "c_init": 0.2,
        }
    )
    geom = build_geometry(cfg)
    prm = build_sim_params(cfg)
    key = jax.random.PRNGKey(0)
    _key, k_ic = jax.random.split(key)
    phi_m, _phi_c, _phi_q, _phi_imp, _c = build_initial_state(cfg, geom, prm, k_ic)
    phi_m_np = np.asarray(phi_m)

    assert float(phi_m_np[:2, :].mean()) > 0.4
    assert float(phi_m_np[10:, :].mean()) < 0.1


def test_slab_no_seed_when_disabled() -> None:
    cfg = _base_cfg(
        {
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_noise_amplitude": 0.01,
            "phi_m_wall_layer": 0.0,
            "phi_m_wall_width_px": 0,
            "c_init": 0.2,
        }
    )
    geom = build_geometry(cfg)
    prm = build_sim_params(cfg)
    key = jax.random.PRNGKey(0)
    _key, k_ic = jax.random.split(key)
    phi_m, _phi_c, _phi_q, _phi_imp, _c = build_initial_state(cfg, geom, prm, k_ic)
    phi_m_np = np.asarray(phi_m)

    assert float(phi_m_np.mean()) < 0.05
