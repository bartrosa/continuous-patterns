"""Validation tests for physics.eta_wall."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models.slab_reactive import simulate


def test_eta_wall_can_exceed_c_sat(tmp_path: Path) -> None:
    card = """
experiment:
  name: eta_wall_bad
  model: slab_reactive
  seed: 42
geometry:
  type: slab
  L: 100.0
  n: 128
  rim_width_px: 2
  rim_left_width_px: 2
  eps_scale: 2.0
physics:
  W: 1.0
  gamma: 4.0
  kappa_x: 0.5
  kappa_y: 0.5
  M_m: 0.01
  M_c: 1.0
  D_c: 1.0
  k_rxn: 0.5
  c_sat: 0.2
  eta_wall: 1.0
  c_0: 1.0
  c_ostwald: 0.6
  w_ostwald: 0.1
  lambda_bar: 10.0
stress:
  mode: none
time:
  dt: 0.01
  T: 1.0
  snapshot_every: 10
"""
    p = tmp_path / "eta_wall_bad.yaml"
    p.write_text(card, encoding="utf-8")
    cfg = load_run_config(p)
    assert float(cfg["physics"]["eta_wall"]) == pytest.approx(1.0)


def test_eta_wall_above_one_rejected(tmp_path: Path) -> None:
    card = """
experiment:
  name: eta_wall_too_high
  model: slab_reactive
  seed: 42
geometry:
  type: slab
  L: 30.0
  n: 32
  rim_width_px: 2
  rim_left_width_px: 2
  eps_scale: 2.0
physics:
  W: 1.0
  gamma: 4.0
  kappa_x: 0.5
  kappa_y: 0.5
  M_m: 0.01
  M_c: 1.0
  D_c: 1.0
  k_rxn: 0.5
  c_sat: 0.2
  eta_wall: 1.1
  c_0: 1.0
  c_ostwald: 0.6
  w_ostwald: 0.1
  lambda_bar: 10.0
stress:
  mode: none
time:
  dt: 0.01
  T: 0.1
  snapshot_every: 10
"""
    p = tmp_path / "eta_wall_too_high.yaml"
    p.write_text(card, encoding="utf-8")
    with pytest.raises(ValidationError, match="eta_wall"):
        load_run_config(p)


def test_eta_wall_one_step_no_nan() -> None:
    cfg = {
        "experiment": {"name": "eta_wall_step_smoke", "model": "slab_reactive", "seed": 0},
        "geometry": {"type": "slab", "L": 30.0, "n": 32, "rim_width_px": 2, "rim_left_width_px": 2},
        "physics": {
            "W": 1.0,
            "gamma": 4.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.01,
            "M_c": 1.0,
            "D_c": 1.0,
            "k_rxn": 0.5,
            "c_sat": 0.2,
            "eta_wall": 1.0,
            "c_0": 1.0,
            "c_ostwald": 0.6,
            "w_ostwald": 0.1,
            "lambda_bar": 10.0,
            "use_ratchet": True,
            "phi_m_ratchet_low": 0.3,
            "phi_m_ratchet_high": 0.5,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 0.01, "snapshot_every": 10},
        "output": {"save_final_state": False},
        "initial": {
            "c_init": 0.2,
            "phi_m_init": 0.0,
            "phi_c_init": 0.0,
            "phi_noise_amplitude": 0.01,
            "phi_m_wall_layer": 0.5,
            "phi_m_wall_width_px": 2,
        },
    }
    res = simulate(cfg, chunk_size=1, show_progress=False)
    assert jnp.all(jnp.isfinite(res.state_final.phi_m))
    assert jnp.all(jnp.isfinite(res.state_final.phi_c))
