"""v1.5 acceptance tests (overshoot, short runs)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from continuous_patterns.agate_ch.diagnostics import overshoot_slack_fraction
from continuous_patterns.agate_ch.model import dfdphi_total
from continuous_patterns.agate_ch.solver import integrate_chunks


def test_barrier_derivative_shape() -> None:
    p = jnp.ones((3, 3)) * 0.5
    g = dfdphi_total(p, 1.0, 10.0)
    assert g.shape == p.shape


def test_overshoot_slack_after_1k_steps() -> None:
    """<1% of pixels outside [-0.05,1.05] after 1000 steps (v1.5-style parameters)."""
    cfg = {
        "grid": 64,
        "L": 200.0,
        "R": 80.0,
        "W": 1.0,
        "gamma": 4.0,
        "kappa": 0.5,
        "lambda_barrier": 10.0,
        "D_c": 1.0,
        "k_reaction": 0.5,
        "M_m": 0.01,
        "M_c": 1.0,
        "c_sat": 0.2,
        "c_0": 1.0,
        "c_ostwald": 0.6,
        "w_ostwald": 0.1,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
        "use_ratchet": True,
        "dt": 0.01,
        "T": 10.0,
        "snapshot_every": 999999,
        "seed": 42,
        "uniform_supersaturation": False,
        "progress": False,
    }
    _, pm, pc, _ = integrate_chunks(cfg, chunk_size=250, on_snapshot=None)
    pm = np.asarray(pm)
    pc = np.asarray(pc)
    frac = overshoot_slack_fraction(pm, pc)
    assert frac < 1.0, f"slack overshoot {frac}% >= 1%"
