"""Smoke test for gravity-enabled canonical card."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models.cavity_reactive import simulate

_REPO = Path(__file__).resolve().parents[2]


def test_gravity_demo_canonical_short_run() -> None:
    path = _REPO / "experiments" / "canonical" / "gravity_demo.yaml"
    cfg = load_run_config(path)
    cfg["time"]["T"] = 0.5
    cfg["time"]["dt"] = 0.01
    cfg["geometry"]["n"] = 128
    cfg.setdefault("output", {})
    cfg["output"]["save_final_state"] = False
    cfg["output"]["record_spectral_mass_diagnostic"] = True
    res = simulate(cfg, chunk_size=50, show_progress=False)
    assert jnp.all(jnp.isfinite(res.state_final.c))
