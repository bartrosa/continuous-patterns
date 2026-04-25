"""Integration: Stage I snapshot cadence with ``save_snapshots_h5`` / GIF pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from continuous_patterns.models.cavity_reactive import simulate


def test_snapshot_count_matches_schedule() -> None:
    """T=10, dt=0.1 → 100 steps; snapshot_every=2 → steps 0,2,...,100 → 51 frames."""
    cfg = {
        "experiment": {"name": "snap_test", "model": "cavity_reactive", "seed": 0},
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
        "time": {"dt": 0.1, "T": 10.0, "snapshot_every": 2},
        "output": {
            "save_final_state": True,
            "flux_sample_dt": 1.0,
            "save_snapshots_h5": True,
            "record_spectral_mass_diagnostic": False,
        },
    }

    result = simulate(cfg, chunk_size=2000, show_progress=False)

    snaps = result.meta.get("h5_snapshots", [])
    assert len(snaps) == 51
    assert snaps[0]["step"] == 0
    assert snaps[-1]["step"] == 100
    assert all(snaps[i]["step"] == i * 2 for i in range(len(snaps)))
    assert jnp.all(jnp.isfinite(jnp.asarray(snaps[5]["phi_m"])))


def test_gif_snapshots_align_with_h5_when_both_on() -> None:
    cfg = {
        "experiment": {"name": "snap_gif", "model": "cavity_reactive", "seed": 1},
        "geometry": {"type": "circular_cavity", "L": 8.0, "R": 2.0, "n": 16},
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
            "use_ratchet": False,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.05, "T": 0.5, "snapshot_every": 5},
        "output": {
            "save_final_state": True,
            "flux_sample_dt": 1.0,
            "save_snapshots_h5": True,
            "record_evolution_gif": True,
            "record_spectral_mass_diagnostic": False,
        },
    }
    result = simulate(cfg, chunk_size=2000, show_progress=False)
    h5s = result.meta["h5_snapshots"]
    gifs = result.meta["gif_snapshots"]
    assert len(h5s) == len(gifs)
    for hs, (tg, _) in zip(h5s, gifs, strict=True):
        assert hs["t"] == pytest.approx(tg)
