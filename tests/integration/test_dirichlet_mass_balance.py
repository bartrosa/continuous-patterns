"""Regression checks for Dirichlet mass-balance bookkeeping."""

from __future__ import annotations

from continuous_patterns.models.cavity_reactive import simulate as simulate_cavity
from continuous_patterns.models.slab_reactive import simulate as simulate_slab


def _base_cfg(model: str, geometry: dict) -> dict:
    return {
        "experiment": {"name": f"mass_balance_{model}", "model": model, "seed": 42},
        "geometry": geometry,
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
        "time": {"dt": 0.01, "T": 1.0, "snapshot_every": 100},
        "output": {"save_final_state": False, "record_spectral_mass_diagnostic": True},
        "initial": {"phi_m_init": 0.0, "phi_c_init": 0.0, "phi_m_noise": 0.0, "phi_c_noise": 0.0},
    }


def test_cavity_dirichlet_mass_balance_residual_under_5pct() -> None:
    cfg = _base_cfg(
        "cavity_reactive",
        {"type": "circular_cavity", "L": 100.0, "R": 30.0, "n": 64, "eps_scale": 2.0},
    )
    res = simulate_cavity(cfg, chunk_size=100, show_progress=False)
    dmb = res.diagnostics["dirichlet_mass_balance"]
    assert float(dmb["residual_pct"]) < 5.0


def test_slab_dirichlet_mass_balance_residual_under_5pct() -> None:
    cfg = _base_cfg(
        "slab_reactive",
        {"type": "slab", "L": 100.0, "n": 64, "eps_scale": 2.0, "rim_width_px": 2},
    )
    res = simulate_slab(cfg, chunk_size=100, show_progress=False)
    dmb = res.diagnostics["dirichlet_mass_balance"]
    assert float(dmb["residual_pct"]) < 5.0
