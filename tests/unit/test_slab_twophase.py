"""Unit tests for ``continuous_patterns.models.slab_twophase`` (v6 reference kernel)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from pydantic import ValidationError

from continuous_patterns.core.io import load_run_config, validate_run_config_dict
from continuous_patterns.models.slab_twophase import (
    DEFAULT_PHI_GATE,
    KERNEL_W,
    build_geometry,
    build_params,
    ch_spinodal_wavelength,
    estimate_integration_steps,
    simulate,
)
from continuous_patterns.models.twophase_kernel_reference import run as reference_run


def test_geometry_cell_centers() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["geometry"]["n"] = 100
    geom = build_geometry(cfg)
    assert geom.dx == pytest.approx(geom.L / geom.n)
    assert geom.x[0] == pytest.approx(0.5 * geom.dx)


def test_ch_config_rejects_W() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["physics"]["demixing"] = "ch"
    cfg["physics"]["kappa"] = 0.5
    cfg["physics"]["M0"] = 1.0
    cfg["physics"]["W"] = 1.0
    with pytest.raises(ValidationError, match="W"):
        validate_run_config_dict(cfg)


def test_ch_spectral_dt_limit() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["physics"]["demixing"] = "ch"
    cfg["physics"]["kappa"] = 0.5
    cfg["physics"]["M0"] = 1.0
    dt, n_steps = estimate_integration_steps(cfg)
    geom = build_geometry(cfg)
    prm = build_params(cfg)
    qm = np.pi / geom.dx
    fpp_abs = 4.0 * KERNEL_W + 4.0 * prm.gamma
    dt_ch = 2.0 / (prm.M0 * qm * qm * (fpp_abs + prm.kappa * qm * qm))
    expected = 0.4 * min(geom.dx**2 / (2.0 * prm.D_c), 1.0 / (prm.k_rxn * 1.0), dt_ch)
    assert dt == pytest.approx(expected, rel=1e-9)
    assert n_steps > 0


def test_integrated_v1_matches_reference() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["geometry"]["n"] = 200
    cfg["time"]["T"] = 4000.0
    cfg.setdefault("output", {})["save_evolution"] = False
    ref = reference_run(mode="ac", bc_right="dirichlet", n=200, T=4000.0)
    res = simulate(cfg, show_progress=False)
    assert int(res.diagnostics["n_projection_events"]) == 0
    assert float(res.diagnostics["mass_residual_pct"]) == pytest.approx(
        float(ref["resid_pct"]), rel=1e-6, abs=1e-12
    )
    phi_m = np.asarray(res.state_final.phi_m).ravel()
    phi_c = np.asarray(res.state_final.phi_c).ravel()
    np.testing.assert_allclose(phi_m, ref["phi_m"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(phi_c, ref["phi_c"], rtol=1e-10, atol=1e-12)


def test_clip_rate_in_diagnostics_not_clipped_frac() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["geometry"]["n"] = 64
    cfg["time"]["T"] = 200.0
    cfg.setdefault("output", {})["save_evolution"] = False
    res = simulate(cfg, show_progress=False)
    assert "clipped_mass_frac" not in res.diagnostics
    assert res.diagnostics.get("clip_rate") is not None


def test_phi_gate_default() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    prm = build_params(cfg)
    assert prm.phi_gate == DEFAULT_PHI_GATE


def test_e6_closed_short_run() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["physics"]["bc_right"] = "noflux"
    cfg["initial"]["c_init"] = 0.5
    cfg["time"]["T"] = 3000.0
    cfg.setdefault("output", {})["save_evolution"] = False
    res = simulate(cfg, show_progress=False)
    assert float(res.diagnostics["mass_residual_pct"]) < 5.0
    assert float(res.diagnostics["max_abs_right_flux"]) == pytest.approx(0.0, abs=1e-12)


def test_ac_T1000_wall_time_budget() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["time"]["T"] = 1000.0
    cfg.setdefault("output", {})["save_evolution"] = False
    t0 = time.perf_counter()
    res = simulate(cfg, show_progress=False)
    elapsed = time.perf_counter() - t0
    n_steps = int(res.diagnostics["n_steps"])
    assert n_steps < 200_000
    assert elapsed < 45.0
    assert n_steps / elapsed > 800.0


def test_lambda_ch_formula() -> None:
    lam = ch_spinodal_wavelength(gamma=4.0, kappa=0.5)
    assert lam == pytest.approx(3.14159265, rel=1e-4)
