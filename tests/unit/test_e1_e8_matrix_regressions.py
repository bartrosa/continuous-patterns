"""Regressions for the E1–E8 banding matrix (v6 reference kernel + integrity)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from continuous_patterns.core.io import load_run_config, save_run_config, validate_run_config_dict
from continuous_patterns.models.slab_twophase import (
    KERNEL_W,
    build_geometry,
    build_params,
    build_time_params,
    ch_spinodal_wavelength,
    estimate_integration_steps,
    simulate,
)

REPO = Path(__file__).resolve().parents[2]
MATRIX_YAML = REPO / "experiments/comparisons/e1_e8_matrix_v6.yaml"
SCRIPT = REPO / "scripts/run_banding_comparison.py"


def _banding_script():
    spec = importlib.util.spec_from_file_location("run_banding_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _matrix_entries() -> list[dict]:
    raw = yaml.safe_load(MATRIX_YAML.read_text(encoding="utf-8"))
    return list(raw["comparison"]["experiments"])


def _entry(label: str) -> dict:
    for entry in _matrix_entries():
        if entry["run_label"] == label:
            return entry
    raise KeyError(label)


@pytest.fixture(scope="module")
def banding():
    return _banding_script()


def test_matrix_yaml_preflight_passes(banding) -> None:
    banding._preflight_comparison_experiments(_matrix_entries(), REPO)


def test_e7_merged_config_validates_and_serializes(banding, tmp_path: Path) -> None:
    cfg = banding._build_run_config(_entry("E7_TP_gamma4_CH_baselineOstwald"), REPO)
    assert cfg["physics"]["demixing"] == "ch"
    assert int(cfg["geometry"]["n"]) == 256
    assert float(cfg["time"]["T"]) == 6000.0
    assert "W" not in cfg["physics"]
    assert "kappa_s" not in cfg["physics"]
    save_run_config(tmp_path / "config.yaml", cfg)


def test_e8_merged_config(banding) -> None:
    cfg = banding._build_run_config(_entry("E8_TP_gamma4_CH_matchedOstwald"), REPO)
    assert cfg["physics"]["c_ostwald"] == pytest.approx(0.25)
    assert cfg["physics"]["w_ostwald"] == pytest.approx(0.05)


def test_e7_unpruned_baseline_rejected() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg.setdefault("physics", {})
    cfg["physics"]["demixing"] = "ch"
    cfg["physics"]["kappa"] = 0.5
    cfg["physics"]["M0"] = 1.0
    with pytest.raises(ValidationError, match="kappa_s|L_s"):
        validate_run_config_dict(cfg)


def test_e7_ch_uses_spectral_dt_limit(banding) -> None:
    cfg = banding._build_run_config(_entry("E7_TP_gamma4_CH_baselineOstwald"), REPO)
    dt, n_steps = estimate_integration_steps(cfg)
    geom = build_geometry(cfg)
    prm = build_params(cfg)
    tm = build_time_params(cfg)
    qm = np.pi / geom.dx
    fpp_abs = 4.0 * KERNEL_W + 4.0 * prm.gamma
    dt_ch = 2.0 / (prm.M0 * qm * qm * (fpp_abs + prm.kappa * qm * qm))
    expected = float(tm.dt_safety) * min(
        geom.dx**2 / (2.0 * prm.D_c),
        1.0 / (prm.k_rxn * 0.3),
        dt_ch,
    )
    assert abs(dt - expected) / expected < 1e-9
    assert dt < 2e-3
    assert n_steps > 100_000


def test_e6_config_builds(banding) -> None:
    cfg = banding._build_run_config(_entry("E6_TP_gamma4_closed"), REPO)
    banding._verify_physics_overrides(_entry("E6_TP_gamma4_closed"), cfg)
    assert cfg["physics"]["bc_right"] == "noflux"
    assert cfg["initial"]["c_init"] == 0.5


def test_e6_canonical_mass_not_91pct() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    cfg["physics"]["bc_right"] = "noflux"
    cfg["initial"]["c_init"] = 0.5
    cfg["time"]["T"] = 3000.0
    cfg.setdefault("output", {})["save_evolution"] = False
    res = simulate(cfg, show_progress=False)
    pct = float(res.diagnostics["mass_residual_pct"])
    assert pct < 5.0
    assert float(res.diagnostics["max_abs_right_flux"]) == pytest.approx(0.0, abs=1e-12)


def test_lambda_ch_e7_value() -> None:
    lam = ch_spinodal_wavelength(gamma=4.0, kappa=0.5)
    assert lam == pytest.approx(3.14159265, rel=1e-4)


def test_tp_high_clip_rate_stays_ok(banding) -> None:
    cfg = {"physics": {"demixing": "ch", "bc_right": "dirichlet"}, "geometry": {"n": 256}}
    status, reasons = banding._row_integrity_status(
        "slab_twophase",
        cfg,
        {
            "mass_residual_pct": 1e-11,
            "min_phi_w": 0.0,
            "n_projection_events": 745_000,
            "n_steps": 23_450_605,
        },
    )
    assert status == "OK"
    assert reasons == []


def test_tp_projection_clipped_rule_ok_with_tiny_clips(banding) -> None:
    cfg = {"physics": {"demixing": "ch", "bc_right": "dirichlet"}, "geometry": {"n": 256}}
    status, reasons = banding._row_integrity_status(
        "slab_twophase",
        cfg,
        {
            "mass_residual_pct": 1e-11,
            "min_phi_w": 0.0,
            "n_projection_events": 5,
            "n_steps": 1_000_000,
        },
    )
    assert status == "OK"
    assert reasons == []


def test_open_twophase_min_phi_w_zero_not_invalid(banding) -> None:
    cfg = {"physics": {"demixing": "ac", "bc_right": "dirichlet"}}
    status, reasons = banding._row_integrity_status(
        "slab_twophase",
        cfg,
        {"min_phi_w": 0.0, "n_projection_events": 0, "clipped_mass_frac": 0.0},
    )
    assert status == "OK"
    assert reasons == []


def test_invalid_laws_stub_cv_q_tail_no_keyerror() -> None:
    laws = {
        "n_bands": 0,
        "n_segments": 0,
        "p_fit": None,
        "Q_cv_tail": None,
        "p_fit_excl_last": None,
        "Q_cv_tail_excl_last": None,
        "laws_reason": "row_invalid",
        "time_law": {"alpha": None},
    }
    cv_tail = laws.get("Q_cv_tail", laws.get("Q_cv"))
    assert cv_tail is None
