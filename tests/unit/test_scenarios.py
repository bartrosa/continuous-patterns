"""Unit tests for ``initial.scenario`` presets in :mod:`continuous_patterns.core.io`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from continuous_patterns.core.io import load_run_config


def _minimal_agate_ch_card(extra: str) -> str:
    return f"""
experiment:
  name: scen_unit
  model: cavity_reactive
  seed: 1
geometry:
  type: circular_cavity
  L: 20.0
  n: 32
  R: 5.0
  eps_scale: 2.0
physics:
  W: 1.0
  gamma: 2.0
  kappa_x: 0.5
  kappa_y: 0.5
  M_m: 0.1
  M_c: 1.0
  D_c: 0.1
  k_rxn: 0.5
  c_sat: 0.2
  c_0: 0.5
  lambda_bar: 10.0
  c_ostwald: 0.5
  w_ostwald: 0.1
  use_ratchet: true
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.05
  T: 1.0
  snapshot_every: 10
{extra}
"""


def test_scenario_closed_aging_preset_values(tmp_path: Path) -> None:
    p = tmp_path / "ca.yaml"
    p.write_text(
        _minimal_agate_ch_card(
            """
initial:
  scenario: closed_aging
"""
        ),
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["physics"]["reaction_active"] is False
    assert cfg["physics"]["dirichlet_active"] is False
    assert cfg["physics"]["aging"]["active"] is True
    assert cfg["physics"]["aging"]["k_age"] == pytest.approx(0.01)
    assert cfg["initial"]["phi_m_init"] == pytest.approx(0.95)
    assert cfg["initial"]["phi_c_init"] == pytest.approx(0.05)
    assert cfg["initial"]["phi_c_noise"] == pytest.approx(0.02)
    assert cfg["initial"]["c_init"] == pytest.approx(0.0)


def test_scenario_user_override_wins_on_nested_aging(tmp_path: Path) -> None:
    p = tmp_path / "ca_override.yaml"
    p.write_text(
        """
experiment:
  name: scen_unit
  model: cavity_reactive
  seed: 1
geometry:
  type: circular_cavity
  L: 20.0
  n: 32
  R: 5.0
  eps_scale: 2.0
physics:
  W: 1.0
  gamma: 2.0
  kappa_x: 0.5
  kappa_y: 0.5
  M_m: 0.1
  M_c: 1.0
  D_c: 0.1
  k_rxn: 0.5
  c_sat: 0.2
  c_0: 0.5
  lambda_bar: 10.0
  c_ostwald: 0.5
  w_ostwald: 0.1
  use_ratchet: true
  aging:
    k_age: 0.5
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.05
  T: 1.0
  snapshot_every: 10
initial:
  scenario: closed_aging
""",
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["physics"]["aging"]["k_age"] == pytest.approx(0.5)


def test_scenario_closed_supersaturated_c_init_factor(tmp_path: Path) -> None:
    p = tmp_path / "css.yaml"
    p.write_text(
        _minimal_agate_ch_card(
            """
initial:
  scenario: closed_supersaturated
"""
        ),
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["physics"]["dirichlet_active"] is False
    assert cfg["initial"]["c_init_factor"] == pytest.approx(1.5)


def test_scenario_unknown_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        _minimal_agate_ch_card(
            """
initial:
  scenario: not_a_real_scenario
"""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_run_config(p)


def test_scenario_bulk_relaxation_requires_stage2_model_in_card(tmp_path: Path) -> None:
    """Preset suggests Stage II; the experiment card must keep ``bulk_relaxation``."""
    p = tmp_path / "bulk.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "bulk_scen",
                    "model": "bulk_relaxation",
                    "seed": 1,
                },
                "geometry": {"type": "circular_cavity", "L": 20.0, "n": 32, "R": 0.0},
                "physics": {
                    "W": 1.0,
                    "gamma": 2.0,
                    "kappa_x": 0.5,
                    "kappa_y": 0.5,
                    "M_m": 1.0,
                    "M_c": 1.0,
                    "D_c": 0.0,
                    "lambda_bar": 10.0,
                    "k_rxn": 0.0,
                    "c_sat": 0.0,
                    "c_0": 0.0,
                    "c_ostwald": 0.5,
                    "w_ostwald": 0.1,
                    "use_ratchet": False,
                },
                "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
                "time": {"dt": 0.01, "T": 1.0, "snapshot_every": 10},
                "initial": {"scenario": "bulk_relaxation"},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["experiment"]["model"] == "bulk_relaxation"
    assert cfg["initial"]["phi_m_init"] == pytest.approx(0.5)


def test_canonical_scenario_yaml_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments" / "canonical" / "scenario_open_aging.yaml"
    if not path.is_file():
        pytest.skip("canonical scenario YAML not in workspace")
    load_run_config(path)
