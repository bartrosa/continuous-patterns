"""Integration: comparison YAML overrides reach the solver-facing config."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MATRIX_V4 = REPO / "experiments/comparisons/e1_e7_matrix_v4.yaml"
SCRIPT = REPO / "scripts/run_banding_comparison.py"


def _banding_script():
    spec = importlib.util.spec_from_file_location("run_banding_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(label: str) -> dict:
    raw = yaml.safe_load(MATRIX_V4.read_text(encoding="utf-8"))
    for entry in raw["comparison"]["experiments"]:
        if entry["run_label"] == label:
            return entry
    raise KeyError(label)


def test_e6_bc_right_reaches_solver_config() -> None:
    """v3 bug: initial.c_init applied but physics.bc_right did not (resume/stale disk)."""
    banding = _banding_script()
    cfg = banding._build_run_config(_entry("E6_TP_gamma4_closed"), REPO)
    banding._verify_physics_overrides(_entry("E6_TP_gamma4_closed"), cfg)
    assert cfg["physics"]["bc_right"] == "noflux"
    assert cfg["initial"]["c_init"] == 0.5


def test_e7_ch_overrides_and_dt_limit() -> None:
    banding = _banding_script()
    matrix_v6 = REPO / "experiments/comparisons/e1_e8_matrix_v6.yaml"
    raw = yaml.safe_load(matrix_v6.read_text(encoding="utf-8"))
    entry = next(
        e
        for e in raw["comparison"]["experiments"]
        if e["run_label"] == "E7_TP_gamma4_CH_baselineOstwald"
    )
    cfg = banding._build_run_config(entry, REPO)
    banding._verify_physics_overrides(entry, cfg)
    assert cfg["physics"]["demixing"] == "ch"
    assert "W" not in cfg["physics"]
    assert float(cfg["time"]["T"]) == 6000.0
    from continuous_patterns.models.slab_twophase import estimate_integration_steps

    dt, n_steps = estimate_integration_steps(cfg)
    assert dt < 2e-3, f"CH dt={dt:.4e} should be spectral limit"
    assert dt > 1e-5
    assert n_steps > 100_000
