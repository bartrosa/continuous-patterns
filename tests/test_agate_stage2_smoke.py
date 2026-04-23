"""Smoke tests for ``agate_stage2`` (Stage II periodic bulk CH)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


pytest.importorskip("jax")
pytest.importorskip("yaml")


def test_baseline_yaml_flattens_and_stage2_geometry() -> None:
    import jax.numpy as jnp
    import yaml

    from continuous_patterns.agate_stage2.model import build_geometry
    from continuous_patterns.agate_stage2.run import flatten_nested_cfg
    from continuous_patterns.agate_stage2.solver import cfg_to_sim_params

    path = REPO_ROOT / "configs" / "agate_stage2" / "baseline.yaml"
    raw = yaml.safe_load(path.read_text())
    flat = flatten_nested_cfg(raw)

    assert flat["enable_reaction"] is False
    assert flat["enable_dirichlet"] is False
    assert flat["initial_condition"] == "homogeneous"
    assert flat["mass_balance_mode"] == "spectral_only"
    assert flat["flux_sample_dt"] == 0.0

    prm = cfg_to_sim_params(flat)
    assert prm.enable_reaction == 0.0
    assert prm.disable_dirichlet is True

    geom = build_geometry(float(flat["L"]), float(flat["R"]), int(flat["grid"]))
    assert jnp.allclose(geom.chi, 1.0)
    assert float(jnp.max(geom.ring)) == 0.0
    assert not bool(jnp.any(geom.ring_accounting))


def test_gamma_scan_yaml_matches_expected_layout() -> None:
    """γ-scan configs live alongside baseline under ``configs/agate_stage2/``."""
    import yaml

    from continuous_patterns.agate_stage2.run import flatten_nested_cfg

    path = REPO_ROOT / "configs" / "agate_stage2" / "gamma_5.yaml"
    assert path.is_file(), path
    raw = yaml.safe_load(path.read_text())
    flat = flatten_nested_cfg(raw)
    assert flat["gamma"] == 5.0
    assert flat["experiment_name"] == "stage2_gamma_5_0"
    assert flat["noise_amplitude"] == 0.05


def test_homogeneous_ic_shape_matches_grid() -> None:
    import jax.numpy as jnp

    from continuous_patterns.agate_stage2.model import build_geometry
    from continuous_patterns.agate_stage2.solver import initial_state_homogeneous

    n = 32
    geom = build_geometry(100.0, 40.0, n)
    c, pm, pc = initial_state_homogeneous(geom, seed=0)
    assert c.shape == (n, n)
    assert pm.shape == (n, n)
    assert jnp.isfinite(c).all() and jnp.isfinite(pm).all()
