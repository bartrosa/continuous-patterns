"""Regression-style checks for ``agate_ch`` solver parameters and nested configs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_flat_cfg_matches_pre_flag_behavior() -> None:
    """Configs without ``enable_*`` keys behave like before (reaction on, Dirichlet on)."""
    from continuous_patterns.agate_ch.solver import cfg_to_sim_params

    cfg = {
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
        "use_ratchet": True,
        "phi_m_ratchet_low": 0.3,
        "phi_m_ratchet_high": 0.5,
    }
    prm = cfg_to_sim_params(cfg)
    assert prm.disable_dirichlet is False
    assert prm.enable_reaction == 1.0
    assert prm.apply_cavity_mask is True
    assert prm.project_c_on_cavity is True
    assert prm.c0_alpha == 0.0


def test_enable_flags_override() -> None:
    from continuous_patterns.agate_ch.solver import cfg_to_sim_params

    cfg = {
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
        "enable_reaction": False,
        "enable_dirichlet": False,
    }
    prm = cfg_to_sim_params(cfg)
    assert prm.enable_reaction == 0.0
    assert prm.disable_dirichlet is True
    assert prm.apply_cavity_mask is True


def test_apply_cavity_mask_can_be_disabled() -> None:
    """Explicit ``apply_cavity_mask: false`` reaches SimParams (full-domain experiments)."""
    from continuous_patterns.agate_ch.solver import cfg_to_sim_params

    cfg = {
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
        "apply_cavity_mask": False,
    }
    assert cfg_to_sim_params(cfg).apply_cavity_mask is False


def test_project_c_on_cavity_can_be_disabled() -> None:
    from continuous_patterns.agate_ch.solver import cfg_to_sim_params

    cfg = {
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
        "project_c_on_cavity": False,
    }
    assert cfg_to_sim_params(cfg).project_c_on_cavity is False


def test_flatten_stage_sequence_run_a_yaml() -> None:
    pytest.importorskip("yaml")
    import yaml

    from continuous_patterns.agate_ch.run import flatten_nested_cfg

    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "agate_ch" / "stage_sequence" / "run_a_stage1.yaml").read_text()
    )
    flat = flatten_nested_cfg(raw)
    assert flat["initial_condition"] == "cavity"
    assert flat["save_final_state"] is True
    assert flat["gamma"] == 3.0


def test_flatten_stage_sequence_run_b_long_yaml() -> None:
    pytest.importorskip("yaml")
    import yaml

    from continuous_patterns.agate_ch.run import flatten_nested_cfg

    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "agate_ch" / "stage_sequence" / "run_b_long.yaml").read_text()
    )
    flat = flatten_nested_cfg(raw)
    assert flat["experiment_name"] == "stage_seq_run_b_long"
    assert flat["initial_condition"] == "from_snapshot"
    assert flat["apply_cavity_mask"] is True
    assert flat["project_c_on_cavity"] is False
    assert flat["T"] == 100000.0
    assert flat["snapshot_every"] == 2000


def test_flatten_gravity_yaml_c0_alpha() -> None:
    pytest.importorskip("yaml")
    import yaml

    from continuous_patterns.agate_ch.run import flatten_nested_cfg

    raw = yaml.safe_load(
        (REPO_ROOT / "configs" / "agate_ch" / "gravity" / "alpha_0_10.yaml").read_text()
    )
    flat = flatten_nested_cfg(raw)
    assert flat["c0_alpha"] == 0.1
    assert flat["experiment_name"] == "gravity_alpha_0_10"


def test_initial_state_from_snapshot_roundtrip(tmp_path: Path) -> None:
    import jax.numpy as jnp

    from continuous_patterns.agate_ch.model import build_geometry
    from continuous_patterns.agate_ch.solver import initial_state_from_snapshot

    n = 20
    L, R = 2.0, 0.8
    geom = build_geometry(L, R, n)
    rng = np.random.default_rng(0)
    pm = rng.normal(size=(n, n)).astype(np.float32) * 0.02
    pc = rng.normal(size=(n, n)).astype(np.float32) * 0.02
    cc = np.full((n, n), 0.25, dtype=np.float32)
    npz_path = tmp_path / "final_state.npz"
    np.savez_compressed(npz_path, phi_m=pm, phi_c=pc, c=cc)
    meta_path = tmp_path / "final_state_meta.json"
    meta_path.write_text(json.dumps({"grid": n, "L": float(L)}, indent=2))

    cj, pmj, pcj = initial_state_from_snapshot(geom, npz_path, expected_L=L, expected_n=n)
    assert cj.shape == (n, n)
    assert float(jnp.max(jnp.abs(pmj - jnp.asarray(pm)))) < 1e-5
    assert float(jnp.max(jnp.abs(pcj - jnp.asarray(pc)))) < 1e-5
    assert float(jnp.max(jnp.abs(cj - jnp.asarray(cc)))) < 1e-5
