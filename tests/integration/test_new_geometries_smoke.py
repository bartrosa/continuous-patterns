"""Smoke: new cavity geometries run a short Stage I integration without NaNs."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "yaml_name",
    [
        "elliptic_pinning",
        "polygon_pinning",
        "wedge_pinning",
        "rectangular_slot_pinning",
    ],
)
def test_new_geometry_runs_without_nan(yaml_name: str, tmp_path: Path) -> None:
    cfg = load_run_config(REPO_ROOT / "experiments" / "canonical" / f"{yaml_name}.yaml")
    cfg["geometry"]["n"] = 128
    cfg["time"]["T"] = 0.5
    cfg.setdefault("output", {})
    cfg["output"]["save_final_state"] = False
    cfg["output"]["include_params_panel"] = False
    result = run_one(cfg, results_root=tmp_path, write_artifacts=False, show_progress=False)
    pm = jnp.asarray(result.state_final.phi_m)
    pc = jnp.asarray(result.state_final.phi_c)
    cc = jnp.asarray(result.state_final.c)
    assert jnp.all(jnp.isfinite(pm))
    assert jnp.all(jnp.isfinite(pc))
    assert jnp.all(jnp.isfinite(cc))
