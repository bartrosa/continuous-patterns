"""Short-run smoke tests for ``initial.scenario`` canonical cards."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models import bulk_relaxation, cavity_reactive

_REPO = Path(__file__).resolve().parents[2]
_CANON = _REPO / "experiments" / "canonical"


@pytest.mark.parametrize(
    "name,simulate_fn",
    [
        ("scenario_closed_supersaturated", cavity_reactive.simulate),
        ("scenario_closed_aging", cavity_reactive.simulate),
        ("scenario_open_aging", cavity_reactive.simulate),
        ("scenario_bulk_relaxation", bulk_relaxation.simulate),
    ],
)
def test_scenario_canonical_short_run(name: str, simulate_fn) -> None:
    path = _CANON / f"{name}.yaml"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    cfg = load_run_config(path)
    cfg["time"]["T"] = 0.5
    cfg["time"]["dt"] = 0.01
    cfg["geometry"]["n"] = 128
    cfg.setdefault("output", {})
    cfg["output"]["save_final_state"] = False
    cfg["output"]["record_spectral_mass_diagnostic"] = True
    if name == "scenario_bulk_relaxation":
        cfg["output"]["record_evolution_gif"] = False
        cfg["output"]["save_snapshots_h5"] = False
    res = simulate_fn(cfg, chunk_size=50, show_progress=False)
    assert jnp.all(jnp.isfinite(res.state_final.phi_m))
    assert jnp.all(jnp.isfinite(res.state_final.phi_c))
