"""Short-run smoke tests for new lithospheric / Kirsch / Inglis stress modes."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models.cavity_reactive import simulate

_REPO = Path(__file__).resolve().parents[2]
_CANON = _REPO / "experiments" / "canonical"


@pytest.mark.parametrize(
    "name",
    [
        "stress_lithostatic",
        "stress_tectonic",
        "stress_kirsch",
        "stress_inglis",
    ],
)
def test_lithospheric_stress_canonical_short_run(name: str) -> None:
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
    res = simulate(cfg, chunk_size=50, show_progress=False)
    for arr in (
        res.state_final.phi_m,
        res.state_final.phi_c,
        res.state_final.c,
    ):
        assert jnp.all(jnp.isfinite(arr))
