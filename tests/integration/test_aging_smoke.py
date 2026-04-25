"""Short run with aging + α-quartz enabled (``experiments/canonical/aging_demo.yaml``)."""

from __future__ import annotations

from pathlib import Path

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models import agate_ch


def test_aging_demo_short_run(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    yml = repo / "experiments" / "canonical" / "aging_demo.yaml"
    cfg = load_run_config(yml)
    cfg["geometry"]["n"] = 64
    cfg["time"]["dt"] = 0.02
    cfg["time"]["T"] = 2.0
    cfg.setdefault("output", {})
    cfg["output"]["record_spectral_mass_diagnostic"] = False
    cfg["output"]["snapshot_every"] = 10**9
    result = agate_ch.simulate(cfg, chunk_size=100, show_progress=False)
    iq = float(result.diagnostics.get("phi_q_chi_weighted_integral", 0.0))
    assert iq > 0.0
