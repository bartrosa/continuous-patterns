"""Aging: ``aging_demo`` (quartz) and ``scenario_closed_aging`` (seeded conversion)."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one
from continuous_patterns.models import cavity_reactive


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
    result = cavity_reactive.simulate(cfg, chunk_size=100, show_progress=False)
    iq = float(result.diagnostics.get("phi_q_chi_weighted_integral", 0.0))
    assert iq > 0.0


def test_aging_seeded_closed_aging_runs_and_converts(tmp_path: Path) -> None:
    """closed_aging with preset seeds (phi_c_init=0.05) must show phi_c growth and phi_m drop."""
    repo = Path(__file__).resolve().parents[2]
    yml = repo / "experiments" / "canonical" / "scenario_closed_aging.yaml"
    cfg = load_run_config(yml)
    cfg["geometry"]["n"] = 128
    cfg["time"]["T"] = 5.0
    # dt=0.01 from defaults → 500 steps; local nuclei can grow while χ-mean φ_c
    # stays low (kinetic trap in matrix still retracts the background level).
    cfg.setdefault("output", {})
    cfg["output"]["save_final_state"] = True
    cfg["output"]["record_spectral_mass_diagnostic"] = False

    result = run_one(cfg, results_root=tmp_path, write_artifacts=False, show_progress=False)
    geom = cavity_reactive.build_geometry(result.config_resolved)
    chi = np.asarray(jax.device_get(geom.chi), dtype=np.float64)
    w = float(chi.sum()) + 1e-30

    pm = np.asarray(jax.device_get(result.state_final.phi_m), dtype=np.float64)
    pc = np.asarray(jax.device_get(result.state_final.phi_c), dtype=np.float64)
    phi_m_mean = float(np.sum(pm * chi) / w)
    max_pc = float(np.max(pc * chi))

    # Nucleation sites: local φ_c can exceed 0.1 even when the χ-mean drops (trap + diffusion).
    assert max_pc > 0.08, f"no growing chalcedony nuclei: max(phi_c) in cavity={max_pc}"
    assert phi_m_mean < 0.91, f"phi_m did not decrease from 0.95: mean={phi_m_mean}"
    assert np.all(np.isfinite(pc))
    assert np.all(np.isfinite(pm))
