"""D5H hysteresis snapshot diagnostics (Da, consume_front)."""

from __future__ import annotations

import numpy as np

from continuous_patterns.core.io import load_run_config
from continuous_patterns.models.twophase_kernel_d1 import run_hysteresis


def test_hysteresis_da_snapshots_short() -> None:
    cfg = load_run_config("experiments/canonical/slab_twophase/baseline.yaml")
    phys = cfg["physics"]
    kr = run_hysteresis(
        L=100.0,
        n=64,
        T=80.0,
        D=0.03,
        k_rxn=float(phys["k_rxn"]),
        c_sat=0.05,
        c0=0.35,
        c_init=0.3,
        c1=0.30,
        c2=0.15,
        kappa_rate=5.0,
        k_L_scale=3.0,
        n_frames=20,
    )
    da = np.asarray(kr["snapshots_Da"], dtype=np.float64)
    consume = np.asarray(kr["snapshots_consume_front"], dtype=np.float64)
    gap = np.asarray(kr["snapshots_gap_width"], dtype=np.float64)
    assert da.size > 0
    assert consume.size == da.size
    assert gap.size == da.size
    assert np.all(np.isfinite(da))
    assert np.all(da >= 0.0)
