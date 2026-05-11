"""Unit tests for ``continuous_patterns.models.slab_kinetic``."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.models.slab_kinetic import (
    KineticParams,
    build_geometry,
    build_initial_state,
    simulate,
)


def _minimal_cfg(*, n: int = 64, T: float = 0.02, profile: str = "linear") -> dict:
    return {
        "experiment": {"name": "u", "model": "slab_kinetic", "seed": 1},
        "geometry": {"type": "kinetic_1d", "n": int(n)},
        "physics": {"Da": 1.0, "kappa": 3.0, "c1": 0.49, "c2": 0.10},
        "stress": {"mode": "none"},
        "gravity": {},
        "time": {"T": float(T), "dt": 0.01, "dt_safety": 0.4, "snapshot_every": 1000},
        "output": {},
        "initial": {"profile": profile},
    }


def test_kappa_must_exceed_one() -> None:
    with pytest.raises(ValueError, match="kappa"):
        KineticParams(Da=1.0, kappa=0.5, c1=0.5, c2=0.1)


def test_c1_must_exceed_c2() -> None:
    with pytest.raises(ValueError, match="c1"):
        KineticParams(Da=1.0, kappa=2.0, c1=0.1, c2=0.5)


def test_initial_linear_profile() -> None:
    cfg = _minimal_cfg(n=11, T=1e-9)
    geom = build_geometry(cfg)
    prm = KineticParams(Da=1.0, kappa=3.0, c1=0.49, c2=0.10)
    st = build_initial_state(cfg, geom, prm)
    np.testing.assert_allclose(st.c, geom.x)


def test_baseline_runs_without_nan() -> None:
    cfg = _minimal_cfg(n=48, T=0.001)
    res = simulate(cfg, show_progress=False)
    c = np.asarray(res.state_final.c).ravel()
    assert np.all(np.isfinite(c))


def test_dirichlet_bc_preserved() -> None:
    cfg = _minimal_cfg(n=32, T=0.002)
    res = simulate(cfg, show_progress=False)
    c = np.asarray(res.state_final.c).ravel()
    assert float(c[-1]) == pytest.approx(1.0)


def test_robin_bc_satisfied_at_final_step() -> None:
    cfg = _minimal_cfg(n=64, T=0.002)
    res = simulate(cfg, show_progress=False)
    sk = (res.meta or {}).get("slab_kinetic", {})
    prm = cfg["physics"]
    dx = 1.0 / (int(cfg["geometry"]["n"]) - 1)
    c = np.asarray(res.state_final.c, dtype=np.float64).ravel()
    r = 3.0 if str(sk.get("state_final_str")) == "H" else 1.0
    lhs = float((c[1] - c[0]) / dx)
    rhs = float(r * float(prm["Da"]) * c[0])
    assert lhs == pytest.approx(rhs, rel=1e-4, abs=1e-6)


def test_hysteresis_produces_transitions() -> None:
    """Poster-style parameters; at least one L↔H event must occur for finite T."""
    cfg = {
        "experiment": {"name": "bands", "model": "slab_kinetic", "seed": 1},
        "geometry": {"type": "kinetic_1d", "n": 200},
        "physics": {"Da": 1.0, "kappa": 3.0, "c1": 0.49, "c2": 0.10},
        "stress": {"mode": "none"},
        "gravity": {},
        "time": {"T": 5.0, "dt": 0.01, "dt_safety": 0.4, "snapshot_every": 100},
        "output": {},
        "initial": {"profile": "linear"},
    }
    res = simulate(cfg, show_progress=False)
    n_tr = int(res.diagnostics.get("n_transitions", 0))
    assert n_tr >= 1
