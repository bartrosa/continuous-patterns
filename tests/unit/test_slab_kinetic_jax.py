"""Unit tests for the JAX ``slab_kinetic`` backend."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from continuous_patterns.models.slab_kinetic import simulate
from continuous_patterns.models.slab_kinetic_jax import BatchParams, simulate_batch


def _cfg(*, T: float = 2.0, c1: float = 0.49, c2: float = 0.10, n: int = 200) -> dict:
    return {
        "experiment": {"name": "jax_test", "model": "slab_kinetic", "seed": 42},
        "geometry": {"type": "kinetic_1d", "n": int(n)},
        "physics": {"Da": 1.0, "kappa": 3.0, "c1": float(c1), "c2": float(c2)},
        "stress": {"mode": "none"},
        "gravity": {},
        "time": {"T": float(T), "dt_safety": 0.4, "snapshot_every": 1000},
        "output": {},
        "initial": {"profile": "linear"},
    }


def _with_backend(cfg: dict, backend: str, **solver_kw) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("solver", {})
    out["solver"]["backend"] = backend
    out["solver"].update(solver_kw)
    return out


def _compare_metrics(np_res, jax_res) -> None:
    np_d = np_res.diagnostics
    jax_d = jax_res.diagnostics
    assert abs(int(jax_d["n_transitions"]) - int(np_d["n_transitions"])) <= 1

    np_md = float(np_d["mean_dwell_period"])
    jax_md = float(jax_d["mean_dwell_period"])
    if np.isfinite(np_md) and np_md > 0:
        assert abs(jax_md - np_md) / np_md <= 0.01

    np_th = float(np_d["thickness_final"])
    jax_th = float(jax_d["thickness_final"])
    assert abs(jax_th - np_th) / max(abs(np_th), 1e-30) <= 0.005

    c_np = np.asarray(np_res.state_final.c, dtype=np.float64).ravel()
    c_jx = np.asarray(jax_res.state_final.c, dtype=np.float64).ravel()
    assert float(np.max(np.abs(c_jx - c_np))) <= 1e-8


def test_jax_matches_numpy_baseline() -> None:
    cfg = _cfg(T=2.0)
    np_res = simulate(_with_backend(cfg, "numpy"), show_progress=False)
    jax_res = simulate(_with_backend(cfg, "jax"), show_progress=False)
    _compare_metrics(np_res, jax_res)


def test_jax_matches_numpy_oscillating() -> None:
    cfg = _cfg(T=2.0, c1=0.40, c2=0.26)
    np_res = simulate(_with_backend(cfg, "numpy"), show_progress=False)
    jax_res = simulate(_with_backend(cfg, "jax"), show_progress=False)
    _compare_metrics(np_res, jax_res)


def test_vmap_batch_equals_individual() -> None:
    base = _cfg(T=2.0, n=128)
    pairs = [(0.49, 0.10), (0.40, 0.26), (0.33, 0.29)]
    batch = BatchParams(
        Da=np.full(len(pairs), 1.0),
        kappa=np.full(len(pairs), 3.0),
        c1=np.asarray([p[0] for p in pairs]),
        c2=np.asarray([p[1] for p in pairs]),
    )
    base.setdefault("solver", {})
    base["solver"]["backend"] = "jax"
    batch_results, _timing = simulate_batch(base, batch)
    singles = [
        simulate(
            _with_backend(
                {**base, "physics": {**base["physics"], "c1": c1, "c2": c2}},
                "jax",
            ),
            show_progress=False,
        )
        for c1, c2 in pairs
    ]
    for bat, single in zip(batch_results, singles, strict=True):
        c_b = np.asarray(bat.state_final.c, dtype=np.float64).ravel()
        c_s = np.asarray(single.state_final.c, dtype=np.float64).ravel()
        assert float(np.max(np.abs(c_b - c_s))) <= 1e-12
        assert int(bat.diagnostics["n_transitions"]) == int(single.diagnostics["n_transitions"])


def test_transition_buffer_overflow_flag() -> None:
    cfg = _with_backend(_cfg(T=2.0, c1=0.40, c2=0.26), "jax", max_transitions=4)
    res = simulate(cfg, show_progress=False)
    assert bool(res.diagnostics.get("jax_transition_overflow")) is True
    assert int(res.diagnostics["n_transitions"]) <= 4


def test_history_downsampling_consistency() -> None:
    # T must be long enough for >=2 transitions; n kept small for runtime.
    cfg = _cfg(T=2.0, n=32, c1=0.40, c2=0.26)
    res_full = simulate(_with_backend(cfg, "jax", history_every=1), show_progress=False)
    res_coarse = simulate(_with_backend(cfg, "jax", history_every=50), show_progress=False)
    n_full = int(res_full.diagnostics["n_transitions"])
    n_coarse = int(res_coarse.diagnostics["n_transitions"])
    assert n_full == n_coarse
    assert n_full >= 2, "test setup must produce oscillations (n_transitions >= 2)"
    md_full = float(res_full.diagnostics["mean_dwell_period"])
    md_coarse = float(res_coarse.diagnostics["mean_dwell_period"])
    assert md_full == pytest.approx(md_coarse, rel=0.01)
