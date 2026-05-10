"""Unit tests for ``continuous_patterns.core.diagnostics_kinetic``."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.core import diagnostics_kinetic as dk


def test_compute_transitions_detects_simple_oscillation() -> None:
    t = np.linspace(0.0, 10.0, 10001)
    # Piecewise-constant state-like forcing is not how hysteresis replays; instead build c(t)
    # that crosses thresholds at known times under replay rules.
    c = np.zeros_like(t)
    # Stay below c2 early, then spike above c1, etc.: simplified staircase
    c[:] = 0.05
    c[t > 1.0] = 0.6
    c[t > 2.0] = 0.05
    c[t > 3.0] = 0.6
    trans = dk.compute_transitions(c, t, c1=0.49, c2=0.10)
    kinds = [k for _, k in trans]
    assert "L→H" in kinds
    assert "H→L" in kinds


def test_compute_thickness_monotonic_nonnegative() -> None:
    rng = np.random.default_rng(0)
    c0 = rng.uniform(0.0, 1.0, size=500)
    st = rng.integers(0, 2, size=500).astype(np.int32)
    dt = 1e-3
    th = dk.compute_thickness(c0, st, dt, kappa=3.0)
    assert th.shape == c0.shape
    assert np.all(th >= 0.0)
    assert np.all(np.diff(th) >= -1e-15)


def test_compute_dwell_times_consistency() -> None:
    transitions = [(1.0, "L→H"), (3.0, "H→L"), (4.5, "L→H")]
    T_final = 10.0
    dwell_L, dwell_H = dk.compute_dwell_times(transitions, T_final)
    s = float(np.sum(dwell_L) + np.sum(dwell_H))
    assert abs(s - T_final) < 1e-9


def test_compute_temporal_spectrum_finds_known_period() -> None:
    dt = 0.01
    T0 = 2.5
    t = np.arange(0.0, 80.0, dt)
    y = np.sin(2.0 * np.pi * t / T0)
    spec = dk.compute_temporal_spectrum(y, dt)
    pp = float(spec["peak_period"])
    assert pp == pp
    assert abs(pp - T0) / T0 < 0.05


def test_compute_transitions_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        dk.compute_transitions(np.array([1.0]), np.array([0.0, 1.0]), c1=0.5, c2=0.1)
