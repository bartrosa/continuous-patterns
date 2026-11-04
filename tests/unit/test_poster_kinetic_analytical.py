"""Unit tests for poster kinetic analytical helpers."""

from __future__ import annotations

import numpy as np

from continuous_patterns.core.poster_kinetic_analytical import (
    analytical_period_sum,
    solve_T1_T2,
)


def test_solve_T1_T2_poster_pair() -> None:
    t1, t2 = solve_T1_T2(1.0, 3.0, 0.40, 0.26)
    assert t1 is not None and t2 is not None
    assert t1 > 0.0 and t2 > 0.0
    period = analytical_period_sum(1.0, 3.0, 0.40, 0.26)
    assert period is not None
    assert np.isclose(period, t1 + t2)


def test_no_root_when_targets_invalid() -> None:
    t1, t2 = solve_T1_T2(1.0, 3.0, 0.49, 0.10)
    # Poster stuck pair: transcendental targets often outside (0,1)
    assert t1 is None or t2 is None
