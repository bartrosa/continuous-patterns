"""Geometry builder accepts ``uniform_biaxial`` stress mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from continuous_patterns.agate_ch.model import build_geometry


def test_build_geometry_uniform_biaxial() -> None:
    g = build_geometry(
        200.0,
        80.0,
        32,
        stress_mode="uniform_biaxial",
        sigma_0=1.25,
    )
    sxx = np.asarray(g.sigma_xx)
    syy = np.asarray(g.sigma_yy)
    sxy = np.asarray(g.sigma_xy)
    assert np.allclose(sxx, 1.25)
    assert np.allclose(syy, 1.25)
    assert np.allclose(sxy, 0.0)
