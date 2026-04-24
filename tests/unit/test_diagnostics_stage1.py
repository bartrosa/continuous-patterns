"""Unit tests for v1-style sampling helpers in :mod:`diagnostics_stage1`."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.core.diagnostics_stage1 import (
    azimuthal_mean_at_radius_numpy,
    bilinear_sample_field,
    dissolved_mass_disk_numpy,
)


def test_bilinear_sample_field_constant() -> None:
    """Constant field → constant samples."""
    field = np.full((64, 64), 2.5, dtype=np.float64)
    x_s = np.array([3.2, 5.7, 10.1])
    y_s = np.array([1.1, 8.3, 4.5])
    result = bilinear_sample_field(field, L=10.0, x_s=x_s, y_s=y_s)
    np.testing.assert_allclose(result, 2.5, atol=1e-7)


def test_bilinear_sample_linear_x() -> None:
    """``f(x,y) = x`` → bilinear sample returns ``x_s`` (cell-centred grid)."""
    L, n = 10.0, 64
    dx = L / n
    ii = np.arange(n, dtype=np.float64)[:, None]
    xv = (ii + 0.5) * dx
    field = np.broadcast_to(xv, (n, n)).copy()
    x_s = np.array([2.5, 5.0, 7.3])
    y_s = np.array([1.0, 4.0, 8.0])
    result = bilinear_sample_field(field, L=L, x_s=x_s, y_s=y_s)
    np.testing.assert_allclose(result, x_s, rtol=1e-2)


def test_azimuthal_mean_constant() -> None:
    """Constant field → azimuthal mean equals constant."""
    field = np.full((128, 128), 7.1, dtype=np.float64)
    result = azimuthal_mean_at_radius_numpy(field, L=10.0, r_abs=3.0)
    assert abs(result - 7.1) < 1e-6


def test_dissolved_mass_disk_constant() -> None:
    """Uniform ``c`` inside disk radius matches area × value."""
    L, n = 10.0, 256
    dx = L / n
    c = np.ones((n, n), dtype=np.float64) * 0.4
    r_disk = 2.0
    m = dissolved_mass_disk_numpy(c, L=L, r_disk=r_disk)
    xc = 0.5 * L
    ii = np.arange(n, dtype=np.float64)[:, None]
    jj = np.arange(n, dtype=np.float64)[None, :]
    xv = (ii + 0.5) * dx
    yv = (jj + 0.5) * dx
    rv = np.sqrt((xv - xc) ** 2 + (yv - xc) ** 2)
    area = float(np.sum(rv < r_disk) * dx * dx)
    assert m == pytest.approx(0.4 * area, rel=0, abs=1e-6)
