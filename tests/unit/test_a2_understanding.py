"""Unit tests for A2 Ostwald-window analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.models.twophase_kernel_reference import psi_ostwald_partition


def test_psi_ostwald_partition_symmetric() -> None:
    psi_m, psi_c = psi_ostwald_partition(np.array([0.25]), c_ostwald=0.25, w_ostwald=0.05)
    assert psi_m[0] == pytest.approx(0.5)
    assert psi_c[0] == pytest.approx(0.5)


def test_psi_ostwald_minority_at_wings() -> None:
    c = np.array([0.10, 0.40])
    psi_m, psi_c = psi_ostwald_partition(c, c_ostwald=0.25, w_ostwald=0.05)
    f = np.minimum(psi_m, psi_c)
    assert float(f[0]) < 0.5
    assert float(f[1]) < 0.5
    assert float(f[0]) > 0.0
