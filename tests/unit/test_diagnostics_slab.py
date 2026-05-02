"""Unit tests for slab axial diagnostics."""

from __future__ import annotations

import numpy as np

from continuous_patterns.core.diagnostics_slab import compute_slab_axial_diagnostics


def test_slab_axial_detects_known_peak_count() -> None:
    n = 256
    L = 100.0
    x = (np.arange(n, dtype=np.float64) + 0.5) * (L / n)
    centers = [10.0, 25.0, 40.0, 60.0, 80.0]
    width = 1.4
    profile = np.zeros_like(x)
    for c in centers:
        profile += np.exp(-0.5 * ((x - c) / width) ** 2)
    phi_m = np.broadcast_to(profile[:, None], (n, n))
    phi_c = np.zeros((n, n), dtype=np.float64)

    out = compute_slab_axial_diagnostics(phi_m, phi_c, L=L)
    assert out["band_count_axial"] == 5
    assert len(out["peak_positions_x"]) == 5


def test_slab_axial_flat_field_has_zero_bands() -> None:
    n = 128
    L = 50.0
    phi_m = np.zeros((n, n), dtype=np.float64)
    phi_c = np.zeros((n, n), dtype=np.float64)

    out = compute_slab_axial_diagnostics(phi_m, phi_c, L=L)
    assert out["band_count_axial"] == 0
    assert out["peak_positions_x"] == []
