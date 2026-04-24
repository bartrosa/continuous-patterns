"""Unit tests for Stage I/II NumPy diagnostics (``core.diagnostics_*``)."""

from __future__ import annotations

import numpy as np

from continuous_patterns.core.diagnostics_stage1 import (
    cell_xy,
    fft_psi_anisotropy_ratio,
    jab_metrics_canonical_slice,
    option_b_leak_pct_from_meta,
    option_b_residual_pct,
)
from continuous_patterns.core.diagnostics_stage2 import (
    bulk_scalar_stats,
    structure_factor_radial_average,
)


def test_option_b_residual_trivial_equilibrium() -> None:
    assert option_b_residual_pct(0.0, 0.0) == 0.0
    assert option_b_residual_pct(1.0, 1.0) == 0.0
    leak = option_b_leak_pct_from_meta(
        {"dissolved_mass_delta": 2.0, "flux_time_integral": 2.0},
        {},
    )
    assert leak == 0.0


def test_fft_psi_anisotropy_near_one_for_isotropic_blob() -> None:
    L, n = 4.0, 64
    x, y, _dx = cell_xy(L=L, n=n)
    phi_m = np.exp(-((x - 2.0) ** 2 + (y - 2.0) ** 2) / 0.25).astype(np.float64)
    phi_c = np.zeros_like(phi_m)
    out = fft_psi_anisotropy_ratio(phi_m, phi_c, L=L, cavity_R=1.9)
    r = out["psi_fft_anisotropy_ratio"]
    assert abs(r - 1.0) < 0.2


def test_jab_canonical_slice_detects_synthetic_bands() -> None:
    L, n = 128, 128
    x, _y, dx = cell_xy(L=L, n=n)
    freq = 2.0 * np.pi * 14.0 / L
    bands = 0.5 + 0.5 * np.sin(freq * x)
    phi_c = np.broadcast_to(bands, (n, n)).astype(np.float64)
    phi_m = 0.1 * np.ones((n, n), dtype=np.float64)
    m = jab_metrics_canonical_slice(phi_m, phi_c, L=L, R=0.45 * L)
    assert m["n_bands"] >= 6


def test_structure_factor_radial_positive() -> None:
    L, n = 32.0, 32
    rng = np.random.default_rng(0)
    f = rng.standard_normal((n, n))
    out = structure_factor_radial_average(f, L=L)
    assert np.all(np.isfinite(out["S_radial_mean"]))
    assert np.all(out["S_radial_mean"] >= 0.0)


def test_bulk_scalar_stats_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((8, 8))
    b = rng.random((8, 8))
    s = bulk_scalar_stats(a, b)
    assert s["mean_phi_m"] == float(np.mean(a))
