"""Classical Jabłczyński position ratio ``Q_n = x_{n+1}/x_n`` in diagnostics."""

from __future__ import annotations

import numpy as np

from continuous_patterns.core.diagnostics_stage1 import cell_xy, jab_metrics_canonical_slice


def test_Q_ratio_classical_formulation() -> None:
    """``Q_positions`` matches ``x_{n+1}/x_n`` from reported peak positions."""
    L, n = 128, 128
    x, _y, _dx = cell_xy(L=L, n=n)
    freq = 2.0 * np.pi * 12.0 / L
    bands = 0.5 + 0.5 * np.sin(freq * x)
    phi_c = np.broadcast_to(bands, (n, n)).astype(np.float64)
    phi_m = 0.1 * np.ones((n, n), dtype=np.float64)

    jab = jab_metrics_canonical_slice(phi_m, phi_c, L=L, R=0.45 * L)

    assert "Q_positions" in jab
    assert "Q_cv" in jab
    peaks = np.asarray(jab["peak_positions"], dtype=np.float64)
    assert jab["n_bands"] == len(peaks)
    if peaks.size > 1:
        assert len(jab["Q_positions"]) == jab["n_bands"] - 1
        np.testing.assert_allclose(
            np.asarray(jab["Q_positions"], dtype=np.float64),
            peaks[1:] / peaks[:-1],
        )
