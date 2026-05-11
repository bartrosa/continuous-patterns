"""Axial diagnostics for slab geometry runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def compute_slab_axial_diagnostics(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    prominence_frac: float = 0.05,
) -> dict[str, Any]:
    """Compute axial band metrics from ``<phi_m - phi_c>_y(x)`` profile.

    Parameters
    ----------
    phi_m, phi_c
        Final phase fields on an ``(n, n)`` grid.
    L
        Domain length in x/y.
    prominence_frac
        Relative prominence threshold for peak detection.
    """
    psi = np.asarray(phi_m, dtype=np.float64) - np.asarray(phi_c, dtype=np.float64)
    n = int(psi.shape[0])
    dx = float(L) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    profile = np.mean(psi, axis=1)

    amp = float(np.max(profile) - np.min(profile))
    if amp <= 1e-12:
        return {
            "band_count_axial": 0,
            "peak_positions_x": [],
            "band_spacings_x": [],
            "q_cv": 0.0,
        }

    prom = float(prominence_frac) * (amp + 1e-12)
    peaks, _props = signal.find_peaks(profile, prominence=prom)
    peak_positions = [float(x[idx]) for idx in peaks]
    spacings = [float(b - a) for a, b in zip(peak_positions[:-1], peak_positions[1:], strict=False)]

    if len(spacings) >= 2:
        logs = np.log(np.asarray(spacings, dtype=np.float64))
        q_cv = float(np.std(logs) / (np.mean(np.abs(logs)) + 1e-30))
    else:
        q_cv = 0.0

    return {
        "band_count_axial": int(len(peaks)),
        "peak_positions_x": peak_positions,
        "band_spacings_x": spacings,
        "q_cv": q_cv,
    }
