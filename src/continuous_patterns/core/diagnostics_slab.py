"""Axial diagnostics for slab geometry runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


def detect_profile_peaks(
    profile: np.ndarray,
    x: np.ndarray,
    *,
    prominence_frac: float = 0.05,
) -> dict[str, Any]:
    """Detect peaks in a 1D profile using the standard Liesegang prominence rule.

    Parameters
    ----------
    profile
        1D field sampled at ``x`` (e.g. y-averaged ``phi_m - phi_c``).
    x
        Physical coordinates for each profile sample.
    prominence_frac
        Relative prominence threshold: ``prominence = prominence_frac * (max - min)``.

    Returns
    -------
    dict
        ``peak_indices``, ``peak_positions``, ``prominence_threshold``, ``profile_amp``.
        When amplitude is negligible, ``peak_indices`` and ``peak_positions`` are empty.
    """
    profile = np.asarray(profile, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if profile.size != x.size:
        raise ValueError("profile and x must have the same length")

    amp = float(np.max(profile) - np.min(profile))
    if amp <= 1e-12:
        return {
            "peak_indices": np.asarray([], dtype=np.int64),
            "peak_positions": np.asarray([], dtype=np.float64),
            "prominence_threshold": 0.0,
            "profile_amp": amp,
        }

    prom = float(prominence_frac) * (amp + 1e-12)
    peaks, _props = signal.find_peaks(profile, prominence=prom)
    peak_positions = x[peaks].astype(np.float64, copy=False)
    return {
        "peak_indices": peaks.astype(np.int64, copy=False),
        "peak_positions": peak_positions,
        "prominence_threshold": prom,
        "profile_amp": amp,
    }


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

    detected = detect_profile_peaks(profile, x, prominence_frac=prominence_frac)
    peak_positions = [float(v) for v in detected["peak_positions"]]
    spacings = [float(b - a) for a, b in zip(peak_positions[:-1], peak_positions[1:], strict=False)]

    if len(spacings) >= 2:
        logs = np.log(np.asarray(spacings, dtype=np.float64))
        q_cv = float(np.std(logs) / (np.mean(np.abs(logs)) + 1e-30))
    else:
        q_cv = 0.0

    return {
        "band_count_axial": int(len(peak_positions)),
        "peak_positions_x": peak_positions,
        "band_spacings_x": spacings,
        "q_cv": q_cv,
    }
