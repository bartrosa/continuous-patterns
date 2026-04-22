"""Radial averages, peak finding, Jabłczyński classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def radial_profile(
    field: np.ndarray,
    *,
    L: float,
    R: float,
    nbins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged ⟨field⟩(r) for r ∈ [0, R]."""
    n = field.shape[0]
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(n) + 0.5) * dx
    xv, yv = np.meshgrid(x, y, indexing="ij")
    xc = yc = L / 2
    rv = np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2).ravel()
    fv = np.asarray(field).ravel()
    mask = rv <= R
    rv = rv[mask]
    fv = fv[mask]
    bins = np.linspace(0.0, R, nbins + 1)
    sums = np.zeros(nbins, dtype=np.float64)
    cnt = np.zeros(nbins, dtype=np.float64)
    ii = np.digitize(rv, bins) - 1
    ii = np.clip(ii, 0, nbins - 1)
    np.add.at(sums, ii, fv)
    np.add.at(cnt, ii, 1)
    cnt = np.maximum(cnt, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, sums / cnt


def band_metrics(
    r_centers: np.ndarray,
    phi_tot_profile: np.ndarray,
    cavity_R: float,
    *,
    prominence_frac: float = 0.05,
) -> dict[str, Any]:
    prom = prominence_frac * float(np.max(phi_tot_profile))
    peaks, _ = find_peaks(phi_tot_profile, prominence=max(prom, 1e-9))
    nb = len(peaks)
    if nb == 0:
        return {
            "N_b": 0,
            "r_outer_in": [],
            "d": [],
            "q": [],
            "mean_q": float("nan"),
            "std_q": float("nan"),
            "cv_q": float("nan"),
            "classification": "NO BANDS",
        }

    r_peaks = r_centers[peaks]
    outer_in = sorted(r_peaks.tolist(), reverse=True)
    r_arr = np.array(outer_in, dtype=np.float64)
    dd_list: list[float] = []
    if r_arr.size:
        dd_list.append(float(cavity_R - r_arr[0]))
    for i in range(len(r_arr) - 1):
        dd_list.append(float(r_arr[i] - r_arr[i + 1]))
    dd = np.array(dd_list, dtype=np.float64)
    qq = dd[1:] / dd[:-1] if dd.size >= 2 else np.array([], dtype=np.float64)

    mean_q = float(np.nanmean(qq)) if qq.size else float("nan")
    std_q = float(np.nanstd(qq)) if qq.size else float("nan")
    cv_q = float(std_q / mean_q) if qq.size and mean_q > 1e-12 else float("nan")

    if nb < 3:
        klass = "NO BANDS"
    elif qq.size == 0:
        klass = "NO BANDS"
    elif mean_q > 1.05 and not np.isnan(cv_q) and cv_q < 0.1:
        klass = "LIESEGANG-LIKE"
    else:
        klass = "NON-LIESEGANG"

    return {
        "N_b": nb,
        "r_outer_in": outer_in,
        "d": dd_list,
        "q": qq.tolist(),
        "mean_q": mean_q,
        "std_q": std_q,
        "cv_q": cv_q,
        "classification": klass,
    }
