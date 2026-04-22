"""Radial averages, peaks, mass flux, time-resolved band tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import find_peaks


def radial_profile(
    field: np.ndarray,
    *,
    L: float,
    R: float,
    nbins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
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


def boundary_flux_mass_rate(
    c: np.ndarray, L: float, R: float, D_c: float, c_0: float
) -> float:
    """Approx inward silica flux through r=R (mass per unit time, 2D units)."""
    n = c.shape[0]
    dx = L / n
    xc = yc = L / 2
    x = (np.arange(n) + 0.5) * dx
    xv, yv = np.meshgrid(x, x, indexing="ij")
    rv = np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2)
    inner = (rv >= R - 2.8 * dx) & (rv < R - 0.9 * dx) & (rv < R)
    if not np.any(inner):
        inner = (rv < R) & (rv > R - 4.0 * dx)
    if not np.any(inner):
        return 0.0
    grad_est = np.mean(np.maximum(0.0, c_0 - c[inner])) / dx
    return float(D_c * grad_est * (2.0 * np.pi * R))


def overshoot_fraction(phi_m: np.ndarray, phi_c: np.ndarray) -> float:
    """% pixel values with φ ∉ [0,1] (either species)."""
    both = np.concatenate([phi_m.ravel(), phi_c.ravel()])
    bad = np.sum((both < 0.0) | (both > 1.0))
    return float(100.0 * bad / max(both.size, 1))


def overshoot_slack_fraction(phi_m: np.ndarray, phi_c: np.ndarray) -> float:
    """% outside [-0.05, 1.05] — acceptance test."""
    both = np.concatenate([phi_m.ravel(), phi_c.ravel()])
    bad = np.sum((both < -0.05) | (both > 1.05))
    return float(100.0 * bad / max(both.size, 1))


def classify_from_q(nb: int, mean_q: float, cv_q: float) -> str:
    if nb < 3:
        return "INSUFFICIENT BANDS"
    if np.isnan(mean_q) or np.isnan(cv_q):
        return "INSUFFICIENT BANDS"
    if mean_q > 1.05 and cv_q < 0.1:
        return "LIESEGANG-LIKE"
    return "NON-LIESEGANG"


def band_metrics(
    r_centers: np.ndarray,
    phi_tot_profile: np.ndarray,
    cavity_R: float,
    *,
    prominence_frac: float = 0.05,
    distance: int = 1,
) -> dict[str, Any]:
    prom = prominence_frac * float(np.max(phi_tot_profile))
    peaks, _ = find_peaks(
        phi_tot_profile,
        prominence=max(prom, 1e-9),
        distance=distance,
    )
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
            "classification": "INSUFFICIENT BANDS",
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

    klass = classify_from_q(nb, mean_q, cv_q)

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


def analyse_all_snapshots(
    h5_path: Path,
    *,
    L: float,
    R: float,
    dt: float,
    skip_before: int = 500,
) -> dict[str, Any]:
    """Peak tracking over HDF5 snapshots for kymograph / peak-time Jabłczyński."""
    records: list[tuple[int, int, list[float]]] = []
    kymo_t: list[float] = []
    kymo_r: list[float] = []

    peak_prof_metrics: dict[str, Any] | None = None
    final_prof_metrics: dict[str, Any] | None = None
    peak_rc_arr: np.ndarray | None = None
    peak_pt_arr: np.ndarray | None = None
    max_peaks = -1
    max_step = 0

    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            step = int(k.split("_")[1])
            if step < skip_before:
                continue
            pm = np.asarray(h5[k]["phi_m"])
            pc = np.asarray(h5[k]["phi_c"])
            tot = pm + pc
            rc, prof = radial_profile(tot, L=L, R=R, nbins=200)
            prange = float(np.max(prof) - np.min(prof))
            prom = max(0.10 * prange, 1e-9)
            peaks, _ = find_peaks(prof, prominence=prom, distance=3)
            nb = int(len(peaks))
            rpk = [float(rc[i]) for i in peaks]
            records.append((step, nb, rpk))
            t_phys = step * dt
            for rp in rpk:
                kymo_t.append(t_phys)
                kymo_r.append(rp)
            if nb > max_peaks:
                max_peaks = nb
                max_step = step

        if keys:
            last = keys[-1]
            pm = np.asarray(h5[last]["phi_m"])
            pc = np.asarray(h5[last]["phi_c"])
            rc, pt = radial_profile(pm + pc, L=L, R=R)
            final_prof_metrics = band_metrics(rc, pt, R)

        if max_peaks >= 1 and max_step >= skip_before:
            gname = f"t_{max_step:07d}"
            if gname in h5:
                pm = np.asarray(h5[gname]["phi_m"])
                pc = np.asarray(h5[gname]["phi_c"])
                rc, pt = radial_profile(pm + pc, L=L, R=R)
                peak_prof_metrics = band_metrics(
                    rc,
                    pt,
                    R,
                    prominence_frac=0.10,
                    distance=3,
                )
                peak_rc_arr = rc
                peak_pt_arr = pt

    if peak_prof_metrics is None:
        peak_prof_metrics = {"N_b": 0, "classification": "INSUFFICIENT BANDS"}

    return {
        "records": records,
        "kymograph_t": kymo_t,
        "kymograph_r": kymo_r,
        "peak_band_count": max(0, max_peaks),
        "peak_band_count_step": max_step,
        "peak_band_count_time": float(max_step * dt),
        "metrics_at_peak": peak_prof_metrics,
        "metrics_at_final": final_prof_metrics or {},
        "peak_radial_centers": peak_rc_arr,
        "peak_radial_profile": peak_pt_arr,
    }
