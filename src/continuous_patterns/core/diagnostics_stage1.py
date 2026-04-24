"""Post-run diagnostics for Stage I (cavity, rim, Option B, Jabłczyński, …).

NumPy-only analysis: mass flux budgets, canonical-slice metrics, FFT ψ
anisotropy, stability scalars. Not used for Stage II bulk runs
(``docs/ARCHITECTURE.md`` §2.3, §3.5).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage, signal, stats

# ---------------------------------------------------------------------------
# Option B / surface flux — v1-style circle sampling (PHYSICS §10.3)
# ---------------------------------------------------------------------------


def bilinear_sample_field(
    field: np.ndarray,
    L: float,
    x_s: np.ndarray,
    y_s: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation at arbitrary ``(x, y)`` with periodic wrap.

    ``field`` is ``(n, n)`` on ``[0, L)²`` with cell centres at ``((i+0.5) dx, (j+0.5) dx)``,
    indices ``(i, j)`` matching :func:`cell_xy` (row ``i`` ↔ ``x``, column ``j`` ↔ ``y``).
    """
    n = int(field.shape[0])
    if field.shape[1] != n:
        raise ValueError("bilinear_sample_field expects a square (n, n) field")
    dx = L / n
    x_s = np.asarray(x_s, dtype=np.float64)
    y_s = np.asarray(y_s, dtype=np.float64)
    i_f = x_s / dx - 0.5
    j_f = y_s / dx - 0.5
    i0 = np.floor(i_f).astype(np.int64) % n
    j0 = np.floor(j_f).astype(np.int64) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    fi = i_f - np.floor(i_f)
    fj = j_f - np.floor(j_f)
    fi = np.clip(fi, 0.0, 1.0)
    fj = np.clip(fj, 0.0, 1.0)
    return (
        (1.0 - fi) * (1.0 - fj) * field[i0, j0]
        + (1.0 - fi) * fj * field[i0, j1]
        + fi * (1.0 - fj) * field[i1, j0]
        + fi * fj * field[i1, j1]
    )


def azimuthal_mean_at_radius_numpy(
    c_field: np.ndarray,
    *,
    L: float,
    r_abs: float,
    n_theta: int = 360,
) -> float:
    """Azimuthal mean on a circle of radius ``r_abs`` about ``(L/2, L/2)`` via bilinear sampling."""
    xc = 0.5 * float(L)
    yc = 0.5 * float(L)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    x_s = xc + float(r_abs) * np.cos(theta)
    y_s = yc + float(r_abs) * np.sin(theta)
    vals = bilinear_sample_field(np.asarray(c_field, dtype=np.float64), L, x_s, y_s)
    return float(np.mean(vals))


def dissolved_mass_disk_numpy(
    c: np.ndarray,
    *,
    L: float,
    r_disk: float,
) -> float:
    """Integral of dissolved ``c`` over ``r < r_disk`` (hard disk, **no** ``χ`` weight)."""
    n = int(c.shape[0])
    dx = L / n
    xc = 0.5 * float(L)
    yc = 0.5 * float(L)
    ii = np.arange(n, dtype=np.float64)[:, None]
    jj = np.arange(n, dtype=np.float64)[None, :]
    xv = (ii + 0.5) * dx
    yv = (jj + 0.5) * dx
    rv = np.sqrt((xv - xc) ** 2 + (yv - yc) ** 2)
    mask = rv < float(r_disk)
    return float(np.sum(np.asarray(c) * mask) * dx * dx)


def option_b_residual_pct(
    dissolved_mass_delta: float,
    flux_time_integral: float,
    *,
    eps: float = 1e-30,
) -> float:
    """Relative Option B residual in percent: ``100 |ΔM - F| / denom``."""
    dm = float(dissolved_mass_delta)
    fl = float(flux_time_integral)
    denom = max(abs(fl), abs(dm), eps)
    return 100.0 * abs(dm - fl) / denom


def option_b_leak_pct_from_meta(meta: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Option B leak % from integration metadata (PHYSICS §10.3).

    Reads ``dissolved_mass_delta`` and ``flux_time_integral`` from ``meta``.
    ``cfg`` is reserved for future overrides (e.g. ``r_fix``, ``D_c``).
    """
    _ = cfg
    return option_b_residual_pct(
        float(meta.get("dissolved_mass_delta", 0.0)),
        float(meta.get("flux_time_integral", 0.0)),
    )


# ---------------------------------------------------------------------------
# χ-weighted silica (PHYSICS §10.4)
# ---------------------------------------------------------------------------


def chi_weighted_silica_integral(
    c: np.ndarray,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    chi: np.ndarray,
    *,
    rho_m: float,
    rho_c: float,
    dx: float,
) -> float:
    """Integral of ``(c + ρ_m φ_m + ρ_c φ_c) χ`` over the cell-centred grid."""
    s = c + float(rho_m) * phi_m + float(rho_c) * phi_c
    return float(np.sum(s * chi) * dx * dx)


# ---------------------------------------------------------------------------
# Grid helpers (row = x, col = y; matches ``masks`` / ``stress``)
# ---------------------------------------------------------------------------


def cell_xy(*, L: float, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell-centred ``x, y`` on ``[0, L)²`` and ``Δx``."""
    dx = L / n
    ii = np.arange(n, dtype=np.float64)[:, None]
    jj = np.arange(n, dtype=np.float64)[None, :]
    x = (ii + 0.5) * dx
    y = (jj + 0.5) * dx
    x = np.broadcast_to(x, (n, n))
    y = np.broadcast_to(y, (n, n))
    return x, y, dx


def hard_disk_mask(
    *,
    L: float,
    n: int,
    cavity_R: float,
    xc: float | None = None,
    yc: float | None = None,
) -> np.ndarray:
    """Boolean mask ``r < cavity_R`` with ``r`` from domain centre."""
    x, y, _dx = cell_xy(L=L, n=n)
    xc = 0.5 * L if xc is None else float(xc)
    yc = 0.5 * L if yc is None else float(yc)
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    return r < float(cavity_R)


# ---------------------------------------------------------------------------
# FFT ψ-anisotropy (PHYSICS §10.4)
# ---------------------------------------------------------------------------


def fft_psi_anisotropy_ratio(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    cavity_R: float,
    k_low_frac: float = 0.15,
) -> dict[str, float]:
    """Low-|k_x| vs low-|k_y| band power of ``|FFT(ψ χ_disk)|²`` (PHYSICS §10.4)."""
    n = int(phi_m.shape[0])
    x, y, dx = cell_xy(L=L, n=n)
    xc = 0.5 * L
    yc = 0.5 * L
    chi = (np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < float(cavity_R)).astype(np.float64)
    psi = (phi_m - phi_c) * chi
    u_hat = np.fft.fft2(psi)
    P = np.abs(u_hat) ** 2
    kx = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    k_lim = float(k_low_frac) * (2.0 * math.pi / L)
    band_x = np.abs(KX) <= k_lim
    band_y = np.abs(KY) <= k_lim
    p_x = float(np.sum(P * band_x))
    p_y = float(np.sum(P * band_y))
    ratio = p_x / max(p_y, 1e-30)
    return {"psi_fft_anisotropy_ratio": ratio, "P_low_kx": p_x, "P_low_ky": p_y}


# ---------------------------------------------------------------------------
# Pixel noise (PHYSICS §10.5)
# ---------------------------------------------------------------------------


def pixel_noise_rms(
    phi_m: np.ndarray,
    chi_disk: np.ndarray,
    *,
    periodic: bool = True,
) -> float:
    """RMS of ``φ_m - uniform_filter_{3×3}(φ_m)`` on the disk (PHYSICS §10.5)."""
    mode = "wrap" if periodic else "nearest"
    sm = ndimage.uniform_filter(phi_m.astype(np.float64), size=3, mode=mode)
    resid = (phi_m.astype(np.float64) - sm) * (chi_disk > 0.5).astype(np.float64)
    w = float(np.sum(chi_disk > 0.5))
    if w <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(resid**2) / w))


# ---------------------------------------------------------------------------
# Jabłczyński canonical slice (PHYSICS §10.6)
# ---------------------------------------------------------------------------


def jab_metrics_canonical_slice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    cavity_R: float | None = None,
) -> dict[str, Any]:
    """Horizontal centreline ``y = L/2``, right half ``x ≥ L/2``, ``r < R`` (PHYSICS §10.6)."""
    n = int(phi_m.shape[0])
    x, y, _dx = cell_xy(L=L, n=n)
    R_eff = float(R if cavity_R is None else cavity_R)
    xc = 0.5 * L
    yc = 0.5 * L
    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    j_mid = int(np.argmin(np.abs(y[0, :] - 0.5 * L)))
    mask = (r[:, j_mid] < R_eff) & (x[:, j_mid] >= 0.5 * L)
    line_c = phi_c[:, j_mid][mask]
    line_m = phi_m[:, j_mid][mask]
    xs = x[:, j_mid][mask]
    use_c = line_c.size >= 8
    field = line_c if use_c else line_m
    if field.size < 5:
        return {
            "n_bands": 0,
            "peak_positions": [],
            "spacings": [],
            "q_ratios": [],
            "q_cv": 0.0,
            "spearman_d_vs_index": float("nan"),
            "used_field": "phi_c" if use_c else "phi_m",
        }
    prom = 0.05 * (np.max(field) - np.min(field) + 1e-9)
    peaks, _props = signal.find_peaks(field, prominence=prom)
    peak_x = [float(xs[p]) for p in peaks]
    spacings: list[float] = []
    for i in range(1, len(peak_x)):
        spacings.append(peak_x[i] - peak_x[i - 1])
    q_ratios: list[float] = []
    for i in range(1, len(spacings)):
        if spacings[i - 1] > 1e-12:
            q_ratios.append(spacings[i] / spacings[i - 1])
    q_cv = float(np.std(q_ratios) / (np.mean(q_ratios) + 1e-30)) if q_ratios else 0.0
    if len(spacings) >= 2:
        rho, _p = stats.spearmanr(np.arange(len(spacings), dtype=np.float64), np.asarray(spacings))
        spear = float(rho) if not np.isnan(rho) else float("nan")
    else:
        spear = float("nan")
    return {
        "n_bands": int(len(peaks)),
        "peak_positions": peak_x,
        "spacings": spacings,
        "q_ratios": q_ratios,
        "q_cv": q_cv,
        "spearman_d_vs_index": spear,
        "used_field": "phi_c" if use_c else "phi_m",
    }


# ---------------------------------------------------------------------------
# Multislice band counts (PHYSICS §10.7)
# ---------------------------------------------------------------------------


def _ray_sample_1d(
    field: np.ndarray,
    *,
    L: float,
    xc: float,
    yc: float,
    cavity_R: float,
    theta: float,
    dx: float,
) -> np.ndarray:
    """Samples ``field`` along a ray from ``(xc, yc)`` while ``r < cavity_R`` and ``x ≥ L/2``."""
    vals: list[float] = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    s = 0.0
    while s <= cavity_R * 1.02:
        x = xc + s * cos_t
        y = yc + s * sin_t
        if x < 0.5 * L or x >= L or y < 0.0 or y >= L:
            s += dx
            continue
        r = math.hypot(x - xc, y - yc)
        if r >= cavity_R:
            break
        xi = x / dx - 0.5
        yj = y / dx - 0.5
        v = float(
            ndimage.map_coordinates(
                field.astype(np.float64),
                np.array([[xi], [yj]]),
                order=1,
                mode="wrap",
            )[0]
        )
        vals.append(v)
        s += dx
    return np.asarray(vals, dtype=np.float64)


def count_bands_multislice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    cavity_R: float | None = None,
    n_angles: int = 8,
) -> dict[str, Any]:
    """Median peak count over angles on right-half rays (PHYSICS §10.7)."""
    R_eff = float(R if cavity_R is None else cavity_R)
    xc = 0.5 * L
    yc = 0.5 * L
    dx = L / float(phi_m.shape[0])
    # Rays with ``cos θ ≥ 0`` stay in ``x ≥ L/2`` when stepping from domain centre.
    angles = [-0.5 * math.pi + math.pi * (k + 0.5) / float(n_angles) for k in range(n_angles)]
    counts_m: list[int] = []
    counts_c: list[int] = []
    for th in angles:
        for ph, bucket in ((phi_m, counts_m), (phi_c, counts_c)):
            line = _ray_sample_1d(ph, L=L, xc=xc, yc=yc, cavity_R=R_eff, theta=th, dx=dx)
            if line.size < 5:
                bucket.append(0)
                continue
            prom = 0.05 * (float(np.max(line)) - float(np.min(line)) + 1e-9)
            peaks, _ = signal.find_peaks(line, prominence=prom)
            bucket.append(int(len(peaks)))
    all_c = counts_m + counts_c
    return {
        "median_peak_count_multislice": float(np.median(all_c)) if all_c else 0.0,
        "per_angle_counts_m": counts_m,
        "per_angle_counts_c": counts_c,
    }
