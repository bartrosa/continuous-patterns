"""Radial profiles, slices, multi-slice band counts, flux, Jabłczyński ratios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from jax import vmap
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from continuous_patterns.agate_stage2.model import build_geometry


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


def _sample_right_half_ray(phi: np.ndarray, L: float, R: float, theta: float) -> np.ndarray:
    """Values along ray from cavity centre along θ, s ∈ [0, R] (right half-diameter)."""
    n = phi.shape[0]
    dx = L / n
    xc = yc = L / 2
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ds = dx / 4.0
    vals: list[float] = []
    s = 0.0
    while s <= R + ds:
        x = xc + s * cos_t
        y = yc + s * sin_t
        if x < 0 or x > L or y < 0 or y > L:
            break
        r = float(np.hypot(x - xc, y - yc))
        if r > R + 1e-6:
            break
        ii = x / dx - 0.5
        jj = y / dx - 0.5
        if ii < -0.5 or ii > n - 0.51 or jj < -0.5 or jj > n - 0.51:
            break
        v = map_coordinates(
            phi,
            np.array([[ii], [jj]]),
            order=1,
            mode="nearest",
        )[0]
        vals.append(float(v))
        s += ds
    return np.array(vals, dtype=np.float64)


def count_bands_multislice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    L: float,
    R: float,
    *,
    height: float = 0.5,
    distance: int = 3,
) -> int:
    """Median peak count over 8 angles × (φ_m, φ_c) right-half rays."""
    counts: list[int] = []
    for k in range(8):
        theta = k * np.pi / 8.0
        seg_m = _sample_right_half_ray(phi_m, L, R, theta)
        seg_c = _sample_right_half_ray(phi_c, L, R, theta)
        pm, _ = find_peaks(seg_m, height=height, distance=distance)
        pc, _ = find_peaks(seg_c, height=height, distance=distance)
        counts.append(len(pm))
        counts.append(len(pc))
    return int(np.median(counts))


def horizontal_centerline(phi: np.ndarray, L: float, R: float) -> tuple[np.ndarray, np.ndarray]:
    """φ along y = L/2 for |x−xc| ≤ R (full diameter inside cavity)."""
    n = phi.shape[0]
    dx = L / n
    xc = L / 2
    j0 = int(np.clip(round(xc / dx - 0.5), 0, n - 1))
    xs: list[float] = []
    vs: list[float] = []
    for i in range(n):
        x = (i + 0.5) * dx
        if abs(x - xc) <= R + 1e-9:
            xs.append(x)
            vs.append(float(phi[i, j0]))
    return np.array(xs, dtype=np.float64), np.array(vs, dtype=np.float64)


def horizontal_right_half(phi: np.ndarray, L: float, R: float) -> tuple[np.ndarray, np.ndarray]:
    """Right half (x ≥ xc) of horizontal centreline for peak finding."""
    xs, vs = horizontal_centerline(phi, L, R)
    xc = L / 2
    m = xs >= xc - 1e-9
    return xs[m], vs[m]


def moganite_chalcedony_anticorr(phi_m: np.ndarray, phi_c: np.ndarray, L: float, R: float) -> float:
    _, vm = horizontal_centerline(phi_m, L, R)
    _, vc = horizontal_centerline(phi_c, L, R)
    if vm.size < 3 or vc.size < 3:
        return float("nan")
    n = min(vm.size, vc.size)
    vm = vm[:n]
    vc = vc[:n]
    if np.std(vm) < 1e-12 or np.std(vc) < 1e-12:
        return float("nan")
    return float(np.corrcoef(vm, vc)[0, 1])


def jab_metrics_canonical_slice(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    L: float,
    R: float,
    cavity_R: float,
    *,
    height: float = 0.5,
    distance: int = 3,
) -> dict[str, Any]:
    """Horizontal θ=0 right half; prefer φ_c peaks, else φ_m."""
    xr, vc = horizontal_right_half(phi_c, L, R)
    peaks_c, _ = find_peaks(vc, height=height, distance=distance)
    field_used = "phi_c"
    xr_u, v_u, peaks = xr, vc, peaks_c
    if len(peaks) < 3:
        xr, vm = horizontal_right_half(phi_m, L, R)
        peaks_m, _ = find_peaks(vm, height=height, distance=distance)
        xr_u, v_u, peaks = xr, vm, peaks_m
        field_used = "phi_m"

    nb = len(peaks)
    x_peaks = xr_u[peaks]
    outer_in = sorted(x_peaks.tolist(), reverse=True)
    x_arr = np.array(outer_in, dtype=np.float64)
    dd_list: list[float] = []
    xc = L / 2
    if x_arr.size:
        dd_list.append(float((xc + cavity_R) - x_arr[0]))
    for i in range(len(x_arr) - 1):
        dd_list.append(float(x_arr[i] - x_arr[i + 1]))
    dd = np.array(dd_list, dtype=np.float64)
    qq = dd[1:] / dd[:-1] if dd.size >= 2 else np.array([], dtype=np.float64)

    mean_q = float(np.nanmean(qq)) if qq.size else float("nan")
    std_q = float(np.nanstd(qq)) if qq.size else float("nan")
    cv_q = float(std_q / mean_q) if qq.size and abs(mean_q) > 1e-12 else float("nan")
    cv_d = float(np.std(dd) / np.mean(dd)) if dd.size > 1 else float("nan")

    rho_dn = float("nan")
    if dd.size >= 3:
        n_idx = np.arange(1, len(dd) + 1, dtype=np.float64)
        sr = spearmanr(dd, n_idx)
        rho_dn = float(sr.statistic if hasattr(sr, "statistic") else float(sr[0]))
        if np.isnan(rho_dn):
            rho_dn = float("nan")

    std_qq = float(np.std(qq)) if qq.size else 0.0
    klass = classify_jab_banding(nb, qq, dd, cv_q, rho_dn, std_qq)

    return {
        "N_b": nb,
        "r_outer_in": outer_in,
        "d": dd_list,
        "q": qq.tolist(),
        "mean_q": mean_q,
        "std_q": std_q,
        "cv_q": cv_q,
        "cv_d_spacings": cv_d,
        "spearman_d_vs_index": rho_dn,
        "classification": klass,
        "canonical_field": field_used,
        "x_centers": xr_u.tolist(),
        "profile_values": v_u.tolist(),
    }


def classify_jab_banding(
    nb: int,
    qq: np.ndarray,
    dd: np.ndarray,
    cv_q: float,
    rho_dn: float,
    std_qq: float,
) -> str:
    if nb < 3:
        return "INSUFFICIENT BANDS"
    if qq.size and cv_q < 0.15 and np.all(qq > 1.02):
        return "LIESEGANG-LIKE"
    if not np.isnan(rho_dn) and rho_dn > 0.7:
        return "LIESEGANG-LIKE"
    if std_qq > 0.5:
        return "RATCHET-BANDED"
    return "IRREGULAR-BANDED"


def bilinear_sample_field(
    field: np.ndarray,
    L: float,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Sample cell-centred field at world coordinates (xq, yq)."""
    n = field.shape[0]
    dx = L / n
    ii = xq / dx - 0.5
    jj = yq / dx - 0.5
    return map_coordinates(field, np.array([ii, jj]), order=1, mode="nearest")


def _bilinear_sample_jnp(
    field: jnp.ndarray,
    L: float,
    xq: jnp.ndarray,
    yq: jnp.ndarray,
) -> jnp.ndarray:
    """Single-point bilinear sample (cell-centred grid), JAX-friendly."""
    n = int(field.shape[0])
    dx = L / n
    ii = jnp.clip(xq / dx - 0.5, 0.0, jnp.asarray(n - 1, dtype=jnp.float32) - 1e-5)
    jj = jnp.clip(yq / dx - 0.5, 0.0, jnp.asarray(n - 1, dtype=jnp.float32) - 1e-5)
    i0 = jnp.floor(ii).astype(jnp.int32)
    j0 = jnp.floor(jj).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    j1 = jnp.minimum(j0 + 1, n - 1)
    tx = ii - i0.astype(jnp.float32)
    ty = jj - j0.astype(jnp.float32)
    c00 = field[i0, j0]
    c01 = field[i0, j1]
    c10 = field[i1, j0]
    c11 = field[i1, j1]
    return (1 - tx) * (1 - ty) * c00 + (1 - tx) * ty * c01 + tx * (1 - ty) * c10 + tx * ty * c11


def azimuthal_mean_c_ring_numpy(
    c_field: np.ndarray,
    *,
    L: float,
    R: float,
    dx: float,
    offset_r_in_dx: float,
    n_theta: int = 360,
) -> float:
    """Azimuthal mean of c at radius r = R - offset_r_in_dx * dx."""
    xc = L / 2.0
    r_t = R - offset_r_in_dx * dx
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    x_s = xc + r_t * np.cos(theta)
    y_s = xc + r_t * np.sin(theta)
    vals = bilinear_sample_field(np.asarray(c_field), L, x_s, y_s)
    return float(np.mean(vals))


def azimuthal_mean_at_radius_numpy(
    c_field: np.ndarray,
    *,
    L: float,
    r_abs: float,
    n_theta: int = 360,
) -> float:
    """Azimuthal mean of c on the circle |x − centre| = r_abs."""
    xc = L / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    x_s = xc + r_abs * np.cos(theta)
    y_s = xc + r_abs * np.sin(theta)
    vals = bilinear_sample_field(np.asarray(c_field), L, x_s, y_s)
    return float(np.mean(vals))


def silica_mass_inside_disk_numpy(
    c_field: np.ndarray,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    r_disk: float,
    rho_m: float,
    rho_c: float,
) -> float:
    """Integrate total silica (c + ρ_m φ_m + ρ_c φ_c) over disk r < r_disk."""
    n = int(c_field.shape[0])
    dx = L / n
    xc = L / 2.0
    xs = (np.arange(n, dtype=np.float64) + 0.5) * dx
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    rv = np.sqrt((xv - xc) ** 2 + (yv - xc) ** 2)
    mask = rv < r_disk
    rho_total = np.asarray(c_field) + rho_m * np.asarray(phi_m) + rho_c * np.asarray(phi_c)
    return float(np.sum(rho_total * mask) * dx**2)


def dissolved_silica_mass_inside_disk_numpy(
    c_field: np.ndarray,
    *,
    L: float,
    r_disk: float,
) -> float:
    """Integrate dissolved silica c over disk r < r_disk (matches Fick flux of c)."""
    n = int(c_field.shape[0])
    dx = L / n
    xc = L / 2.0
    xs = (np.arange(n, dtype=np.float64) + 0.5) * dx
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    rv = np.sqrt((xv - xc) ** 2 + (yv - xc) ** 2)
    mask = rv < r_disk
    return float(np.sum(np.asarray(c_field) * mask) * dx**2)


def radial_shell_widths_from_dx(
    dx: float,
    R: float,
    *,
    outer_dx: float = 3.0,
    inner_dx: float = 5.0,
) -> tuple[float, float]:
    """Radii inside dynamics (past Dirichlet forcing): default R−3dx, R−5dx."""
    return R - outer_dx * dx, R - inner_dx * dx


def compute_influx_rate_physical_jnp(
    c_field: jnp.ndarray,
    *,
    L: float,
    R: float,
    D_c: float,
    dx: float,
    outer_dx: float = 3.0,
    inner_dx: float = 5.0,
    n_theta: int = 72,
) -> jnp.ndarray:
    """Finite-difference radial gradient between R−3dx and R−5dx; flux at r=R−3dx."""
    xc = jnp.asarray(L / 2.0)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    r_outer = jnp.asarray(R - outer_dx * dx)
    r_inner = jnp.asarray(R - inner_dx * dx)
    x_o = xc + r_outer * jnp.cos(theta)
    y_o = xc + r_outer * jnp.sin(theta)
    x_i = xc + r_inner * jnp.cos(theta)
    y_i = xc + r_inner * jnp.sin(theta)
    samp_o = vmap(lambda xa, ya: _bilinear_sample_jnp(c_field, L, xa, ya))(x_o, y_o)
    samp_i = vmap(lambda xa, ya: _bilinear_sample_jnp(c_field, L, xa, ya))(x_i, y_i)
    c_outer = jnp.mean(samp_o)
    c_inner = jnp.mean(samp_i)
    dr_shell = r_outer - r_inner
    dc_dr = (c_outer - c_inner) / jnp.maximum(dr_shell, jnp.asarray(1e-12))
    perimeter = 2.0 * jnp.pi * r_outer
    return jnp.asarray(D_c) * dc_dr * perimeter


def compute_influx_rate_physical(
    c_field: np.ndarray,
    *,
    L: float,
    R: float,
    D_c: float,
    c_0: float | None = None,
    outer_dx: float = 3.0,
    inner_dx: float = 5.0,
    n_theta: int = 360,
) -> float:
    """Estimate diffusive silica influx [mass/time] through cylinder ``r = r_outer``.

    Uses Fick's law with a finite-difference radial gradient between two circles
    inside the dynamics (defaults ``R−3dx`` and ``R−5dx``, past Dirichlet forcing).

    Comparing ∫(influx_rate) dt to the **full-cavity** silica gain often gives a ratio
    not near 1: precipitation and a coarse two-point gradient are not the same as a
    surface integral of the true normal flux. Optionally widen the stencil via YAML
    ``physical_flux_outer_dx`` / ``physical_flux_inner_dx``.

    ``c_0`` is unused (call-site compatibility).
    """
    del c_0
    n = int(c_field.shape[0])
    dx = L / n
    r_outer, r_inner = radial_shell_widths_from_dx(dx, R, outer_dx=outer_dx, inner_dx=inner_dx)
    c_outer = azimuthal_mean_at_radius_numpy(c_field, L=L, r_abs=r_outer, n_theta=n_theta)
    c_inner = azimuthal_mean_at_radius_numpy(c_field, L=L, r_abs=r_inner, n_theta=n_theta)
    dr_shell = r_outer - r_inner
    dc_dr = (c_outer - c_inner) / max(dr_shell, 1e-12)
    perimeter = 2.0 * np.pi * r_outer
    return float(D_c * dc_dr * perimeter)


def flux_validation_milestone_line(
    *,
    t: float,
    influx_rate: float,
    d_c_dt_disk: float,
) -> str:
    """Compare Fick influx to backward difference of ∫ c dA inside flux cylinder."""
    if not (d_c_dt_disk == d_c_dt_disk) or abs(d_c_dt_disk) < 1e-30:
        ratio_s = "n/a"
    else:
        ratio_s = f"{influx_rate / d_c_dt_disk:.3f}"
    return (
        f"[t={t:.2f}] influx_rate={influx_rate:.3f}, "
        f"d(c_disk)/dt={d_c_dt_disk:.3f}, ratio={ratio_s}"
    )


def azimuthal_variance_at_radius(
    field: np.ndarray,
    *,
    L: float,
    R: float,
    r_frac: float = 0.75,
    n_theta: int = 360,
) -> float:
    xc = L / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr = float(r_frac) * R
    x_s = xc + rr * np.cos(theta)
    y_s = xc + rr * np.sin(theta)
    vals = bilinear_sample_field(np.asarray(field), L, x_s, y_s)
    return float(np.var(vals))


def labyrinth_heuristic(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    final_band_count: int,
) -> bool:
    """Rough auto-detect: low band count or dominance of θ-variation over radial."""
    if final_band_count < 10:
        return True
    tot = np.asarray(phi_m) + np.asarray(phi_c)
    az_v = azimuthal_variance_at_radius(tot, L=L, R=R)
    rc, prof = radial_profile(tot, L=L, R=R)
    rad_v = float(np.var(prof))
    if rad_v < 1e-18:
        return az_v > 1e-9
    return az_v > 1.2 * rad_v


def total_silica_numpy(
    c: np.ndarray,
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
    rho_m: float,
    rho_c: float,
) -> float:
    """Same silica norm as solver (numpy, host arrays)."""
    geom = build_geometry(L, R, int(c.shape[0]))
    chi = np.asarray(geom.chi)
    dx = float(geom.dx)
    return float(np.sum((c + rho_m * phi_m + rho_c * phi_c) * chi) * dx**2)


def boundary_flux_mass_rate(c: np.ndarray, L: float, R: float, D_c: float, c_0: float) -> float:
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
    """Fraction outside soft clip [-0.05, 1.05] (matches IMEX clip); all pixels."""
    both = np.concatenate([phi_m.ravel(), phi_c.ravel()])
    bad = np.sum((both < -0.05) | (both > 1.05))
    return float(100.0 * bad / max(both.size, 1))


def overshoot_slack_fraction(phi_m: np.ndarray, phi_c: np.ndarray) -> float:
    """Alias for overshoot_fraction (legacy name)."""
    return overshoot_fraction(phi_m, phi_c)


def overshoot_slack_fraction_cavity(
    phi_m: np.ndarray,
    phi_c: np.ndarray,
    *,
    L: float,
    R: float,
) -> float:
    """Outside [-0.05, 1.05] restricted to cavity mask χ > 0.5."""
    n = phi_m.shape[0]
    geom = build_geometry(L, R, n)
    chi = np.asarray(geom.chi)
    m = chi > 0.5
    both = np.concatenate([phi_m.ravel()[m.ravel()], phi_c.ravel()[m.ravel()]])
    if both.size == 0:
        return 0.0
    bad = np.sum((both < -0.05) | (both > 1.05))
    return float(100.0 * bad / both.size)


def band_metrics(
    r_centers: np.ndarray,
    phi_tot_profile: np.ndarray,
    cavity_R: float,
    *,
    prominence_frac: float = 0.05,
    distance: int = 1,
) -> dict[str, Any]:
    """Legacy azimuthal profile (plots only — not classification)."""
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
    cv_q = float(std_q / mean_q) if qq.size and abs(mean_q) > 1e-12 else float("nan")

    klass = (
        "INSUFFICIENT BANDS"
        if nb < 3
        else ("LIESEGANG-LIKE" if mean_q > 1.05 and cv_q < 0.1 else "NON-LIESEGANG")
    )

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
    """Time-resolved multislice counts + kymograph from canonical φ_c peaks."""
    records: list[tuple[int, int, list[float]]] = []
    kymo_t: list[float] = []
    kymo_r: list[float] = []

    peak_prof_metrics: dict[str, Any] | None = None
    final_prof_metrics: dict[str, Any] | None = None
    peak_rc_arr: np.ndarray | None = None
    peak_pt_arr: np.ndarray | None = None
    max_peaks = -1
    max_step = 0
    final_multislice_band_count = 0

    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            step = int(k.split("_")[1])
            if step < skip_before:
                continue
            pm = np.asarray(h5[k]["phi_m"])
            pc = np.asarray(h5[k]["phi_c"])
            nb_ms = count_bands_multislice(pm, pc, L, R)
            xr, vc = horizontal_right_half(pc, L, R)
            peaks, _ = find_peaks(vc, height=0.5, distance=3)
            if len(peaks) < 3:
                xr, vm = horizontal_right_half(pm, L, R)
                peaks, _ = find_peaks(vm, height=0.5, distance=3)
                xp = xr[peaks]
            else:
                xp = xr[peaks]
            rpk = [float(xp_i - L / 2) for xp_i in xp]
            records.append((step, nb_ms, rpk))
            t_phys = step * dt
            for rp in rpk:
                kymo_t.append(t_phys)
                kymo_r.append(rp)
            if nb_ms > max_peaks:
                max_peaks = nb_ms
                max_step = step

        if keys:
            last = keys[-1]
            pm = np.asarray(h5[last]["phi_m"])
            pc = np.asarray(h5[last]["phi_c"])
            final_multislice_band_count = count_bands_multislice(pm, pc, L, R)
            final_prof_metrics = jab_metrics_canonical_slice(pm, pc, L, R, R)

        if max_peaks >= 1 and max_step >= skip_before:
            gname = f"t_{max_step:07d}"
            if gname in h5:
                pm = np.asarray(h5[gname]["phi_m"])
                pc = np.asarray(h5[gname]["phi_c"])
                peak_prof_metrics = jab_metrics_canonical_slice(pm, pc, L, R, R)
                rc, pt = radial_profile(pm + pc, L=L, R=R)
                peak_rc_arr = rc
                peak_pt_arr = pt

    if peak_prof_metrics is None:
        peak_prof_metrics = {"N_b": 0, "classification": "INSUFFICIENT BANDS"}

    return {
        "records": records,
        "kymograph_t": kymo_t,
        "kymograph_r": kymo_r,
        "final_multislice_band_count": final_multislice_band_count,
        "peak_band_count": max(0, max_peaks),
        "peak_band_count_step": max_step,
        "peak_band_count_time": float(max_step * dt),
        "metrics_at_peak": peak_prof_metrics,
        "metrics_at_final": final_prof_metrics or {},
        "peak_radial_centers": peak_rc_arr,
        "peak_radial_profile": peak_pt_arr,
    }
