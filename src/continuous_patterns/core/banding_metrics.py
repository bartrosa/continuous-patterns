"""Model-agnostic banding record and Liesegang-law metrics.

Conventions follow the Liesegang literature: band positions are precipitate
density maxima measured from the nucleation surface; widths are FWHM;
laws computed: spacing (Jablczynski), width, time (Morse-Pierce).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import signal, stats

from continuous_patterns.core.diagnostics_slab import detect_profile_peaks
from continuous_patterns.core.types import SimResult

NucleationSide = Literal["left", "right"]

_INSUFFICIENT = "insufficient_bands"


@dataclass
class BandingRecord:
    """Unified banding description, model-agnostic."""

    source: str  # "label_history" | "profile"
    model: str
    run_label: str
    band_positions: np.ndarray  # band centers measured from nucleation wall
    band_widths: np.ndarray  # segment width (FWHM or label segment)
    band_labels: list[str]
    band_times: np.ndarray | None  # formation time per band (None for profile)
    front_pos_t: np.ndarray | None  # (t, front position) shape (k, 2) or None
    c_front_t: np.ndarray | None  # (t, c at wall / cavity mean) shape (k, 2)
    domain_length: float
    extra: dict = field(default_factory=dict)


def _profile_x_coords(n: int, domain_length: float) -> np.ndarray:
    dx = float(domain_length) / float(n)
    return (np.arange(n, dtype=np.float64) + 0.5) * dx


def _fwhm_widths(profile: np.ndarray, x: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    if peak_indices.size == 0:
        return np.asarray([], dtype=np.float64)
    _widths, _, left_ips, right_ips = signal.peak_widths(
        profile,
        peak_indices,
        rel_height=0.5,
    )
    left_x = np.interp(left_ips, np.arange(x.size, dtype=np.float64), x)
    right_x = np.interp(right_ips, np.arange(x.size, dtype=np.float64), x)
    return (right_x - left_x).astype(np.float64)


def profile_to_label_segments(
    profile_x: np.ndarray,
    profile: np.ndarray,
    *,
    solid_tol_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Contiguous sign segments covering the active deposit extent (no gaps)."""
    x = np.asarray(profile_x, dtype=np.float64).ravel()
    prof = np.asarray(profile, dtype=np.float64).ravel()
    if x.size != prof.size or x.size == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
        )

    amp = float(np.max(np.abs(prof)))
    if amp < 1e-14:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
        )

    tol = solid_tol_frac * amp
    active = np.abs(prof) >= tol
    if not np.any(active):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            [],
        )

    i0 = int(np.argmax(active))
    i1 = int(x.size - 1 - np.argmax(active[::-1]))
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0

    positions: list[float] = []
    widths: list[float] = []
    labels: list[str] = []

    i = i0
    while i <= i1:
        if not active[i]:
            i += 1
            continue
        sign = 1 if prof[i] > 0.0 else -1
        j = i + 1
        while j <= i1 and active[j] and ((prof[j] > 0.0) == (sign > 0)):
            j += 1
        left_edge = float(x[i]) - 0.5 * dx
        right_edge = float(x[j - 1]) + 0.5 * dx
        positions.append(0.5 * (left_edge + right_edge))
        widths.append(right_edge - left_edge)
        labels.append("m" if sign > 0 else "c")
        i = j

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(widths, dtype=np.float64),
        labels,
    )


def deposit_extent(positions: np.ndarray, widths: np.ndarray) -> float:
    """Total labeled deposit extent (sum of segment widths)."""
    w = np.asarray(widths, dtype=np.float64).ravel()
    return float(np.sum(w)) if w.size else 0.0


def _segment_right_edge(positions: np.ndarray, widths: np.ndarray, index: int) -> float:
    return float(positions[index]) + 0.5 * float(widths[index])


def alternating_zone_metrics(
    positions: np.ndarray,
    widths: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Metrics for the m/c alternating zone (terminal uniform block excluded).

    ``band_zone_extent`` is the right edge of the last band followed by a band
    of the other label. Spacings are center-to-center between consecutive bands
    inside the zone.
    """
    pos = np.asarray(positions, dtype=np.float64).ravel()
    wid = np.asarray(widths, dtype=np.float64).ravel()
    labs = [str(lb) for lb in labels]
    n = int(pos.size)
    nan = float("nan")
    if n == 0:
        return {
            "band_zone_extent": 0.0,
            "n_bands_zone": 0,
            "mean_spacing_zone": nan,
            "median_spacing_zone": nan,
            "mean_spacing_zone_excl_last": nan,
            "std_spacing_zone": nan,
            "zone_band_positions": [],
        }
    if n == 1:
        return {
            "band_zone_extent": _segment_right_edge(pos, wid, 0),
            "n_bands_zone": 1,
            "mean_spacing_zone": nan,
            "median_spacing_zone": nan,
            "mean_spacing_zone_excl_last": nan,
            "std_spacing_zone": nan,
            "zone_band_positions": pos.tolist(),
        }

    last_alt = -1
    for i in range(n - 1):
        if labs[i] != labs[i + 1]:
            last_alt = i

    if last_alt < 0:
        return {
            "band_zone_extent": _segment_right_edge(pos, wid, n - 1),
            "n_bands_zone": n,
            "mean_spacing_zone": nan,
            "median_spacing_zone": nan,
            "mean_spacing_zone_excl_last": nan,
            "std_spacing_zone": nan,
            "zone_band_positions": pos.tolist(),
        }

    zone_end = last_alt + 1
    zone_pos = pos[: zone_end + 1]
    zone_w = wid[: zone_end + 1]
    extent = _segment_right_edge(zone_pos, zone_w, last_alt)
    if zone_pos.size >= 2:
        d_sp = np.diff(zone_pos)
        mean_d = float(np.mean(d_sp))
        median_d = float(np.median(d_sp))
        std_d = float(np.std(d_sp))
        mean_excl = float(np.mean(d_sp[:-1])) if d_sp.size >= 2 else nan
    else:
        mean_d = median_d = std_d = mean_excl = nan

    return {
        "band_zone_extent": extent,
        "n_bands_zone": int(zone_end + 1),
        "mean_spacing_zone": mean_d,
        "median_spacing_zone": median_d,
        "mean_spacing_zone_excl_last": mean_excl,
        "std_spacing_zone": std_d,
        "zone_band_positions": zone_pos.tolist(),
    }


def concentration_ahead_of_front(c: np.ndarray, phi_tot: np.ndarray) -> float:
    """``c`` one cell ahead of the last cell with ``phi_tot > 0.5``."""
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    phi = np.asarray(phi_tot, dtype=np.float64).ravel()
    idx = np.where(phi > 0.5)[0]
    if not idx.size:
        return float(c_arr[0])
    j = int(min(int(idx[-1]) + 1, c_arr.size - 1))
    return float(c_arr[j])


def front_velocity_series(
    snap_t: np.ndarray,
    snap_x_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Front velocity from a front-position time series (finite differences)."""
    t = np.asarray(snap_t, dtype=np.float64).ravel()
    x = np.asarray(snap_x_front, dtype=np.float64).ravel()
    if t.size < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dt = np.diff(t)
    dx = np.diff(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = dx / np.maximum(dt, 1e-30)
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t_mid, v


def band_formation_times(
    band_positions: np.ndarray,
    snap_t: np.ndarray,
    snap_x_front: np.ndarray,
) -> np.ndarray:
    """Estimate formation time when the front first reaches each band center."""
    pos = np.asarray(band_positions, dtype=np.float64).ravel()
    t = np.asarray(snap_t, dtype=np.float64).ravel()
    xf = np.asarray(snap_x_front, dtype=np.float64).ravel()
    if pos.size == 0 or t.size == 0 or xf.size == 0:
        return np.asarray([], dtype=np.float64)
    out = np.empty(pos.size, dtype=np.float64)
    for i, p in enumerate(pos):
        hits = np.flatnonzero(xf >= p)
        if hits.size:
            out[i] = float(t[int(hits[0])])
        else:
            out[i] = float(t[-1])
    return out


def interpolate_series(t_query: np.ndarray, t_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    t_q = np.asarray(t_query, dtype=np.float64).ravel()
    t_s = np.asarray(t_src, dtype=np.float64).ravel()
    y_s = np.asarray(y_src, dtype=np.float64).ravel()
    if t_q.size == 0 or t_s.size == 0:
        return np.asarray([], dtype=np.float64)
    if t_s.size == 1:
        return np.full(t_q.size, float(y_s[0]), dtype=np.float64)
    return np.interp(t_q, t_s, y_s).astype(np.float64)


def s1_twophase_validity(diagnostics: dict[str, Any]) -> str:
    """Mass / void-fraction validity tiers (clip_rate is informational only)."""
    mass = diagnostics.get("mass_residual_pct")
    min_pw = diagnostics.get("min_phi_w_wetted", diagnostics.get("min_phi_w"))
    if min_pw is not None and float(min_pw) < -1e-6:
        return "NUMERICALLY_INVALID"
    if mass is not None:
        m = float(mass)
        if m >= 1e-1:
            return "NUMERICALLY_INVALID"
        if m >= 1e-3:
            return "WARN"
    return "OK"


def compute_clip_rate(
    n_projection_events: int | float | None,
    n_steps: int | float | None,
    n_cells: int | float | None,
) -> float | None:
    """Fraction of cell-steps that triggered a projection event, in [0, 1] if defined."""
    if n_projection_events is None or n_steps is None or n_cells is None:
        return None
    denom = float(n_steps) * float(n_cells)
    if denom <= 0.0:
        return None
    return float(n_projection_events) / denom


def _has_terminal_band_spike(metrics: dict[str, Any]) -> bool:
    """True when the last position-ratio point is an outlier vs the tail."""
    q_raw = metrics.get("Q_ratios")
    if not isinstance(q_raw, list) or len(q_raw) < 3:
        return False
    arr = np.asarray(q_raw, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False
    tail_med = float(np.median(arr[:-1]))
    if tail_med <= 0:
        return False
    return float(arr[-1]) > 5.0 * tail_med


def compute_front_origin(
    *,
    solid_xt: np.ndarray,
    frame_times: np.ndarray,
    x: np.ndarray,
    threshold: float = 0.3,
    seed_width_frac: float = 0.05,
) -> dict[str, Any]:
    """Measured growth-direction diagnostic from a solid-indicator kymograph."""
    solid = np.asarray(solid_xt, dtype=np.float64)
    times = np.asarray(frame_times, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()

    if solid.ndim != 2 or solid.shape[1] != x_arr.size or times.size != solid.shape[0]:
        raise ValueError("solid_xt must have shape (n_frames, n_x) matching x and frame_times")

    n_x = x_arr.size
    t_arr = np.full(n_x, np.inf, dtype=np.float64)
    for ix in range(n_x):
        hits = np.flatnonzero(solid[:, ix] > threshold)
        if hits.size:
            t_arr[ix] = float(times[int(hits[0])])

    finite = np.isfinite(t_arr) & (t_arr < np.inf)
    if not np.any(finite):
        return {
            "origin": "NONE",
            "t_arr": t_arr,
            "spearman_rho": float("nan"),
            "meeting_x": float("nan"),
        }

    x_valid = x_arr[finite]
    t_valid = t_arr[finite]
    if x_valid.size < 2:
        rho = float("nan")
    else:
        rho, _ = stats.spearmanr(x_valid, t_valid)
        rho = float(rho)

    x_span = float(x_arr[-1] - x_arr[0]) if n_x > 1 else 1.0
    seed_w = max(float(x_arr[1] - x_arr[0]) if n_x > 1 else x_span, seed_width_frac * x_span)
    advanced = finite & (x_arr > x_arr[0] + seed_w)
    if not np.any(advanced):
        return {
            "origin": "NONE",
            "t_arr": t_arr,
            "spearman_rho": rho,
            "meeting_x": float("nan"),
        }

    if rho > 0.8:
        origin = "LEFT"
        meeting_x = float("nan")
    elif rho < -0.8:
        origin = "RIGHT"
        meeting_x = float("nan")
    else:
        t_finite = np.where(finite, t_arr, -np.inf)
        med_t = float(np.nanmedian(t_arr[finite]))
        left_early = finite[0] and t_arr[0] <= med_t
        right_early = finite[-1] and t_arr[-1] <= med_t
        interior_idx = int(np.argmax(t_finite))
        interior_late = (
            0 < interior_idx < n_x - 1
            and finite[interior_idx]
            and t_arr[interior_idx] > max(t_arr[0], t_arr[-1])
        )
        if left_early and right_early and interior_late:
            origin = "BOTH"
            meeting_x = float(x_arr[interior_idx])
        elif rho >= 0.0:
            origin = "LEFT"
            meeting_x = float("nan")
        else:
            origin = "RIGHT"
            meeting_x = float("nan")

    return {
        "origin": origin,
        "t_arr": t_arr,
        "spearman_rho": rho,
        "meeting_x": meeting_x,
    }


def compute_front_origin_from_sk_history(
    t_history: np.ndarray,
    thickness_history: np.ndarray,
    x: np.ndarray,
    *,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Heaney (slab_kinetic) adapter: deposit thickness h(t) defines a left-origin solid field."""
    t_hist = np.asarray(t_history, dtype=np.float64).ravel()
    th_hist = np.asarray(thickness_history, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    if t_hist.size == 0:
        return {
            "origin": "NONE",
            "t_arr": np.asarray([], dtype=np.float64),
            "spearman_rho": float("nan"),
            "meeting_x": float("nan"),
        }

    h_max = float(np.max(th_hist)) if th_hist.size else 0.0
    if h_max <= 0.0:
        return {
            "origin": "NONE",
            "t_arr": np.full(x_arr.size, np.inf),
            "spearman_rho": float("nan"),
            "meeting_x": float("nan"),
        }

    # Downsample time axis for Spearman (monotone LEFT by construction).
    n_frames = min(120, t_hist.size)
    idx = np.linspace(0, t_hist.size - 1, n_frames, dtype=int)
    times = t_hist[idx]
    h_sub = th_hist[idx]
    solid = (x_arr[None, :] <= h_sub[:, None]).astype(np.float64)
    out = compute_front_origin(
        solid_xt=solid,
        frame_times=times,
        x=x_arr,
        threshold=threshold,
    )
    out["origin"] = "LEFT"
    out["spearman_rho"] = 1.0
    return out


def front_position_vs_time(
    solid_xt: np.ndarray,
    frame_times: np.ndarray,
    x: np.ndarray,
    *,
    threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Leading solid edge position vs time (max x with solid > threshold per frame)."""
    solid = np.asarray(solid_xt, dtype=np.float64)
    times = np.asarray(frame_times, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    front = np.full(solid.shape[0], np.nan, dtype=np.float64)
    for i in range(solid.shape[0]):
        hits = np.flatnonzero(solid[i, :] > threshold)
        if hits.size:
            front[i] = float(x_arr[int(hits[-1])])
    return times, front


def _mirror_record_from_right(rec: BandingRecord, domain_length: float) -> BandingRecord:
    """Express band positions as distance from the right nucleation wall."""
    L = float(domain_length)
    pos = L - np.asarray(rec.band_positions, dtype=np.float64)
    order = np.argsort(pos)
    rec.band_positions = pos[order]
    rec.band_widths = np.asarray(rec.band_widths, dtype=np.float64)[order]
    rec.band_labels = [rec.band_labels[i] for i in order]
    rec.extra = {**rec.extra, "nucleation_side": "right", "position_origin": "right_wall"}
    return rec


def record_from_label_history(
    *,
    model: str,
    run_label: str,
    t_history: np.ndarray,
    thickness_history: np.ndarray,
    state_history: np.ndarray,
    transitions: list[tuple[float, str]],
    c_at_x0_history: np.ndarray,
    domain_length: float = 1.0,
) -> BandingRecord:
    """Build a label-history record from kinetic L/H transition events."""
    t_hist = np.asarray(t_history, dtype=np.float64).ravel()
    th_hist = np.asarray(thickness_history, dtype=np.float64).ravel()
    c0_hist = np.asarray(c_at_x0_history, dtype=np.float64).ravel()
    if not (t_hist.size == th_hist.size == c0_hist.size):
        raise ValueError("t_history, thickness_history, c_at_x0_history must match in length")

    t_final = float(t_hist[-1]) if t_hist.size else 0.0
    th_final = float(th_hist[-1]) if th_hist.size else 0.0

    boundary_times = [0.0]
    boundary_thickness = [0.0]
    for tv, _kind in sorted(transitions, key=lambda item: float(item[0])):
        t_ev = float(tv)
        if t_ev <= boundary_times[-1] or t_ev > t_final + 1e-15:
            continue
        idx = int(np.searchsorted(t_hist, t_ev, side="left"))
        idx = min(max(idx, 0), t_hist.size - 1)
        boundary_times.append(t_ev)
        boundary_thickness.append(float(th_hist[idx]))

    boundary_times.append(t_final)
    boundary_thickness.append(th_final)

    n_segments = len(boundary_times) - 1
    positions = np.empty(n_segments, dtype=np.float64)
    widths = np.empty(n_segments, dtype=np.float64)
    times = np.empty(n_segments, dtype=np.float64)
    labels: list[str] = []

    state = "L"
    for i in range(n_segments):
        widths[i] = float(boundary_thickness[i + 1] - boundary_thickness[i])
        positions[i] = 0.5 * (float(boundary_thickness[i]) + float(boundary_thickness[i + 1]))
        times[i] = float(boundary_times[i + 1])
        labels.append(state)
        if i < n_segments - 1:
            state = "H" if state == "L" else "L"

    front_pos_t = np.column_stack([t_hist, th_hist]) if t_hist.size else None
    c_front_t = np.column_stack([t_hist, c0_hist]) if t_hist.size else None

    return BandingRecord(
        source="label_history",
        model=model,
        run_label=run_label,
        band_positions=positions,
        band_widths=widths,
        band_labels=labels,
        band_times=times,
        front_pos_t=front_pos_t,
        c_front_t=c_front_t,
        domain_length=float(domain_length if domain_length > 0 else th_final),
        extra={"thickness_final": th_final, "t_final": t_final, "nucleation_side": "left"},
    )


def record_from_profile(
    *,
    model: str,
    run_label: str,
    profile_x: np.ndarray,
    profile: np.ndarray,
    domain_length: float,
    nucleation_side: NucleationSide = "left",
    prominence_frac: float = 0.05,
) -> BandingRecord:
    """Build a profile record with contiguous sign segments (full deposit coverage)."""
    x = np.asarray(profile_x, dtype=np.float64).ravel()
    prof = np.asarray(profile, dtype=np.float64).ravel()
    if x.size != prof.size:
        raise ValueError("profile_x and profile must have the same length")

    positions, widths, labels = profile_to_label_segments(x, prof)
    detected = detect_profile_peaks(prof, x, prominence_frac=prominence_frac)

    rec = BandingRecord(
        source="profile",
        model=model,
        run_label=run_label,
        band_positions=positions,
        band_widths=widths,
        band_labels=labels,
        band_times=None,
        front_pos_t=None,
        c_front_t=None,
        domain_length=float(domain_length),
        extra={
            "profile_x": x,
            "profile": prof,
            "prominence_threshold": float(detected["prominence_threshold"]),
            "peak_positions": np.asarray(detected["peak_positions"], dtype=np.float64),
            "n_peaks": int(detected["peak_indices"].size),
            "nucleation_side": nucleation_side,
        },
    )
    if nucleation_side == "right":
        rec = _mirror_record_from_right(rec, domain_length)
    return rec


def _nan_laws(n_bands: int, *, reason: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        "n_bands": n_bands,
        "Q_ratios": [],
        "p_fit": nan,
        "Q_cv": nan,
        "Q_cv_tail": nan,
        "p_fit_excl_last": nan,
        "Q_cv_tail_excl_last": nan,
        "d_spacings": [],
        "q_ratios": [],
        "q_cv": nan,
        "spearman_d_vs_n": nan,
        "width_law": {"slope": nan, "intercept": nan, "r2": nan},
        "time_law": {"alpha": nan, "A": nan, "r2": nan},
        "mean_band_width": nan,
        "total_deposit": deposit_extent(np.asarray([]), np.asarray([])),
        "laws_reason": reason,
    }


def _tail_position_laws(q_ratios: np.ndarray) -> tuple[float, float]:
    if q_ratios.size == 0:
        return float("nan"), float("nan")
    tail = int(max(3, q_ratios.size // 2))
    tail_q = q_ratios[-tail:]
    return float(np.mean(tail_q) - 1.0), _coeff_var(tail_q)


def _excl_last_position_laws(x: np.ndarray) -> tuple[float, float]:
    """Tail laws after dropping the terminal band position."""
    x_excl = np.asarray(x, dtype=np.float64).ravel()[:-1]
    if x_excl.size < 5:
        return float("nan"), float("nan")
    if not np.all(x_excl[:-1] > 1e-30):
        return float("nan"), float("nan")
    q_excl = x_excl[1:] / x_excl[:-1]
    if q_excl.size < 4:
        return float("nan"), float("nan")
    return _tail_position_laws(q_excl)


def _coeff_var(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 2:
        return float("nan")
    mu = float(np.mean(arr))
    if abs(mu) < 1e-30:
        return float("nan")
    return float(np.std(arr) / mu)


def _linreg_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(x, y)
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _power_law_fit(x: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    mask = (x > 0.0) & (t > 0.0) & np.isfinite(x) & np.isfinite(t)
    xv = x[mask]
    tv = t[mask]
    if xv.size < 3:
        return float("nan"), float("nan"), float("nan")
    lx = np.log(xv)
    lt = np.log(tv)
    alpha, log_a, r2 = _linreg_r2(lt, lx)
    a = float(np.exp(log_a)) if np.isfinite(log_a) else float("nan")
    return alpha, a, r2


def compute_liesegang_laws(rec: BandingRecord) -> dict[str, Any]:
    """Compute spacing, width, and (when available) time laws for a banding record."""
    n_bands_record = int(np.asarray(rec.band_positions, dtype=np.float64).size)

    peak_x = rec.extra.get("peak_positions")
    if peak_x is not None and np.asarray(peak_x).size >= 6:
        x = np.asarray(peak_x, dtype=np.float64).ravel()
        prof = rec.extra.get("profile")
        px = rec.extra.get("profile_x")
        if prof is not None and px is not None:
            detected = detect_profile_peaks(
                np.asarray(prof, dtype=np.float64),
                np.asarray(px, dtype=np.float64),
            )
            w = _fwhm_widths(
                np.asarray(prof, dtype=np.float64),
                np.asarray(px, dtype=np.float64),
                detected["peak_indices"],
            )
        else:
            w = np.full_like(x, np.nan)
    else:
        x = np.asarray(rec.band_positions, dtype=np.float64).ravel()
        w = np.asarray(rec.band_widths, dtype=np.float64).ravel()

    n_bands = n_bands_record

    if n_bands < 6:
        out = _nan_laws(n_bands, reason=_INSUFFICIENT)
        out["total_deposit"] = deposit_extent(
            np.asarray(rec.band_positions, dtype=np.float64),
            np.asarray(rec.band_widths, dtype=np.float64),
        )
        seg_w = np.asarray(rec.band_widths, dtype=np.float64)
        if seg_w.size:
            out["mean_band_width"] = float(np.mean(seg_w))
        if peak_x is not None:
            out["n_peaks"] = int(np.asarray(peak_x).size)
        out["n_segments"] = n_bands_record
        return out

    if np.all(x[:-1] > 1e-30):
        q_ratios = x[1:] / x[:-1]
    else:
        q_ratios = np.asarray([], dtype=np.float64)

    p_fit, q_cv_tail = _tail_position_laws(q_ratios)
    q_cv = _coeff_var(q_ratios)
    p_fit_excl_last, q_cv_tail_excl_last = _excl_last_position_laws(x)
    d_spacings = x[1:] - x[:-1] if n_bands >= 2 else np.asarray([], dtype=np.float64)

    if d_spacings.size >= 2:
        q_spacing = d_spacings[1:] / d_spacings[:-1]
    else:
        q_spacing = np.asarray([], dtype=np.float64)

    q_spacing_cv = _coeff_var(q_spacing)

    if d_spacings.size >= 3:
        rho, _p = stats.spearmanr(np.arange(1, d_spacings.size + 1), d_spacings)
        spearman_d = float(rho)
    else:
        spearman_d = float("nan")

    w_slope, w_intercept, w_r2 = _linreg_r2(x, w)

    if rec.band_times is not None:
        t_arr = np.asarray(rec.band_times, dtype=np.float64).ravel()
        alpha, a_time, t_r2 = _power_law_fit(x, t_arr)
    else:
        alpha, a_time, t_r2 = float("nan"), float("nan"), float("nan")

    return {
        "n_bands": n_bands,
        "Q_ratios": q_ratios.tolist(),
        "p_fit": p_fit,
        "Q_cv": q_cv,
        "Q_cv_tail": q_cv_tail,
        "p_fit_excl_last": p_fit_excl_last,
        "Q_cv_tail_excl_last": q_cv_tail_excl_last,
        "d_spacings": d_spacings.tolist(),
        "q_ratios": q_spacing.tolist(),
        "q_cv": q_spacing_cv,
        "spearman_d_vs_n": spearman_d,
        "width_law": {"slope": w_slope, "intercept": w_intercept, "r2": w_r2},
        "time_law": {"alpha": alpha, "A": a_time, "r2": t_r2},
        "mean_band_width": float(np.mean(w)) if w.size and np.all(np.isfinite(w)) else float("nan"),
        "total_deposit": deposit_extent(
            np.asarray(rec.band_positions, dtype=np.float64),
            np.asarray(rec.band_widths, dtype=np.float64),
        ),
        "laws_reason": None,
        "n_peaks": int(np.asarray(peak_x).size) if peak_x is not None else n_bands,
        "n_segments": int(np.asarray(rec.band_positions).size),
    }


def record_from_slab_twophase_result(result: SimResult, run_label: str) -> BandingRecord:
    """Adapter: slab_twophase SimResult → BandingRecord from ψ = φ_m − φ_c profile."""
    meta = result.meta if isinstance(result.meta, dict) else {}
    tp_raw = meta.get("slab_twophase")
    if not isinstance(tp_raw, dict):
        raise ValueError("slab_twophase result missing meta['slab_twophase'] block")

    phi_m = np.asarray(result.state_final.phi_m, dtype=np.float64).ravel()
    phi_c = np.asarray(result.state_final.phi_c, dtype=np.float64).ravel()
    g = tp_raw.get("geometry") if isinstance(tp_raw.get("geometry"), dict) else {}
    L = float(tp_raw.get("domain_L", g.get("L", 100.0)))
    n = int(phi_m.size)
    if g.get("dx") is not None:
        dx = float(g["dx"])
        x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    else:
        x = np.linspace(0.0, L, n, dtype=np.float64)
    profile = phi_m - phi_c

    snap_t = np.asarray(tp_raw.get("snapshots_t", []), dtype=np.float64)
    snap_c0 = np.asarray(tp_raw.get("snapshots_c0", []), dtype=np.float64)
    snap_cf = np.asarray(tp_raw.get("snapshots_c_front", []), dtype=np.float64)
    snap_xf = np.asarray(tp_raw.get("snapshots_x_front", []), dtype=np.float64)

    rec = record_from_profile(
        model="slab_twophase",
        run_label=run_label,
        profile_x=x,
        profile=profile,
        domain_length=L,
        nucleation_side="left",
    )
    extra = dict(rec.extra)
    extra["snapshots_t"] = snap_t
    extra["snapshots_c0"] = snap_c0
    extra["snapshots_c_front"] = snap_cf
    extra["snapshots_x_front"] = snap_xf
    extra["seed_exclude_x"] = float(tp_raw.get("seed_exclude_x", 0.0))
    return BandingRecord(
        source=rec.source,
        model=rec.model,
        run_label=rec.run_label,
        band_positions=rec.band_positions,
        band_widths=rec.band_widths,
        band_labels=rec.band_labels,
        band_times=rec.band_times,
        front_pos_t=snap_xf if snap_xf.size else None,
        c_front_t=snap_cf if snap_cf.size else (snap_c0 if snap_c0.size else None),
        domain_length=rec.domain_length,
        extra=extra,
    )


def record_from_slab_kinetic_result(result: SimResult, run_label: str) -> BandingRecord:
    """Adapter: slab_kinetic SimResult → BandingRecord from L/H label history."""
    meta = result.meta if isinstance(result.meta, dict) else {}
    sk_raw = meta.get("slab_kinetic")
    if not isinstance(sk_raw, dict):
        raise ValueError("slab_kinetic result missing meta['slab_kinetic'] block")

    transitions_raw = sk_raw.get("transitions", [])
    transitions: list[tuple[float, str]] = []
    if isinstance(transitions_raw, list):
        for item in transitions_raw:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                transitions.append((float(item[0]), str(item[1])))

    th_final = float(sk_raw.get("thickness_final", 0.0))
    return record_from_label_history(
        model="slab_kinetic",
        run_label=run_label,
        t_history=np.asarray(sk_raw["t_history"], dtype=np.float64),
        thickness_history=np.asarray(sk_raw["thickness_history"], dtype=np.float64),
        state_history=np.asarray(sk_raw["state_history"], dtype=np.float64),
        transitions=transitions,
        c_at_x0_history=np.asarray(sk_raw["c_at_x0_history"], dtype=np.float64),
        domain_length=th_final if th_final > 0 else 1.0,
    )


def record_from_slab_reactive_outputs(
    final_state_npz: Path,
    config: dict[str, Any],
    run_label: str,
    *,
    nucleation_side: NucleationSide = "left",
) -> BandingRecord:
    """Adapter: slab_reactive final_state.npz → BandingRecord from axial profile."""
    with np.load(final_state_npz) as z:
        phi_m = np.asarray(z["phi_m"], dtype=np.float64)
        phi_c = np.asarray(z["phi_c"], dtype=np.float64)
    gcfg = config.get("geometry", {})
    L = float(gcfg.get("L", 1.0))
    n = int(phi_m.shape[0])
    x = _profile_x_coords(n, L)
    profile = np.mean(phi_m - phi_c, axis=1)
    return record_from_profile(
        model="slab_reactive",
        run_label=run_label,
        profile_x=x,
        profile=profile,
        domain_length=L,
        nucleation_side=nucleation_side,
    )


def front_origin_for_record(
    rec: BandingRecord,
    *,
    snapshots: dict[str, np.ndarray] | None = None,
    history: dict[str, np.ndarray] | None = None,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Compute ``front_origin`` from run artifacts attached to a record."""
    if rec.model == "slab_kinetic" and history is not None:
        th_final = float(rec.domain_length)
        n_x = 64
        x = np.linspace(0.0, max(th_final, 1e-6), n_x, dtype=np.float64)
        return compute_front_origin_from_sk_history(
            history["t_history"],
            history["thickness_history"],
            x,
            threshold=threshold,
        )

    if rec.model == "slab_twophase" and snapshots is not None:
        times = np.asarray(snapshots.get("snapshots_t"), dtype=np.float64).ravel()
        solid = snapshots.get("snapshots_solid")
        if solid is not None and times.size:
            solid = np.asarray(solid, dtype=np.float64)
            domain = float(snapshots.get("domain_L", rec.domain_length))
            n_x = solid.shape[1]
            x = np.linspace(0.0, domain, n_x, dtype=np.float64)
            seed_x = float(snapshots.get("seed_exclude_x", rec.extra.get("seed_exclude_x", 0.0)))
            # ``seed_exclude_x`` is 3× the physical seed width (band-metric rim).
            seed_frac = (seed_x / 3.0) / max(domain, 1e-30)
            return compute_front_origin(
                solid_xt=solid,
                frame_times=times,
                x=x,
                threshold=min(threshold, 0.15),
                seed_width_frac=max(seed_frac, 0.01),
            )

    if snapshots is None:
        return {
            "origin": "NONE",
            "t_arr": np.asarray([], dtype=np.float64),
            "spearman_rho": float("nan"),
            "meeting_x": float("nan"),
        }

    times = np.asarray(snapshots.get("snapshots_t"), dtype=np.float64).ravel()
    if "snapshots_solid" in snapshots:
        solid = np.asarray(snapshots["snapshots_solid"], dtype=np.float64)
    elif "snapshots_profile" in snapshots:
        prof = np.asarray(snapshots["snapshots_profile"], dtype=np.float64)
        amp = float(np.max(np.abs(prof))) if prof.size else 1.0
        solid = (np.abs(prof) > 0.05 * amp).astype(np.float64)
    elif "snapshots_c" in snapshots:
        solid = np.asarray(snapshots["snapshots_c"], dtype=np.float64)
    else:
        return {
            "origin": "NONE",
            "t_arr": np.asarray([], dtype=np.float64),
            "spearman_rho": float("nan"),
            "meeting_x": float("nan"),
        }

    domain = float(snapshots.get("domain_L", rec.domain_length))
    n_x = solid.shape[1]
    x = (np.arange(n_x, dtype=np.float64) + 0.5) * (domain / n_x)
    return compute_front_origin(
        solid_xt=solid,
        frame_times=times,
        x=x,
        threshold=threshold,
    )
