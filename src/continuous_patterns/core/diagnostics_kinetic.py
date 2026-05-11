"""Diagnostics for kinetic models (1D banding via boundary kinetics).

Functions
---------
compute_transitions       — L↔H transition events (hysteresis replay)
compute_thickness         — cumulative ∫ r(state)·c(0,τ) dτ
compute_dwell_times       — durations spent in L vs H
compute_temporal_spectrum — FFT of c(x=0, t); dominant frequency / period

Pure NumPy + SciPy; no JAX; no plotting.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def compute_transitions(
    c_at_x0: np.ndarray,
    t: np.ndarray,
    c1: float,
    c2: float,
) -> list[tuple[float, str]]:
    """Replay hysteresis on a sampled trace and list L↔H transitions.

    Rules (same as the slab_kinetic integrator): start in ``L``. While in ``L``,
    switch to ``H`` when ``c(0) > c1``. While in ``H``, switch to ``L`` when
    ``c(0) < c2``. Transition times are taken at sample times *after* the
    condition is observed (the sample index where the crossing is detected).
    """
    c_at_x0 = np.asarray(c_at_x0, dtype=np.float64).ravel()
    tt = np.asarray(t, dtype=np.float64).ravel()
    if c_at_x0.size != tt.size or c_at_x0.size == 0:
        raise ValueError("c_at_x0 and t must be nonempty and the same length")

    state = "L"
    out: list[tuple[float, str]] = []
    for i in range(c_at_x0.size):
        cv = float(c_at_x0[i])
        tv = float(tt[i])
        if state == "L" and cv > float(c1):
            state = "H"
            out.append((tv, "L→H"))
        elif state == "H" and cv < float(c2):
            state = "L"
            out.append((tv, "H→L"))
    return out


def compute_thickness(
    c_at_x0: np.ndarray,
    state: np.ndarray,
    dt: float,
    kappa: float,
) -> np.ndarray:
    """Cumulative thickness ∫ r(state(τ))·c(0,τ) dτ using rectangle quadrature.

    ``state`` uses ``0`` for L (r=1) and ``1`` for H (r=kappa).
    """
    c_at_x0 = np.asarray(c_at_x0, dtype=np.float64).ravel()
    st = np.asarray(state, dtype=np.int32).ravel()
    if c_at_x0.size != st.size:
        raise ValueError("c_at_x0 and state must have the same shape")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    r = np.where(st != 0, float(kappa), 1.0)
    incr = float(dt) * r * c_at_x0
    return np.cumsum(incr)


def compute_dwell_times(
    transitions: list[tuple[float, str]],
    T_final: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Durations spent in L and in H between transitions over ``[0, T_final]``.

    Starts in ``L`` at ``t = 0``. Each interval ends at the transition instant.
    """
    tf = float(T_final)
    if tf < 0.0:
        raise ValueError("T_final must be >= 0")

    dwell_L: list[float] = []
    dwell_H: list[float] = []
    state = "L"
    t0 = 0.0
    events = sorted(transitions, key=lambda x: float(x[0]))

    for tv, _kind in events:
        t_ev = float(tv)
        if t_ev <= t0 or t_ev > tf + 1e-15:
            continue
        dt_seg = t_ev - t0
        if state == "L":
            dwell_L.append(dt_seg)
            state = "H"
        else:
            dwell_H.append(dt_seg)
            state = "L"
        t0 = t_ev

    tail = tf - t0
    if tail < 0.0:
        tail = 0.0
    if state == "L":
        dwell_L.append(tail)
    else:
        dwell_H.append(tail)

    return np.asarray(dwell_L, dtype=np.float64), np.asarray(dwell_H, dtype=np.float64)


def compute_temporal_spectrum(
    c_at_x0: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray | float]:
    """FFT of ``c(x=0, t)`` after removing mean and linear trend.

    Returns keys: ``freqs``, ``power``, ``peak_freq``, ``peak_period``.
    """
    y = np.asarray(c_at_x0, dtype=np.float64).ravel()
    if y.size < 8:
        return {
            "freqs": np.asarray([], dtype=np.float64),
            "power": np.asarray([], dtype=np.float64),
            "peak_freq": float("nan"),
            "peak_period": float("nan"),
        }
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    y_det = signal.detrend(y, type="linear")
    n = int(y_det.size)
    freqs = np.fft.rfftfreq(n, d=float(dt))
    spec = np.fft.rfft(y_det)
    power = (np.abs(spec) ** 2).astype(np.float64)
    if power.size <= 1:
        pk_f = float("nan")
    else:
        k = int(1 + np.argmax(power[1:]))
        pk_f = float(freqs[k])
    pk_period = float(1.0 / pk_f) if pk_f > 1e-18 else float("nan")

    return {
        "freqs": freqs.astype(np.float64),
        "power": power,
        "peak_freq": pk_f,
        "peak_period": pk_period,
    }


def compute_mean_dwell_period(
    dwell_L_samples: np.ndarray,
    dwell_H_samples: np.ndarray,
    n_transitions: int,
) -> dict[str, float]:
    """Physical oscillation period as mean dwell-time sum.

    More robust than FFT peak detection for high-frequency oscillations where the
    Nyquist limit is approached.

    Returns
    -------
    dict with keys:
        mean_dwell_period: float
            T_period ≈ mean(dwell_L) + mean(dwell_H). NaN if ``n_transitions < 2``.
        mean_dwell_freq: float
            1 / mean_dwell_period when period > 0, else NaN.
        dwell_L_mean, dwell_H_mean: float
            Echoed means for convenience (NaN if no samples).
    """
    if int(n_transitions) < 2:
        nan = float("nan")
        return {
            "mean_dwell_period": nan,
            "mean_dwell_freq": nan,
            "dwell_L_mean": nan,
            "dwell_H_mean": nan,
        }

    dl = np.asarray(dwell_L_samples, dtype=np.float64).ravel()
    dh = np.asarray(dwell_H_samples, dtype=np.float64).ravel()
    dl_mean = float(np.mean(dl)) if dl.size > 0 else float("nan")
    dh_mean = float(np.mean(dh)) if dh.size > 0 else float("nan")

    if np.isnan(dl_mean) or np.isnan(dh_mean):
        period = float("nan")
    else:
        period = dl_mean + dh_mean

    return {
        "mean_dwell_period": period,
        "mean_dwell_freq": (1.0 / period) if period > 0.0 and np.isfinite(period) else float("nan"),
        "dwell_L_mean": dl_mean,
        "dwell_H_mean": dh_mean,
    }
