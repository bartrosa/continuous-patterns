"""Analytical helpers for UW poster reference solutions (semi-infinite Green's function).

Formulas transcribed from ``poster2.pdf`` (UW agate poster, 2024). Used to validate
the Heaney ``slab_kinetic`` solver; see ``docs/TERMINOLOGY.md`` for model naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import special
from scipy.optimize import brentq

# Five (c1, c2) pairs on the poster (Numerical / Analytical sections).
POSTER_C_PAIRS: list[tuple[float, float]] = [
    (0.49, 0.10),
    (0.40, 0.26),
    (0.25, 0.10),
    (0.50, 0.15),
    (0.33, 0.20),
]


def poster_formula_notes() -> str:
    """Human-readable record of formulas read from the poster image."""
    return """\
Poster source: poster2.pdf (UW agate reference poster).

Green's function (semi-infinite, constant r; poster uses k for κ):
  G(x1,x2,t) = (Da/sqrt(4πt)) * [exp(-(x1-x2)²/(4t)) + exp(-(x1+x2)²/(4t))]
               - s * exp(s*(x1+x2+s*t)) * erfc((x1+x2+2*s*t)/sqrt(4t))
  with s = κ·Da (Robin parameter; poster labels it k·Da).

Transcendental dwell times (linear IC c(x,0)=x, equate c(0,T) to thresholds):
  (c2*(Da*κ+1) - 1) / (c1*(Da*κ+1) - 1) = exp(s_H²*T2) * erfc(s_H*sqrt(T2)),  s_H = κ*Da
  (c1*(Da+1) - 1) / (c2*(Da+1) - 1) = exp(s_L²*T1) * erfc(s_L*sqrt(T1)),  s_L = Da

T1 + T2 is compared to numerical mean_dwell_period when both roots exist.
"""


def _g_transcendent(T: float, s: float) -> float:
    """exp(s²T)·erfc(s√T) = erfcx(s√T) (stable for large T)."""
    if T <= 0.0:
        return 1.0
    st = float(s) * np.sqrt(float(T))
    return float(special.erfcx(st))


def _solve_transcendent(
    target: float, s: float, *, t_lo: float = 1e-10, t_hi: float = 1e4
) -> float | None:
    """Solve exp(s²T) erfc(s√T) = target for T > 0."""
    tgt = float(target)
    if not np.isfinite(tgt) or tgt <= 0.0 or tgt >= 1.0:
        return None

    def f(T: float) -> float:
        return _g_transcendent(T, s) - tgt

    try:
        if f(t_lo) * f(t_hi) > 0:
            # widen upper bracket
            hi = t_hi
            for _ in range(12):
                if f(hi) <= 0.0:
                    break
                hi *= 10.0
            else:
                return None
            t_hi = hi
        return float(brentq(f, t_lo, t_hi, maxiter=200))
    except (ValueError, RuntimeError):
        return None


def solve_T1_T2(Da: float, kappa: float, c1: float, c2: float) -> tuple[float | None, float | None]:
    """Return (T1, T2) from poster transcendental equations; None if infeasible."""
    da = float(Da)
    ka = float(kappa)
    fc1 = float(c1)
    fc2 = float(c2)

    num_h = fc2 * (da * ka + 1.0) - 1.0
    den_h = fc1 * (da * ka + 1.0) - 1.0
    if abs(den_h) < 1e-14:
        t2 = None
    else:
        target_h = num_h / den_h
        t2 = _solve_transcendent(target_h, ka * da)

    num_l = fc1 * (da + 1.0) - 1.0
    den_l = fc2 * (da + 1.0) - 1.0
    if abs(den_l) < 1e-14:
        t1 = None
    else:
        target_l = num_l / den_l
        t1 = _solve_transcendent(target_l, da)

    return t1, t2


def greens_function(x1: np.ndarray, x2: np.ndarray, t: float, Da: float, s: float) -> np.ndarray:
    """Poster Green's function G(x1, x2, t) for constant r (s = κ·Da)."""
    if t <= 0.0:
        return np.zeros_like(x1, dtype=np.float64)
    tt = float(t)
    pref = float(Da) / np.sqrt(4.0 * np.pi * tt)
    kern = np.exp(-((x1 - x2) ** 2) / (4.0 * tt)) + np.exp(-((x1 + x2) ** 2) / (4.0 * tt))
    arg = (x1 + x2 + 2.0 * float(s) * tt) / np.sqrt(4.0 * tt)
    robin = float(s) * np.exp(float(s) * (x1 + x2 + float(s) * tt)) * special.erfc(arg)
    return pref * kern - robin


def convolve_profile_at_x(
    x_query: float,
    x_grid: np.ndarray,
    c0: np.ndarray,
    t: float,
    Da: float,
    s: float,
) -> float:
    """c(x_query, t) = ∫ G(x_query, x', t) c0(x') dx' on the semi-infinite grid."""
    if t <= 0.0:
        return float(np.interp(x_query, x_grid, c0))
    g = greens_function(
        np.full_like(x_grid, float(x_query)),
        x_grid,
        t,
        Da,
        s,
    )
    return float(np.trapezoid(g * c0, x_grid))


def convolve_profile_all(
    x_grid: np.ndarray,
    c0: np.ndarray,
    t: float,
    Da: float,
    s: float,
) -> np.ndarray:
    """Vectorized c(x, t) for all grid points via Green's function."""
    if t <= 0.0:
        return np.asarray(c0, dtype=np.float64).copy()
    x1 = x_grid[:, np.newaxis]
    x2 = x_grid[np.newaxis, :]
    g = greens_function(x1, x2, t, Da, s)
    return np.trapezoid(g * c0[np.newaxis, :], x_grid, axis=1)


def simulate_semiinfinite_green(
    *,
    Da: float,
    kappa: float,
    c1: float,
    c2: float,
    T_final: float,
    dt_out: float = 0.002,
    x_max: float = 12.0,
    n_x: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hysteresis evolution on [0, ∞) via Green's function (poster F1 red curve)."""
    da = float(Da)
    ka = float(kappa)
    fc1 = float(c1)
    fc2 = float(c2)
    x_grid = np.linspace(0.0, float(x_max), int(n_x), dtype=np.float64)
    c_prof = np.asarray(x_grid, dtype=np.float64)  # c(x,0)=x
    state = "L"
    t_phase0 = 0.0
    t_global = 0.0

    t_out: list[float] = [0.0]
    c0_out: list[float] = [float(c_prof[0])]

    while t_global < float(T_final) - 1e-15:
        r_s = ka * da if state == "H" else da
        t_next = min(t_global + float(dt_out), float(T_final))
        tau = t_next - t_phase0
        if tau > 0.0:
            c_prof = convolve_profile_all(x_grid, c_prof, tau, da, r_s)
        t_global = t_next
        c0 = float(c_prof[0])
        t_out.append(t_global)
        c0_out.append(c0)

        if state == "L" and c0 > fc1:
            state = "H"
            t_phase0 = t_global
        elif state == "H" and c0 < fc2:
            state = "L"
            t_phase0 = t_global

    tt = np.asarray(t_out, dtype=np.float64)
    c0_hist = np.asarray(c0_out, dtype=np.float64)
    return tt, c0_hist, x_grid


def analytical_period_sum(Da: float, kappa: float, c1: float, c2: float) -> float | None:
    t1, t2 = solve_T1_T2(Da, kappa, c1, c2)
    if t1 is None or t2 is None:
        return None
    return float(t1) + float(t2)


@dataclass(frozen=True)
class AnalyticalGrid:
    Da: np.ndarray
    kappa: np.ndarray
    T1: np.ndarray  # shape (n_kappa, n_Da, n_pairs)
    T2: np.ndarray

    def period(self) -> np.ndarray:
        out = np.full_like(self.T1, np.nan, dtype=np.float64)
        ok = np.isfinite(self.T1) & np.isfinite(self.T2)
        out[ok] = self.T1[ok] + self.T2[ok]
        return out


def compute_T_grid(
    Da_vals: np.ndarray,
    kappa_vals: np.ndarray,
    pairs: list[tuple[float, float]],
) -> AnalyticalGrid:
    n_d = Da_vals.size
    n_k = kappa_vals.size
    n_p = len(pairs)
    t1 = np.full((n_p, n_k, n_d), np.nan, dtype=np.float64)
    t2 = np.full((n_p, n_k, n_d), np.nan, dtype=np.float64)
    for ip, (c1, c2) in enumerate(pairs):
        for ik, ka in enumerate(kappa_vals):
            for ida, da in enumerate(Da_vals):
                a, b = solve_T1_T2(float(da), float(ka), float(c1), float(c2))
                if a is not None:
                    t1[ip, ik, ida] = a
                if b is not None:
                    t2[ip, ik, ida] = b
    return AnalyticalGrid(Da=Da_vals, kappa=kappa_vals, T1=t1, T2=t2)


def grid_metadata() -> dict[str, Any]:
    return {
        "formula_notes": poster_formula_notes(),
        "Da_range": [0.5, 5.0],
        "kappa_range": [1.5, 10.0],
        "grid_size": 30,
        "pairs": list(POSTER_C_PAIRS),
    }
