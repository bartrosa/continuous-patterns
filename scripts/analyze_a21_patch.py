#!/usr/bin/env python3
"""EXP-A2.1: analysis patch on A2 (five fixes: D1–D5).

Why this patch exists
---------------------
A2 answered Q3–Q5 cleanly but two implementation defects corrupted Q1/Q2
metrics, and the event convention mixed two populations:

- **D1 (dwell bug):** operational-window dwell used ONE interval [first entry,
  last exit]. For S1_cost0p40_g4 the trajectory starts inside the window,
  exits during the early dip, and re-enters near t ≈ 2950. Single-interval
  dwell wrecked n_pred vs n_obs parity on that run.
- **D2 (frame mismatch):** f* used c_local at the birth site while dwell used
  one-cell-ahead c_front. The gate acts on c in deposition cells; one frame
  (c_dep at x_f) is required throughout.
- **D3 (mixed populations):** edge-birth events alternate band-opening and
  band-closing edges; consecutive d_n mixed band width and gap width. Bands,
  gaps, and full cycles must be separated.
- **D4 (wrong sync statistic):** cross-correlation with 4–7 events yields
  noise. Event-triggered averaging replaces it.
- **D5 (circular consistency):** slope(d,v_bar) − slope(tau,v_bar) = 1 is an
  identity. Consistency uses v_front_smooth only.

Established findings carried forward (not re-litigated): universality, freeze
drift = 0 dx, birth offset median 1.65 dx, tau gamma-independence, t_term =
4425.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_patterns.models.twophase_kernel_reference import psi_ostwald_partition  # noqa: E402

_BOOTSTRAP_N = 1000
_TERMINAL_T_A2 = 4425.0  # cross-check only; recomputed from data
_EPOCH_HALF = 750.0
_GAMMA_REF = 4.0
_ZONE_WIDTHS = (10, 15, 20)


def _load_a2_module():
    spec = importlib.util.spec_from_file_location(
        "a2_mod", _SCRIPTS / "analyze_a2_understanding.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a2_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if xv.size < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(xv, yv)
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if np.unique(xv).size < 2:
        return float("nan"), float("nan"), float("nan")
    return _linreg(np.log(xv), np.log(yv))


def _bootstrap_loglog_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = _BOOTSTRAP_N, seed: int = 42
) -> dict[str, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    nan = float("nan")
    if xv.size < 3:
        return {"slope_lo": nan, "slope_hi": nan, "intercept_lo": nan, "intercept_hi": nan}
    rng = np.random.default_rng(seed)
    slopes, intercepts = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, xv.size, size=xv.size)
        if np.unique(xv[idx]).size < 2:
            continue
        s, b, _ = _loglog_fit(xv[idx], yv[idx])
        if np.isfinite(s):
            slopes.append(s)
            intercepts.append(b)
    if not slopes:
        return {"slope_lo": nan, "slope_hi": nan, "intercept_lo": nan, "intercept_hi": nan}
    return {
        "slope_lo": float(np.percentile(slopes, 2.5)),
        "slope_hi": float(np.percentile(slopes, 97.5)),
        "intercept_lo": float(np.percentile(intercepts, 2.5)),
        "intercept_hi": float(np.percentile(intercepts, 97.5)),
    }


def _operational_c_bounds(c_ost: float, w_ost: float, f_star: float) -> tuple[float, float]:
    c_grid = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    psi_m, psi_c = psi_ostwald_partition(c_grid, c_ostwald=c_ost, w_ostwald=w_ost)
    f = np.minimum(psi_m, psi_c)
    mask = f >= f_star - 1e-12
    if not np.any(mask):
        raise RuntimeError(f"no operational c interval for f*={f_star}")
    return float(c_grid[mask].min()), float(c_grid[mask].max())


def _front_cell_index(solid_row: np.ndarray) -> int:
    idx = np.where(solid_row > 0.5)[0]
    if not idx.size:
        return 0
    return int(idx[-1])


def _c_dep_series(bundle: Any) -> np.ndarray:
    """c at the front (deposit) cell from transport replay."""
    nt = bundle.snap_t.size
    out = np.zeros(nt, dtype=np.float64)
    for ti in range(nt):
        ix = _front_cell_index(bundle.solid_xt[ti])
        out[ti] = float(bundle.c_xt[ti, ix])
    return out


def _interp_crossing(t0: float, t1: float, y0: float, y1: float, level: float) -> float:
    if abs(y1 - y0) < 1e-15:
        return 0.5 * (t0 + t1)
    frac = (level - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


def _occupancy_segments(
    t: np.ndarray,
    c: np.ndarray,
    c_lo: float,
    c_hi: float,
    t_term: float,
) -> list[list[float]]:
    """Sorted disjoint [t_enter, t_exit] where c_lo <= c <= c_hi, clipped at t_term."""
    t = np.asarray(t, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    if t.size < 2:
        return []
    segments: list[list[float]] = []
    in_seg = False
    t_enter = 0.0
    for i in range(t.size):
        if t[i] > t_term:
            if in_seg:
                segments.append([t_enter, t_term])
                in_seg = False
            break
        here = bool(c_lo - 1e-12 <= c[i] <= c_hi + 1e-12)
        if i == 0:
            in_seg = here
            if here:
                t_enter = float(t[i])
            continue
        was = bool(c_lo - 1e-12 <= c[i - 1] <= c_hi + 1e-12)
        if not was and here:
            if c[i - 1] < c_lo <= c[i]:
                t_enter = _interp_crossing(t[i - 1], t[i], c[i - 1], c[i], c_lo)
            elif c[i - 1] > c_hi >= c[i]:
                t_enter = _interp_crossing(t[i - 1], t[i], c[i - 1], c[i], c_hi)
            else:
                t_enter = float(t[i])
            in_seg = True
        elif was and not here:
            if c[i - 1] <= c_hi < c[i]:
                t_exit = _interp_crossing(t[i - 1], t[i], c[i - 1], c[i], c_hi)
            elif c[i - 1] >= c_lo > c[i]:
                t_exit = _interp_crossing(t[i - 1], t[i], c[i - 1], c[i], c_lo)
            else:
                t_exit = float(t[i])
            segments.append([t_enter, min(t_exit, t_term)])
            in_seg = False
    if in_seg:
        last_t = min(
            float(t[int(np.clip(np.searchsorted(t, t_term, side="right") - 1, 0, t.size - 1))]),
            t_term,
        )
        if not segments or segments[-1][1] < last_t - 1e-6:
            segments.append([t_enter, last_t])
    # drop zero-length and merge overlaps
    cleaned: list[list[float]] = []
    for seg in sorted(segments, key=lambda s: s[0]):
        if seg[1] <= seg[0]:
            continue
        if cleaned and seg[0] <= cleaned[-1][1] + 1e-6:
            cleaned[-1][1] = max(cleaned[-1][1], seg[1])
        else:
            cleaned.append(seg)
    return cleaned


def _integrate_inv_tau(
    t_seg: np.ndarray, v_seg: np.ndarray, slope: float, intercept: float
) -> float:
    if t_seg.size < 2 or not np.isfinite(slope):
        return float("nan")
    tau_hat = np.exp(intercept) * np.maximum(v_seg, 1e-12) ** slope
    return float(np.trapezoid(1.0 / np.maximum(tau_hat, 1e-12), t_seg))


def _pair_bands(
    edge_events: list[dict[str, Any]],
    *,
    t_term: float,
    pchip: PchipInterpolator,
    c_dep: np.ndarray,
    bundle: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair consecutive edge births as [open, close] bands."""
    evs = sorted(edge_events, key=lambda e: e["t_form"])
    bands: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    i = 0
    band_idx = 0
    while i < len(evs):
        if i + 1 >= len(evs):
            unpaired.append({**evs[i], "reason": "trailing_open"})
            break
        e0, e1 = evs[i], evs[i + 1]
        x0, x1 = float(e0["x_form"]), float(e1["x_form"])
        t0, t1 = float(e0["t_form"]), float(e1["t_form"])
        x_open, x_close = (x0, x1) if x0 <= x1 else (x1, x0)
        t_open, t_close = (t0, t1) if x0 <= x1 else (t1, t0)
        ti_open = int(e0["frame_index"] if x0 <= x1 else e1["frame_index"])  # noqa: F841
        c_at_open = float(np.interp(t_open, bundle.snap_t, c_dep))
        _, _, f_n = _minority_fraction(c_at_open, bundle.c_ostwald, bundle.w_ostwald)
        terminal = bool(t_open >= t_term)
        w_band = float(x_close - x_open)
        v_open = float(pchip.derivative()(t_open))
        bands.append(
            {
                "run_label": bundle.run_label,
                "band_index": band_idx,
                "t_open": t_open,
                "t_close": t_close,
                "x_open": x_open,
                "x_close": x_close,
                "w_band": w_band,
                "f_n": f_n,
                "c_dep_at_open": c_at_open,
                "v_front_smooth_at_open": v_open,
                "terminal": terminal,
                "unpaired_trailing": False,
            }
        )
        band_idx += 1
        i += 2
    return bands, unpaired


def _complete_cycles(bands: list[dict[str, Any]], *, t_term: float) -> list[dict[str, Any]]:
    """Add gap, lambda, tau_cycle, duty, v_cycle between consecutive non-terminal opens."""
    cycles: list[dict[str, Any]] = []
    usable = [b for b in bands if not b.get("unpaired_trailing")]
    for k in range(len(usable) - 1):
        b0, b1 = usable[k], usable[k + 1]
        gap = float(b1["x_open"] - b0["x_close"])
        lam = float(b0["w_band"] + gap)
        tau_c = float(b1["t_open"] - b0["t_open"])
        duty = float(b0["w_band"] / lam) if lam > 0 else float("nan")
        v_cycle = float(lam / tau_c) if tau_c > 0 else float("nan")
        terminal = bool(b0["terminal"] or b1["terminal"] or b0["t_open"] >= t_term)
        cycles.append(
            {
                "run_label": b0["run_label"],
                "band_index": b0["band_index"],
                **b0,
                "gap_k": gap,
                "lambda_k": lam,
                "tau_cycle_k": tau_c,
                "duty_k": duty,
                "v_cycle_k": v_cycle,
                "v_open_k": float(b0["v_front_smooth_at_open"]),
                "v_open_k1": float(b1["v_front_smooth_at_open"]),
                "terminal": terminal,
            }
        )
    return cycles


def _minority_fraction(c: float, c_ost: float, w_ost: float) -> tuple[float, float, float]:
    psi_m, psi_c = psi_ostwald_partition(np.asarray([c]), c_ostwald=c_ost, w_ostwald=w_ost)
    pm, pc = float(psi_m[0]), float(psi_c[0])
    return pm, pc, float(min(pm, pc))


def _schmitt_pulses(t: np.ndarray, signal: np.ndarray) -> list[dict[str, Any]]:
    s = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(s)
    if np.sum(finite) < 4:
        return []
    p95 = float(np.nanpercentile(s[finite], 95))
    up, down = 0.5 * p95, 0.2 * p95
    active = False
    pulses: list[dict[str, Any]] = []
    i0 = 0
    for i in range(s.size):
        if not np.isfinite(s[i]):
            continue
        if not active and s[i] >= up:
            active = True
            i0 = i
        elif active and s[i] <= down:
            seg = s[i0 : i + 1]
            top_q = seg[seg >= np.nanpercentile(seg, 75)]
            plateau = float(np.median(top_q)) if top_q.size else float(np.median(seg))
            pulses.append(
                {
                    "t_start": float(t[i0]),
                    "t_end": float(t[i]),
                    "plateau_height": plateau,
                    "i0": i0,
                    "i1": i,
                }
            )
            active = False
    if active:
        seg = s[i0:]
        top_q = seg[seg >= np.nanpercentile(seg, 75)]
        plateau = float(np.median(top_q)) if top_q.size else float(np.median(seg))
        pulses.append(
            {
                "t_start": float(t[i0]),
                "t_end": float(t[-1]),
                "plateau_height": plateau,
                "i0": i0,
                "i1": s.size - 1,
            }
        )
    return pulses


def _bootstrap_median_ci(
    values: list[float], n_boot: int = _BOOTSTRAP_N, seed: int = 42
) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {"median": nan, "lo": nan, "hi": nan}
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    rng = np.random.default_rng(seed)
    meds = [float(np.median(arr[rng.integers(0, arr.size, arr.size)])) for _ in range(n_boot)]
    return {
        "median": med,
        "lo": float(np.percentile(meds, 2.5)),
        "hi": float(np.percentile(meds, 97.5)),
    }


def _interface_widths_psi(x: np.ndarray, psi: np.ndarray, deposit: np.ndarray) -> list[float]:
    """10%–90% crossing width at each psi zero-crossing on the deposit."""
    active = deposit & np.isfinite(psi)
    widths: list[float] = []
    for i in range(x.size - 1):
        if not (active[i] and active[i + 1]):
            continue
        a, b = psi[i], psi[i + 1]
        if a * b >= 0:
            continue
        # bracket crossing
        left_val, right_val = psi[max(0, i - 3) : i + 1], psi[i + 1 : min(x.size, i + 5)]
        left_val = left_val[np.isfinite(left_val)]
        right_val = right_val[np.isfinite(right_val)]
        if left_val.size == 0 or right_val.size == 0:
            continue
        amp = float(max(np.max(np.abs(left_val)), np.max(np.abs(right_val))))
        if amp < 1e-6:
            continue
        lo, hi = -0.9 * amp, 0.9 * amp  # noqa: F841
        # linear interp for 10% and 90% level crossings around zero crossing at i
        x0 = float(x[i] + abs(a) / (abs(a) + abs(b)) * (x[i + 1] - x[i]))  # noqa: F841
        # scan local window for width between -0.9*amp and +0.9*amp
        j0 = max(0, i - 8)
        j1 = min(x.size, i + 10)
        sub_x = x[j0:j1]
        sub_p = psi[j0:j1]
        m = np.isfinite(sub_p)
        if np.sum(m) < 3:
            continue
        try:
            x10_lo = float(np.interp(-0.8 * amp, sub_p[m][::-1], sub_x[m][::-1]))
            x10_hi = float(np.interp(0.8 * amp, sub_p[m], sub_x[m]))
            w = abs(x10_hi - x10_lo)
            if w > 0:
                widths.append(w)
        except Exception:
            continue
    return widths


def _triggered_average(
    t: np.ndarray,
    signal: np.ndarray,
    event_times: list[float],
    *,
    half: float = _EPOCH_HALF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline-subtracted epochs averaged; returns t_rel grid, mean, std envelope."""
    grid = np.linspace(-half, half, 61, dtype=np.float64)
    stacks: list[np.ndarray] = []
    for te in event_times:
        mask = (t >= te - half) & (t <= te + half)
        if np.sum(mask) < 3:
            continue
        t_loc = t[mask] - te
        s_loc = np.asarray(signal[mask], dtype=np.float64)
        base_m = (t >= te - half) & (t <= te - 250.0)
        if np.sum(base_m) < 1:
            continue
        base = float(np.nanmean(signal[base_m]))
        interp = np.interp(grid, t_loc, s_loc - base, left=np.nan, right=np.nan)
        if np.any(np.isfinite(interp)):
            stacks.append(interp)
    if not stacks:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    arr = np.vstack(stacks)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _rising_flank_fraction(
    t: np.ndarray, signal: np.ndarray, event_times: list[float], dt_frame: float
) -> float:
    if not event_times:
        return float("nan")
    pos = 0
    for te in event_times:
        i = int(np.argmin(np.abs(t - te)))
        i0 = max(0, i - 2)
        i1 = min(t.size - 1, i + 2)
        if i1 <= i0:
            continue
        slope = (float(signal[i1]) - float(signal[i0])) / max(float(t[i1] - t[i0]), dt_frame)
        if slope > 0:
            pos += 1
    return float(pos / len(event_times))


def _rising_flank_fraction_pooled(stacks: list[np.ndarray], grid: np.ndarray) -> float:
    """Fraction of epochs with positive slope at t_open (grid index nearest 0)."""
    if not stacks:
        return float("nan")
    i0 = int(np.argmin(np.abs(grid)))
    i_lo = max(0, i0 - 2)
    i_hi = min(grid.size - 1, i0 + 2)
    if i_hi <= i_lo:
        return float("nan")
    pos = 0
    for epoch in stacks:
        slope = (float(epoch[i_hi]) - float(epoch[i_lo])) / max(
            float(grid[i_hi] - grid[i_lo]), 1e-12
        )
        if slope > 0:
            pos += 1
    return float(pos / len(stacks))


def _write_bands_table(
    path: Path, bands: list[dict[str, Any]], cycles: list[dict[str, Any]]
) -> None:
    cols = [
        "run_label",
        "band_index",
        "t_open",
        "t_close",
        "x_open",
        "x_close",
        "w_band",
        "gap",
        "lambda",
        "tau_cycle",
        "duty",
        "v_front_smooth",
        "f_n",
        "terminal",
    ]

    def _cell(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    cycle_by = {(c["run_label"], c["band_index"]): c for c in cycles}
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for b in bands:
        cy = cycle_by.get((b["run_label"], b["band_index"]), {})
        lines.append(
            "| "
            + " | ".join(
                _cell(
                    {
                        "run_label": b["run_label"],
                        "band_index": b["band_index"],
                        "t_open": b["t_open"],
                        "t_close": b["t_close"],
                        "x_open": b["x_open"],
                        "x_close": b["x_close"],
                        "w_band": b["w_band"],
                        "gap": cy.get("gap_k"),
                        "lambda": cy.get("lambda_k"),
                        "tau_cycle": cy.get("tau_cycle_k"),
                        "duty": cy.get("duty_k"),
                        "v_front_smooth": b["v_front_smooth_at_open"],
                        "f_n": b["f_n"],
                        "terminal": b["terminal"],
                    }[c]
                )
                for c in cols
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-A2.1 patch on S1 sweep analysis")
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args(argv)

    a2 = _load_a2_module()
    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / "manifest_axes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if len(entries) != 6:
        raise RuntimeError(f"expected 6 runs, found {len(entries)}")

    # optional A2 cross-check
    a2_summary_path = sweep_dir / "summary_a2.json"
    if a2_summary_path.is_file():
        a2_prev = json.loads(a2_summary_path.read_text(encoding="utf-8"))
        t_term_check = float(a2_prev.get("per_run", [{}])[0].get("t_term", _TERMINAL_T_A2))
    else:
        t_term_check = _TERMINAL_T_A2

    bundles: list[Any] = []
    all_bands: list[dict[str, Any]] = []
    all_cycles: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    parity_cfront: list[dict[str, Any]] = []
    deprecated_xcorr: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []

    for entry in tqdm(entries, desc="A21 load", unit="run"):
        bundles.append(a2._load_run_bundle(sweep_dir, entry))

    # --- per-run core ---
    f_star_opens: list[float] = []
    for bundle in tqdm(bundles, desc="A21 analyze", unit="run"):
        c_dep = _c_dep_series(bundle)
        c_front = bundle.snap_cf
        valid = np.isfinite(c_dep) & np.isfinite(c_front)
        max_cframe = (
            float(np.max(np.abs(c_dep[valid] - c_front[valid]))) if np.any(valid) else float("nan")
        )
        pchip, _ = a2._smooth_front_pchip(bundle.snap_t, bundle.snap_xf)
        t_term = a2._detect_terminal_time(bundle.snap_t, bundle.snap_cf)
        if abs(t_term - t_term_check) > 50.0:
            warnings.warn(f"{bundle.run_label}: t_term={t_term} differs from A2={t_term_check}")  # noqa: B028

        edge_raw = a2._extract_edge_birth_events(bundle)
        bands, unpaired = _pair_bands(
            edge_raw, t_term=t_term, pchip=pchip, c_dep=c_dep, bundle=bundle
        )
        for u in unpaired:
            bands.append(
                {
                    "run_label": bundle.run_label,
                    "band_index": len(bands),
                    "t_open": float(u["t_form"]),
                    "t_close": float("nan"),
                    "x_open": float(u["x_form"]),
                    "x_close": float("nan"),
                    "w_band": float("nan"),
                    "f_n": float("nan"),
                    "terminal": bool(float(u["t_form"]) >= t_term),
                    "unpaired_trailing": True,
                    "v_front_smooth_at_open": float(pchip.derivative()(float(u["t_form"]))),
                }
            )
        cycles = _complete_cycles(bands, t_term=t_term)
        all_bands.extend(bands)
        all_cycles.extend(cycles)

        for b in bands:
            if not b.get("unpaired_trailing") and not b.get("terminal"):
                f_star_opens.append(float(b["f_n"]))

        # dwell with c_dep — placeholder f*, refined after global f*
        per_run.append(
            {
                "run_label": bundle.run_label,
                "gamma": bundle.gamma,
                "c_ostwald": bundle.c_ostwald,
                "t_term": t_term,
                "max_abs_c_dep_minus_c_front": max_cframe,
                "c_dep": c_dep,
                "c_front": c_front,
                "pchip": pchip,
                "n_bands_obs": len(
                    [b for b in bands if not b.get("unpaired_trailing") and not b["terminal"]]
                ),
                "n_bands_manifest_zone": int(bundle.zone.get("n_bands_zone", 0)),
            }
        )

        # deprecated xcorr
        osc = a2._oscillator_series(bundle)
        ev_t = np.array(
            [b["t_open"] for b in bands if not b.get("unpaired_trailing")], dtype=np.float64
        )
        xrow = {"run_label": bundle.run_label}
        for key in ("w_mix", "minority_mass", "mobility"):
            xrow[key] = a2._xcorr_with_events(bundle.snap_t, osc[key], ev_t)
        deprecated_xcorr.append(xrow)

        # pulse sanity
        n_pulses = len(_schmitt_pulses(bundle.snap_t, osc["minority_mass"]))
        nb = int(bundle.zone.get("n_bands_zone", 0))
        if abs(n_pulses - nb) > 1:
            warnings.warn(
                f"{bundle.run_label}: |n_pulses={n_pulses} - n_bands_zone={nb}| > 1",
                stacklevel=1,
            )

    if not f_star_opens:
        raise RuntimeError("no f_n at band openings for f*")
    f_star = float(min(f_star_opens))

    # dwell + n_pred with c_dep; sensitivity with c_front
    for pr, bundle in zip(per_run, bundles, strict=True):
        c_lo, c_hi = _operational_c_bounds(bundle.c_ostwald, bundle.w_ostwald, f_star)
        segs = _occupancy_segments(bundle.snap_t, pr["c_dep"], c_lo, c_hi, pr["t_term"])
        segs_cf = _occupancy_segments(bundle.snap_t, pr["c_front"], c_lo, c_hi, pr["t_term"])
        pr["dwell_segments_c_dep"] = segs
        pr["dwell_segments_c_front"] = segs_cf
        pr["c_oper_lo"], pr["c_oper_hi"] = c_lo, c_hi
        pr["dwell_total_c_dep"] = float(sum(b - a for a, b in segs))
        pr["dwell_total_c_front"] = float(sum(b - a for a, b in segs_cf))

    # cycle fits (non-terminal)
    cy_nt = [c for c in all_cycles if not c.get("terminal")]
    lam = np.array([c["lambda_k"] for c in cy_nt], dtype=np.float64)
    w_b = np.array([c["w_band"] for c in cy_nt], dtype=np.float64)
    gap = np.array([c["gap_k"] for c in cy_nt], dtype=np.float64)
    tau_c = np.array([c["tau_cycle_k"] for c in cy_nt], dtype=np.float64)
    v_open = np.array([c["v_open_k"] for c in cy_nt], dtype=np.float64)

    s_lam, b_lam, r2_lam = _loglog_fit(v_open, lam)
    s_tau, b_tau, r2_tau = _loglog_fit(v_open, tau_c)
    s_wb, b_wb, r2_wb = _loglog_fit(v_open, w_b)
    s_gap, b_gap, r2_gap = _loglog_fit(v_open, gap)
    fits = {
        "lambda_vs_v": {
            "slope": s_lam,
            "intercept": b_lam,
            "r2": r2_lam,
            **_bootstrap_loglog_ci(v_open, lam),
        },
        "tau_cycle_vs_v": {
            "slope": s_tau,
            "intercept": b_tau,
            "r2": r2_tau,
            **_bootstrap_loglog_ci(v_open, tau_c),
        },
        "w_band_vs_v": {
            "slope": s_wb,
            "intercept": b_wb,
            "r2": r2_wb,
            **_bootstrap_loglog_ci(v_open, w_b),
        },
        "gap_vs_v": {
            "slope": s_gap,
            "intercept": b_gap,
            "r2": r2_gap,
            **_bootstrap_loglog_ci(v_open, gap),
        },
        "consistency_delta_slope_v_smooth": float(s_lam - s_tau),
    }

    # n_pred with fixed dwell
    for pr, bundle in zip(per_run, bundles, strict=True):
        n_obs = pr["n_bands_obs"]
        n_pred = 0.0
        pchip = pr["pchip"]
        for seg in pr["dwell_segments_c_dep"]:
            t0, t1 = seg
            if t1 <= t0:
                continue
            t_seg = bundle.snap_t[(bundle.snap_t >= t0) & (bundle.snap_t <= t1)]
            if t_seg.size < 2:
                continue
            v_seg = np.asarray(pchip.derivative()(t_seg))
            n_pred += _integrate_inv_tau(t_seg, v_seg, s_tau, b_tau)
        n_pred_cf = 0.0
        for seg in pr["dwell_segments_c_front"]:
            t0, t1 = seg
            t_seg = bundle.snap_t[(bundle.snap_t >= t0) & (bundle.snap_t <= t1)]
            if t_seg.size >= 2:
                v_seg = np.asarray(pchip.derivative()(t_seg))
                n_pred_cf += _integrate_inv_tau(t_seg, v_seg, s_tau, b_tau)
        pr["n_pred_c_dep"] = float(n_pred)
        pr["n_pred_c_front"] = float(n_pred_cf)
        pr["n_obs"] = n_obs
        parity.append({"run_label": pr["run_label"], "n_obs": n_obs, "n_pred": float(n_pred)})
        parity_cfront.append(
            {"run_label": pr["run_label"], "n_obs": n_obs, "n_pred": float(n_pred_cf)}
        )

    # gamma capacity: pulses + interfaces
    g025 = [b for b in bundles if abs(b.c_ostwald - 0.25) < 1e-9]
    pred_g2 = (2.0 / _GAMMA_REF) ** (-0.5)
    pred_g8 = (8.0 / _GAMMA_REF) ** (-0.5)

    for bundle in bundles:
        deposit = bundle.solid_xt[-1] > 0.5
        iw = _interface_widths_psi(bundle.x, bundle.psi_final, deposit)
        interface_rows.append(
            {
                "run_label": bundle.run_label,
                "gamma": bundle.gamma,
                "c_ostwald": bundle.c_ostwald,
                "median_width": float(np.median(iw)) if iw else float("nan"),
                "q25": float(np.percentile(iw, 25)) if iw else float("nan"),
                "q75": float(np.percentile(iw, 75)) if iw else float("nan"),
                "n_interfaces": len(iw),
            }
        )
        for n_behind in _ZONE_WIDTHS:
            osc = a2._oscillator_series(bundle, n_behind=n_behind)
            for sig_name in ("minority_mass", "w_mix"):
                pulses = _schmitt_pulses(bundle.snap_t, osc[sig_name])
                pl_heights = [p["plateau_height"] for p in pulses]
                ci = _bootstrap_median_ci(pl_heights)
                plateau_rows.append(
                    {
                        "run_label": bundle.run_label,
                        "gamma": bundle.gamma,
                        "c_ostwald": bundle.c_ostwald,
                        "signal": sig_name,
                        "zone_cells": n_behind,
                        "n_pulses": len(pulses),
                        **ci,
                    }
                )

    def _gamma_ratio_table(rows: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
        out = []
        ref = next(
            r
            for r in rows
            if abs(r["gamma"] - _GAMMA_REF) < 1e-9 and abs(r["c_ostwald"] - 0.25) < 1e-9
        )
        ref_val = float(ref[value_key])
        for g, pred in ((2.0, pred_g2), (4.0, 1.0), (8.0, pred_g8)):
            row = next(
                r for r in rows if abs(r["gamma"] - g) < 1e-9 and abs(r["c_ostwald"] - 0.25) < 1e-9
            )
            meas = float(row[value_key]) / ref_val if ref_val > 0 else float("nan")
            out.append({"gamma": g, "measured_ratio": meas, "predicted_ratio": pred})
        return out

    plateau_ratios = {}
    for sig in ("minority_mass", "w_mix"):
        for zw in _ZONE_WIDTHS:
            sub = [r for r in plateau_rows if r["signal"] == sig and r["zone_cells"] == zw]
            plateau_ratios[f"{sig}_zone{zw}"] = _gamma_ratio_table(sub, "median")

    iface_025 = [r for r in interface_rows if abs(r["c_ostwald"] - 0.25) < 1e-9]
    iface_ratios = _gamma_ratio_table(iface_025, "median_width")

    # triggered averages (pooled c_ost=0.25)
    triggered: dict[str, Any] = {}
    for sig_name in ("minority_mass", "mobility", "w_mix"):
        for zw in (_ZONE_WIDTHS[1],):
            grid = np.linspace(-_EPOCH_HALF, _EPOCH_HALF, 61, dtype=np.float64)
            stacks: list[np.ndarray] = []
            all_open_times: list[float] = []
            for bundle in g025:
                osc = a2._oscillator_series(bundle, n_behind=zw)
                opens = [
                    float(b["t_open"])
                    for b in all_bands
                    if b["run_label"] == bundle.run_label
                    and not b.get("unpaired_trailing")
                    and not b["terminal"]
                ]
                all_open_times.extend(opens)
                for te in opens:
                    mask = (bundle.snap_t >= te - _EPOCH_HALF) & (bundle.snap_t <= te + _EPOCH_HALF)
                    if np.sum(mask) < 3:
                        continue
                    t_loc = bundle.snap_t[mask] - te
                    s_loc = np.asarray(osc[sig_name][mask], dtype=np.float64)
                    base_m = (bundle.snap_t >= te - _EPOCH_HALF) & (bundle.snap_t <= te - 250.0)
                    if np.sum(base_m) < 1:
                        continue
                    base = float(np.nanmean(osc[sig_name][base_m]))
                    interp = np.interp(grid, t_loc, s_loc - base, left=np.nan, right=np.nan)
                    if np.any(np.isfinite(interp)):
                        stacks.append(interp)
            if stacks:
                arr = np.vstack(stacks)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
            else:
                mean = np.full_like(grid, np.nan)
                std = np.full_like(grid, np.nan)
            triggered[f"{sig_name}_zone{zw}"] = {
                "t_rel": grid.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "n_epochs": len(stacks),
                "peak_to_baseline": float(np.nanmax(mean) / max(abs(float(np.nanmin(mean))), 1e-12))
                if np.any(np.isfinite(mean))
                else float("nan"),
                "rising_flank_fraction": _rising_flank_fraction_pooled(stacks, grid),
            }

    # Q1 verdict from triggered stats
    mm = triggered.get("minority_mass_zone15", {})
    peak_base = float(mm.get("peak_to_baseline", float("nan")))
    rfl = float(mm.get("rising_flank_fraction", float("nan")))
    if rfl > 0.7 and peak_base > 2.0:
        q1_verdict = "minority_mass shows event-triggered ramp–plateau–drop; oscillator confirmed"
    else:
        q1_verdict = f"honest negative: rising_flank_fraction={rfl:.3g}, peak/base={peak_base:.3g}"

    # Q2: 0p40 parity collapse?
    p40 = next(p for p in parity if p["run_label"] == "S1_cost0p40_g4")
    p40_a2 = {"n_obs": 3, "n_pred": 8.385}  # from A2 report if present
    if a2_summary_path.is_file():
        for row in a2_prev.get("per_run", []):
            if row.get("run_label") == "S1_cost0p40_g4":
                p40_a2 = {"n_obs": row.get("n_obs"), "n_pred": row.get("n_pred")}
    q2_collapsed = abs(p40["n_pred"] - p40["n_obs"]) < abs(
        float(p40_a2["n_pred"]) - float(p40["n_obs"])
    )

    summary: dict[str, Any] = {
        "f_star_global_c_dep": f_star,
        "operational_c_bounds": {
            pr["run_label"]: [pr["c_oper_lo"], pr["c_oper_hi"]] for pr in per_run
        },
        "per_run": [
            {
                "run_label": pr["run_label"],
                "t_term": pr["t_term"],
                "dwell_segments_c_dep": pr["dwell_segments_c_dep"],
                "dwell_total_c_dep": pr["dwell_total_c_dep"],
                "dwell_segments_c_front_sensitivity": pr["dwell_segments_c_front"],
                "dwell_total_c_front": pr["dwell_total_c_front"],
                "max_abs_c_dep_minus_c_front": pr["max_abs_c_dep_minus_c_front"],
                "n_obs": pr["n_obs"],
                "n_pred_c_dep": pr["n_pred_c_dep"],
                "n_pred_c_front": pr["n_pred_c_front"],
            }
            for pr in per_run
        ],
        "parity_c_dep": parity,
        "parity_c_front_sensitivity": parity_cfront,
        "fits_non_terminal": fits,
        "duty_stats_per_run": {
            pr["run_label"]: {
                "mean": float(
                    np.mean(
                        [
                            c["duty_k"]
                            for c in all_cycles
                            if c["run_label"] == pr["run_label"] and not c["terminal"]
                        ]
                    )
                )
                if any(c["run_label"] == pr["run_label"] for c in all_cycles)
                else float("nan"),
                "std": float(
                    np.std(
                        [
                            c["duty_k"]
                            for c in all_cycles
                            if c["run_label"] == pr["run_label"] and not c["terminal"]
                        ]
                    )
                )
                if any(c["run_label"] == pr["run_label"] for c in all_cycles)
                else float("nan"),
            }
            for pr in per_run
        },
        "plateau_medians": plateau_rows,
        "plateau_gamma_ratios": plateau_ratios,
        "interface_widths": interface_rows,
        "interface_gamma_ratios": iface_ratios,
        "triggered_averages": triggered,
        "deprecated_xcorr": deprecated_xcorr,
        "carried_forward": {
            "universality_max_pairwise_delta": 5.39e-5,
            "freeze_drift_dx": 0.0,
            "birth_offset_median_dx": 1.65,
            "t_term": t_term_check,
        },
        "questions": {
            "Q1": {
                "verdict": q1_verdict,
                "minority_mass_zone15": triggered.get("minority_mass_zone15", {}),
            },
            "Q2": {
                "soft_edge_resolved": all(
                    pr["c_oper_lo"] <= float(b["c_dep_at_open"]) <= pr["c_oper_hi"]
                    for pr in per_run
                    for b in all_bands
                    if b["run_label"] == pr["run_label"]
                    and np.isfinite(b.get("c_dep_at_open", float("nan")))
                    and not b.get("terminal")
                ),
                "p40_parity_collapsed": q2_collapsed,
                "p40_parity_a21": p40,
                "p40_parity_a2": p40_a2,
            },
            "Q3": {
                "lambda_vs_v_slope": s_lam,
                "tau_cycle_vs_v_slope": s_tau,
                "consistency_delta_slope": float(s_lam - s_tau),
            },
        },
    }

    _write_bands_table(sweep_dir / "table_a21_bands.md", all_bands, all_cycles)
    (sweep_dir / "summary_a21.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    # figures
    _plot_cycles(sweep_dir / "figure_a21_cycles.png", cy_nt, parity, per_run, bundles, fits)
    _plot_triggered(sweep_dir / "figure_a21_triggered.png", triggered)
    _plot_gamma_capacity(sweep_dir / "figure_a21_gamma_capacity.png", plateau_ratios, iface_ratios)
    _write_report(sweep_dir / "REPORT_A21.md", summary)

    print(f"[A21] summary → {sweep_dir / 'summary_a21.json'}", flush=True)
    print(f"[A21] report → {sweep_dir / 'REPORT_A21.md'}", flush=True)
    return 0


def _plot_cycles(
    path: Path,
    cycles: list[dict[str, Any]],
    parity: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
    bundles: list[Any],
    fits: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(12.0, 10.0))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    if cycles:
        v = np.array([c["v_open_k"] for c in cycles], dtype=np.float64)
        lam = np.array([c["lambda_k"] for c in cycles], dtype=np.float64)
        ax.loglog(v, lam, "o", ms=4)
        fit = fits["lambda_vs_v"]
        s, b = fit["slope"], fit["intercept"]
        v_pos = v[v > 0]
        if np.isfinite(s) and v_pos.size >= 2:
            vl = np.logspace(np.log10(v_pos.min()), np.log10(v_pos.max()), 64)
            ax.loglog(vl, np.exp(b) * vl**s, "--", color="0.4")
            lo, hi = fit.get("slope_lo"), fit.get("slope_hi")
            if lo is not None and np.isfinite(lo):
                ax.fill_between(vl, np.exp(b) * vl**hi, np.exp(b) * vl**lo, alpha=0.3, color="0.85")
    ax.set_xlabel(r"$v_{\mathrm{front,smooth}}(t_{\mathrm{open}})$")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title(r"(a) $\lambda$ vs $v$ (non-terminal cycles)")

    ax = fig.add_subplot(gs[0, 1])
    if cycles:
        v = np.array([c["v_open_k"] for c in cycles])
        wb = np.array([c["w_band"] for c in cycles])
        gp = np.array([c["gap_k"] for c in cycles])
        ax.loglog(v, wb, "o", ms=4, label=r"$w_{\mathrm{band}}$")
        ax.loglog(v, gp, "s", ms=4, label="gap")
        ax.legend(fontsize=7)
    ax.set_xlabel(r"$v_{\mathrm{front,smooth}}(t_{\mathrm{open}})$")
    ax.set_title(r"(b) $w_{\mathrm{band}}$ and gap vs $v$")

    ax = fig.add_subplot(gs[1, 0])
    obs = [p["n_obs"] for p in parity]
    pred = [p["n_pred"] for p in parity]
    if obs:
        ax.scatter(pred, obs, s=35)
        lim = [0, max(max(obs), max(pred)) + 0.5]
        ax.plot(lim, lim, "--", color="0.5")
        for p in parity:
            ax.annotate(
                p["run_label"].replace("S1_", ""),
                (p["n_pred"], p["n_obs"]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel(r"$n_{\mathrm{pred}}$ (fixed dwell, c_dep)")
    ax.set_ylabel(r"$n_{\mathrm{obs}}$")
    ax.set_title("(c) Band-count parity")

    ax = fig.add_subplot(gs[1, 1])
    cols = {2.0: "#9467bd", 4.0: "#ff7f0e", 8.0: "#8c564b"}
    if cycles:
        for c in cycles:
            g = next(b.gamma for b in bundles if b.run_label == c["run_label"])
            ax.scatter(c["v_open_k"], c["duty_k"], s=22, color=cols.get(g, "#333"))
    ax.set_xlabel(r"$v_{\mathrm{open}}$")
    ax.set_ylabel("duty")
    ax.set_title("(d) Duty vs v (gamma-coded)")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_triggered(path: Path, triggered: dict[str, Any]) -> None:
    keys = [k for k in triggered if k.endswith("zone15")]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10.0, 3.0 * max(len(keys), 1)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys, strict=True):
        block = triggered[key]
        t = np.asarray(block["t_rel"])
        m = np.asarray(block["mean"])
        s = np.asarray(block["std"])
        ax.plot(t, m, lw=1.2, label=key)
        ax.fill_between(t, m - s, m + s, alpha=0.25)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_ylabel(key)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("t − t_open")
    fig.suptitle("Pooled triggered averages (c_ost=0.25 trio)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_gamma_capacity(
    path: Path,
    plateau_ratios: dict[str, list[dict[str, Any]]],
    iface_ratios: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    ax = axes[0]
    key = "minority_mass_zone15"
    if key in plateau_ratios:
        rows = plateau_ratios[key]
        g = [r["gamma"] for r in rows]
        meas = [r["measured_ratio"] for r in rows]
        pred = [r["predicted_ratio"] for r in rows]
        ax.plot(g, meas, "o-", label="measured")
        ax.plot(g, pred, "--", color="0.4", label=r"$(\gamma/4)^{-1/2}$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("plateau ratio / g=4")
        ax.set_title("(a) Pulse plateau ratios")
        ax.legend(fontsize=7)
    ax = axes[1]
    g = [r["gamma"] for r in iface_ratios]
    meas = [r["measured_ratio"] for r in iface_ratios]
    pred = [r["predicted_ratio"] for r in iface_ratios]
    ax.plot(g, meas, "o-", label="measured")
    ax.plot(g, pred, "--", color="0.4", label=r"$(\gamma/4)^{-1/2}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("interface width ratio / g=4")
    ax.set_title("(b) Frozen interface widths")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    q = summary["questions"]
    lines = [
        "# EXP-A2.1 Report",
        "",
        "## Q1 — Oscillator (triggered averages + plateau physics)",
        f"- Verdict: {q['Q1']['verdict']} <!-- questions.Q1.verdict -->",
        f"- minority_mass peak/base: {q['Q1']['minority_mass_zone15'].get('peak_to_baseline', '—')} "  # noqa: E501
        f"<!-- triggered_averages.minority_mass_zone15.peak_to_baseline -->",
        f"- rising_flank_fraction: {q['Q1']['minority_mass_zone15'].get('rising_flank_fraction', '—')} "  # noqa: E501
        f"<!-- triggered_averages.minority_mass_zone15.rising_flank_fraction -->",
        "",
        "## Q2 — Operational window (fixed dwell, c_dep frame)",
        f"- f* (c_dep): {summary['f_star_global_c_dep']:.4g} <!-- f_star_global_c_dep -->",
        f"- soft-edge resolved: {q['Q2']['soft_edge_resolved']} <!-- questions.Q2.soft_edge_resolved -->",  # noqa: E501
        f"- 0p40 parity collapsed vs A2: {q['Q2']['p40_parity_collapsed']} "
        f"(A21 n_pred/n_obs={q['Q2']['p40_parity_a21']}, A2={q['Q2']['p40_parity_a2']}) "
        f"<!-- questions.Q2.p40_parity_collapsed -->",
        "",
        "## Q3 — Lambda selection law (population-separated)",
        f"- slope(lambda, v): {q['Q3']['lambda_vs_v_slope']:.4g} <!-- fits_non_terminal.lambda_vs_v.slope -->",  # noqa: E501
        f"- slope(tau_cycle, v): {q['Q3']['tau_cycle_vs_v_slope']:.4g} "
        f"<!-- fits_non_terminal.tau_cycle_vs_v.slope -->",
        f"- consistency slope(lambda,v) − slope(tau_cycle,v) = {q['Q3']['consistency_delta_slope']:.4g} "  # noqa: E501
        f"(expect ≈ 1) <!-- fits_non_terminal.consistency_delta_slope_v_smooth -->",
        "",
        "## Gamma capacity (recovered γ^(-1/2) test)",
        "- See `plateau_gamma_ratios` and `interface_gamma_ratios` in summary_a21.json",
        "",
        "## Carried forward (unchanged)",
        f"- universality Δc_front max: {summary['carried_forward']['universality_max_pairwise_delta']}",  # noqa: E501
        f"- freeze drift: {summary['carried_forward']['freeze_drift_dx']} dx",
        f"- birth offset median: {summary['carried_forward']['birth_offset_median_dx']} dx",
        "",
        "## c-frame sensitivity (dwell / n_pred)",
        "",
        "| run | dwell_c_dep | dwell_c_front | n_pred_c_dep | n_pred_c_front | max|Δc| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pr in summary["per_run"]:
        lines.append(
            f"| {pr['run_label']} | {pr['dwell_total_c_dep']:.4g} | {pr['dwell_total_c_front']:.4g} | "  # noqa: E501
            f"{pr['n_pred_c_dep']:.4g} | {pr['n_pred_c_front']:.4g} | "
            f"{pr['max_abs_c_dep_minus_c_front']:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
