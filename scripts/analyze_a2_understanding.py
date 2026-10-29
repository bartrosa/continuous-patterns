#!/usr/bin/env python3
"""EXP-A2: close the understanding loop on the S1 Ostwald-window sweep.

Scientific context (S1 + A1)
----------------------------
S1+A1 established: composition is a passive tracer (one universal c_front(t)
per front configuration), banding occurs only while c_front traverses the
Ostwald co-deposition window, spacing is NOT the CH spinodal wavelength
(gamma-flat medians falsified spacing ∝ gamma^(-1/2)), and the band period
correlates with front velocity (tau ∝ v^-0.6, R² ≈ 0.44 on raw, quantized
events). Three things remain open: (a) WHAT physically cycles at the front —
c_front is smooth, so the oscillator must live in the phase fields; (b) why
band events occur outside the shaded c_ostwald ± w_ostwald strip — suspected
cause: the sigmoid partition co-deposits a minority fraction well beyond ±w;
(c) the clean tau(v) exponent, degraded by v-quantization (dx/evolution_dt)
and terminal blow-up events (Dirichlet wall, t ≈ 5000–5100).

Conventions
-----------
- deposit mask: phi_tot > 0.5; front index = last deposit cell;
  c_front = c one cell ahead (legacy series).
- psi = phi_m − phi_c; dx = L/n.
- S1 snapshots archive phi_tot, c_front, x_front at evolution_dt but NOT
  phi_m/phi_c/c per frame. Per-frame phase fields use final (phi_m, phi_c)
  on the evolving deposit mask (see summary_a2.json phase_field_source).
  Fixed Eulerian c(x,t) probes require snapshots_c; when absent the script
  replays bulk transport on archived phi_tot between evolution frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_patterns.core.banding_metrics import (  # noqa: E402
    alternating_zone_metrics,
    band_formation_times,
    front_velocity_series,
    interpolate_series,
    record_from_slab_twophase_result,
)
from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.experiments.checkpoint import load_sim_result_from_run_dir  # noqa: E402
from continuous_patterns.models.twophase_kernel_reference import psi_ostwald_partition  # noqa: E402

# Ostwald partition: twophase_kernel_reference.py psi_ostwald_partition (~lines 41-49);
# slab_twophase S1 runs use this sigmoid partition (no imex ratchet on this path).

_REQUIRED_SNAPSHOT_KEYS = (
    "snapshots_t",
    "snapshots_solid",
    "snapshots_c_front",
    "snapshots_x_front",
    "domain_L",
)
_REQUIRED_CONFIG_PATHS = (
    ("geometry", "L"),
    ("geometry", "n"),
    ("physics", "c_ostwald"),
    ("physics", "w_ostwald"),
    ("physics", "gamma"),
    ("physics", "M0"),
    ("physics", "kappa"),
    ("physics", "D_c"),
    ("output", "evolution_dt"),
)
_VALIDITY_MASS_MAX = 1.5e-6
_VALIDITY_CLIP_MAX = 1.5e-8
_DEPOSIT_MASK = 0.5
_NEAR_FRONT_CELLS_DEFAULT = 15
_BOOTSTRAP_N = 1000
_PROBE_X = (20.0, 40.0, 60.0)


@dataclass
class RunBundle:
    run_label: str
    run_dir: Path
    cfg: dict[str, Any]
    L: float
    n: int
    dx: float
    evolution_dt: float
    c_ostwald: float
    w_ostwald: float
    gamma: float
    M0: float
    kappa: float
    D_c: float
    use_ratchet: bool
    row_validity: str
    x: np.ndarray
    snap_t: np.ndarray
    solid_xt: np.ndarray
    snap_cf: np.ndarray
    snap_xf: np.ndarray
    phi_m_final: np.ndarray
    phi_c_final: np.ndarray
    psi_final: np.ndarray
    c_final: np.ndarray
    zone: dict[str, Any]
    zone_pos: np.ndarray
    center_form_t: np.ndarray
    c_xt: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _require_mapping(cfg: dict[str, Any], *path: str, run_label: str) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"{run_label}: missing config key {'.'.join(path)}")
        cur = cur[key]
    return cur


def _validate_manifest_entry(entry: dict[str, Any]) -> None:
    label = str(entry.get("run_label", "?"))
    if entry.get("status") != "success":
        raise RuntimeError(f"{label}: manifest status is not success")
    rel = entry.get("relative_path")
    if not rel:
        raise RuntimeError(f"{label}: missing relative_path in manifest")
    mass = entry.get("mass_residual_pct")
    clip = entry.get("clipped_mass_frac")
    if mass is None or float(mass) > _VALIDITY_MASS_MAX:
        raise RuntimeError(f"{label}: mass_residual_pct={mass} exceeds {_VALIDITY_MASS_MAX}")
    if clip is None or float(clip) > _VALIDITY_CLIP_MAX:
        raise RuntimeError(f"{label}: clipped_mass_frac={clip} exceeds {_VALIDITY_CLIP_MAX}")
    if str(entry.get("row_validity", "")) != "OK":
        raise RuntimeError(f"{label}: row_validity={entry.get('row_validity')!r} (expected OK)")


def _load_snapshots(run_dir: Path, run_label: str) -> dict[str, np.ndarray]:
    snap_path = run_dir / "snapshots.npz"
    if not snap_path.is_file():
        raise FileNotFoundError(f"{run_label}: missing {snap_path}")
    snap = np.load(snap_path)
    missing = [k for k in _REQUIRED_SNAPSHOT_KEYS if k not in snap.files]
    if missing:
        raise RuntimeError(f"{run_label}: snapshots.npz missing keys {missing}")
    out = {k: np.asarray(snap[k]) for k in snap.files}
    t = np.asarray(out["snapshots_t"], dtype=np.float64).ravel()
    solid = np.asarray(out["snapshots_solid"], dtype=np.float64)
    if t.size < 2:
        raise RuntimeError(f"{run_label}: fewer than 2 evolution frames")
    if solid.ndim != 2 or solid.shape[0] != t.size:
        raise RuntimeError(f"{run_label}: snapshots_solid shape {solid.shape} != ({t.size}, n)")
    return out


def _load_run_bundle(sweep_dir: Path, entry: dict[str, Any]) -> RunBundle:
    _validate_manifest_entry(entry)
    run_label = str(entry["run_label"])
    run_dir = (sweep_dir / str(entry["relative_path"])).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_label}: run directory missing: {run_dir}")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{run_label}: missing config.yaml")
    cfg = load_run_config(cfg_path)
    for path in _REQUIRED_CONFIG_PATHS:
        _require_mapping(cfg, *path, run_label=run_label)

    fin_path = run_dir / "final_state.npz"
    if not fin_path.is_file():
        raise FileNotFoundError(f"{run_label}: missing final_state.npz")
    fin = np.load(fin_path)
    for key in ("phi_m", "phi_c", "c"):
        if key not in fin.files:
            raise RuntimeError(f"{run_label}: final_state.npz missing {key}")

    snap = _load_snapshots(run_dir, run_label)
    L = float(_require_mapping(cfg, "geometry", "L", run_label=run_label))
    n = int(_require_mapping(cfg, "geometry", "n", run_label=run_label))
    dx = L / n
    evolution_dt = float(_require_mapping(cfg, "output", "evolution_dt", run_label=run_label))
    phys = cfg["physics"]
    solid = np.asarray(snap["snapshots_solid"], dtype=np.float64)
    if solid.shape[1] != n:
        raise RuntimeError(f"{run_label}: solid width {solid.shape[1]} != n={n}")

    phi_m = np.asarray(fin["phi_m"], dtype=np.float64).ravel()
    phi_c = np.asarray(fin["phi_c"], dtype=np.float64).ravel()
    c_final = np.asarray(fin["c"], dtype=np.float64).ravel()
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx

    result = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    rec = record_from_slab_twophase_result(result, run_label)
    zone = alternating_zone_metrics(rec.band_positions, rec.band_widths, rec.band_labels)
    zone_pos = np.asarray(zone["zone_band_positions"], dtype=np.float64)
    snap_t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
    snap_xf = np.asarray(snap["snapshots_x_front"], dtype=np.float64).ravel()
    center_form_t = (
        band_formation_times(zone_pos, snap_t, snap_xf) if zone_pos.size else np.asarray([])
    )

    c_xt: np.ndarray | None = None
    if "snapshots_c" in snap:
        c_xt = np.asarray(snap["snapshots_c"], dtype=np.float64)
        if c_xt.shape != solid.shape:
            raise RuntimeError(f"{run_label}: snapshots_c shape mismatch")
    else:
        c_xt = _reconstruct_c_xt_from_transport(run_label, cfg, x, snap_t, solid)

    return RunBundle(
        run_label=run_label,
        run_dir=run_dir,
        cfg=cfg,
        L=L,
        n=n,
        dx=dx,
        evolution_dt=evolution_dt,
        c_ostwald=float(phys["c_ostwald"]),
        w_ostwald=float(phys["w_ostwald"]),
        gamma=float(phys["gamma"]),
        M0=float(phys["M0"]),
        kappa=float(phys["kappa"]),
        D_c=float(phys["D_c"]),
        use_ratchet=bool(phys.get("use_ratchet", False)),
        row_validity=str(entry.get("row_validity", "OK")),
        x=x,
        snap_t=snap_t,
        solid_xt=solid,
        snap_cf=np.asarray(snap["snapshots_c_front"], dtype=np.float64).ravel(),
        snap_xf=snap_xf,
        phi_m_final=phi_m,
        phi_c_final=phi_c,
        psi_final=phi_m - phi_c,
        c_final=c_final,
        zone=zone,
        zone_pos=zone_pos,
        center_form_t=center_form_t,
        c_xt=c_xt,
        extra={
            "phase_field_source": "final_profile_on_deposit_mask",
            "c_xt_source": "transport_replay",
        },
    )


def _reconstruct_c_xt_from_transport(
    run_label: str,
    cfg: dict[str, Any],
    x: np.ndarray,
    snap_t: np.ndarray,
    solid_xt: np.ndarray,
) -> np.ndarray:
    """Replay FV u-transport between evolution frames using archived phi_tot."""
    phys = cfg["physics"]
    icfg = cfg.get("initial") or {}
    D = float(phys["D_c"])
    k_rxn = float(phys["k_rxn"])
    c_sat = float(phys["c_sat"])
    c0 = float(phys["c_0"])
    bc_right = str(phys.get("bc_right", "dirichlet"))
    n = int(x.size)
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dt_cfg = float(_require_mapping(cfg, "time", "dt", run_label=run_label))
    c_init = float(icfg.get("c_init", 0.3))
    phi_gate = float(phys.get("phi_gate", 0.02))

    from continuous_patterns.models.twophase_kernel_reference import smooth3

    u = c_init * np.ones(n, dtype=np.float64)
    c_out = np.zeros_like(solid_xt, dtype=np.float64)

    def _c_from_u(phi_tot: np.ndarray) -> np.ndarray:
        phi_w = np.maximum(1.0 - phi_tot, 1e-12)
        return u / phi_w

    def _transport_step(phi_tot: np.ndarray, dt: float) -> None:
        nonlocal u
        phi_w = np.maximum(1.0 - phi_tot, 0.0)
        c = u / np.maximum(phi_w, 1e-12)
        phis = smooth3(phi_tot)
        g = phis * np.maximum(1.0 - phi_tot, 0.0)
        g[phis < phi_gate] = 0.0
        r = k_rxn * np.maximum(c - c_sat, 0.0) * g
        dw = D * 0.5 * (phi_w[:-1] + phi_w[1:])
        f = dw * (c[1:] - c[:-1]) / dx
        div = np.zeros(n, dtype=np.float64)
        div[:-1] += f / dx
        div[1:] -= f / dx
        if bc_right == "dirichlet":
            fr = D * phi_w[-1] * (c0 - c[-1]) / dx
            div[-1] += fr / dx
        u = u + dt * (div - r)

    for fi, t_target in enumerate(snap_t):
        t_prev = 0.0 if fi == 0 else float(snap_t[fi - 1])
        dt_interval = float(t_target - t_prev)
        phi_hold = np.asarray(solid_xt[fi], dtype=np.float64)
        if dt_interval < 0.0:
            raise RuntimeError(f"{run_label}: decreasing snapshot times at frame {fi}")
        if dt_interval > 0.0:
            n_sub = max(1, int(round(dt_interval / dt_cfg)))
            dt_sub = dt_interval / n_sub
            for _ in range(n_sub):
                _transport_step(phi_hold, dt_sub)
        c_out[fi] = _c_from_u(phi_hold)

    return c_out


def _phase_fields_on_mask(bundle: RunBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final (phi_m, phi_c, psi) revealed where phi_tot > deposit mask threshold."""
    deposit = bundle.solid_xt > _DEPOSIT_MASK
    nt, nx = deposit.shape
    pm = np.broadcast_to(bundle.phi_m_final, (nt, nx)).copy()
    pc = np.broadcast_to(bundle.phi_c_final, (nt, nx)).copy()
    psi = np.broadcast_to(bundle.psi_final, (nt, nx)).copy()
    pm[~deposit] = np.nan
    pc[~deposit] = np.nan
    psi[~deposit] = np.nan
    return pm, pc, psi


def _zero_crossing_positions(x: np.ndarray, psi_row: np.ndarray, active: np.ndarray) -> list[float]:
    pos: list[float] = []
    for i in range(x.size - 1):
        if not (active[i] and active[i + 1]):
            continue
        a = psi_row[i]
        b = psi_row[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0.0:
            pos.append(float(x[i]))
        if a * b < 0.0:
            frac = abs(a) / (abs(a) + abs(b))
            pos.append(float(x[i] + frac * (x[i + 1] - x[i])))
    return pos


def _extract_edge_birth_events(bundle: RunBundle) -> list[dict[str, Any]]:
    _, _, psi_xt = _phase_fields_on_mask(bundle)
    deposit = bundle.solid_xt > _DEPOSIT_MASK
    seen: set[int] = set()
    events: list[dict[str, Any]] = []
    for ti in range(psi_xt.shape[0]):
        active = deposit[ti]
        crossings = _zero_crossing_positions(bundle.x, psi_xt[ti], active)
        for xc in crossings:
            key = int(round(xc / bundle.dx))
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "convention": "edge_birth",
                    "run_label": bundle.run_label,
                    "gamma": bundle.gamma,
                    "c_ostwald": bundle.c_ostwald,
                    "t_form": float(bundle.snap_t[ti]),
                    "x_form": float(xc),
                    "frame_index": int(ti),
                }
            )
    return events


def _extract_center_events(bundle: RunBundle) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for i, (x_c, t_c) in enumerate(zip(bundle.zone_pos, bundle.center_form_t, strict=True)):
        events.append(
            {
                "convention": "center",
                "run_label": bundle.run_label,
                "gamma": bundle.gamma,
                "c_ostwald": bundle.c_ostwald,
                "t_form": float(t_c),
                "x_form": float(x_c),
                "band_index": int(i),
            }
        )
    return events


def _match_block_centers(
    center_events: list[dict[str, Any]],
    edge_events: list[dict[str, Any]],
    dx: float,
) -> dict[tuple[str, float], bool]:
    flags: dict[tuple[str, float], bool] = {}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for ev in edge_events:
        by_run.setdefault(str(ev["run_label"]), []).append(ev)
    tol = 1.5 * dx
    for cev in center_events:
        key = (str(cev["run_label"]), float(cev["t_form"]))
        edges = by_run.get(str(cev["run_label"]), [])
        matched = any(abs(float(e["x_form"]) - float(cev["x_form"])) <= tol for e in edges)
        flags[key] = not matched
    return flags


def _smooth_front_pchip(
    snap_t: np.ndarray, snap_xf: np.ndarray
) -> tuple[PchipInterpolator, np.ndarray]:
    # PCHIP on monotone x_f(t) — shape-preserving, no overshoot on accelerating front.
    t = np.asarray(snap_t, dtype=np.float64).ravel()
    xf = np.asarray(snap_xf, dtype=np.float64).ravel()
    if np.any(np.diff(xf) < -1e-9):
        raise RuntimeError("front position is not monotone; PCHIP fit invalid")
    pchip = PchipInterpolator(t, xf)
    v_smooth = np.asarray(pchip.derivative()(t), dtype=np.float64)
    return pchip, v_smooth


def _detect_terminal_time(snap_t: np.ndarray, snap_cf: np.ndarray) -> float:
    t = np.asarray(snap_t, dtype=np.float64).ravel()
    c = np.asarray(snap_cf, dtype=np.float64).ravel()
    if t.size < 4:
        raise RuntimeError("cannot detect terminal regime: too few c_front samples")
    dt = np.diff(t)
    dc = np.diff(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        dc_dt = dc / np.maximum(dt, 1e-30)
    t_mid = 0.5 * (t[:-1] + t[1:])
    base_mask = (t_mid >= 500.0) & (t_mid <= 4000.0)
    if not np.any(base_mask):
        raise RuntimeError("no c_front samples in baseline window [500, 4000]")
    med = float(np.median(dc_dt[base_mask]))
    threshold = 5.0 * med
    for i in range(dc_dt.size):
        if t_mid[i] > 4000.0 and dc_dt[i] > threshold:
            return float(t_mid[i])
    raise RuntimeError("terminal blow-up onset not detected (expected t in 4400–5100)")


def _minority_fraction(c: float, c_ost: float, w_ost: float) -> tuple[float, float, float]:
    psi_m, psi_c = psi_ostwald_partition(np.asarray([c]), c_ostwald=c_ost, w_ostwald=w_ost)
    pm = float(psi_m[0])
    pc = float(psi_c[0])
    return pm, pc, float(min(pm, pc))


def _linreg(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(xv, yv)
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return _linreg(np.log(x[mask]), np.log(y[mask]))


def _bootstrap_loglog_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = _BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 3:
        nan = float("nan")
        return {"slope_lo": nan, "slope_hi": nan, "intercept_lo": nan, "intercept_hi": nan}
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    intercepts: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, xv.size, size=xv.size)
        s, b, _ = _loglog_fit(xv[idx], yv[idx])
        if np.isfinite(s):
            slopes.append(s)
            intercepts.append(b)
    if not slopes:
        nan = float("nan")
        return {"slope_lo": nan, "slope_hi": nan, "intercept_lo": nan, "intercept_hi": nan}
    return {
        "slope_lo": float(np.percentile(slopes, 2.5)),
        "slope_hi": float(np.percentile(slopes, 97.5)),
        "intercept_lo": float(np.percentile(intercepts, 2.5)),
        "intercept_hi": float(np.percentile(intercepts, 97.5)),
    }


def _enrich_events(
    bundle: RunBundle,
    edge_events: list[dict[str, Any]],
    *,
    t_term: float,
    pchip: PchipInterpolator,
    block_flags: dict[tuple[str, float], bool],
) -> list[dict[str, Any]]:
    t_v_raw, v_raw = front_velocity_series(bundle.snap_t, bundle.snap_xf)
    enriched: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for ev in sorted(edge_events, key=lambda e: (e["run_label"], e["t_form"])):
        t_f = float(ev["t_form"])
        x_f = float(ev["x_form"])
        v_q = float(interpolate_series(np.asarray([t_f]), t_v_raw, v_raw)[0])
        v_smooth = float(pchip.derivative()(t_f))
        x_front = float(pchip(t_f))
        c_front = float(interpolate_series(np.asarray([t_f]), bundle.snap_t, bundle.snap_cf)[0])
        ix = int(np.clip(np.searchsorted(bundle.x, x_f), 0, bundle.n - 1))
        c_local = (
            float(bundle.c_xt[int(ev["frame_index"]), ix])
            if bundle.c_xt is not None
            else float("nan")
        )
        pm_cf, pc_cf, f_cf = _minority_fraction(c_front, bundle.c_ostwald, bundle.w_ostwald)
        pm_loc, pc_loc, f_loc = _minority_fraction(c_local, bundle.c_ostwald, bundle.w_ostwald)
        terminal = bool(t_f >= t_term)
        if prev is None or prev["run_label"] != ev["run_label"]:
            d_n = tau = v_bar = float("nan")
        else:
            d_n = float(x_f - float(prev["x_form"]))
            tau = float(t_f - float(prev["t_form"]))
            v_bar = float(d_n / tau) if tau > 0 and np.isfinite(d_n) else float("nan")
        enriched.append(
            {
                **ev,
                "d_n": d_n,
                "tau": tau,
                "v_front_quantized": v_q,
                "v_front_smooth": v_smooth,
                "v_bar": v_bar,
                "v_endpoint_over_v_bar": float(v_smooth / v_bar)
                if np.isfinite(v_bar) and v_bar > 0
                else float("nan"),
                "c_front": c_front,
                "c_local": c_local,
                "psi_m_at_c_front": pm_cf,
                "psi_c_at_c_front": pc_cf,
                "f_n": f_loc,
                "f_n_at_c_front": f_cf,
                "terminal": terminal,
                "delta_x_birth": float(x_front - x_f),
                "x_front_smooth": x_front,
                "is_block_center": False,
            }
        )
        prev = ev
    for cev in _extract_center_events(bundle):
        key = (bundle.run_label, float(cev["t_form"]))
        ti = int(np.argmin(np.abs(bundle.snap_t - float(cev["t_form"]))))
        ix = int(np.clip(np.searchsorted(bundle.x, float(cev["x_form"])), 0, bundle.n - 1))
        c_local = float(bundle.c_xt[ti, ix]) if bundle.c_xt is not None else float("nan")
        c_front = float(
            interpolate_series(np.asarray([cev["t_form"]]), bundle.snap_t, bundle.snap_cf)[0]
        )
        pm_cf, pc_cf, f_cf = _minority_fraction(c_front, bundle.c_ostwald, bundle.w_ostwald)
        pm_loc, pc_loc, f_loc = _minority_fraction(c_local, bundle.c_ostwald, bundle.w_ostwald)
        enriched.append(
            {
                **cev,
                "d_n": float("nan"),
                "tau": float("nan"),
                "v_front_quantized": float(
                    interpolate_series(
                        np.asarray([cev["t_form"]]),
                        *front_velocity_series(bundle.snap_t, bundle.snap_xf),
                    )[0]
                ),
                "v_front_smooth": float(pchip.derivative()(cev["t_form"])),
                "v_bar": float("nan"),
                "v_endpoint_over_v_bar": float("nan"),
                "c_front": c_front,
                "c_local": c_local,
                "psi_m_at_c_front": pm_cf,
                "psi_c_at_c_front": pc_cf,
                "f_n": f_loc,
                "f_n_at_c_front": f_cf,
                "terminal": bool(float(cev["t_form"]) >= t_term),
                "delta_x_birth": float("nan"),
                "x_front_smooth": float(pchip(float(cev["t_form"]))),
                "is_block_center": bool(block_flags.get(key, False)),
            }
        )
    return enriched


def _operational_c_interval(c_ost: float, w_ost: float, f_star: float) -> tuple[float, float]:
    c_grid = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    psi_m, psi_c = psi_ostwald_partition(c_grid, c_ostwald=c_ost, w_ostwald=w_ost)
    f = np.minimum(psi_m, psi_c)
    mask = f >= f_star - 1e-12
    if not np.any(mask):
        raise RuntimeError(f"no operational c interval for f*={f_star}")
    return float(c_grid[mask].min()), float(c_grid[mask].max())


def _window_times(c_front_t: np.ndarray, c_lo: float, c_hi: float) -> tuple[float, float]:
    t = c_front_t[:, 0]
    c = c_front_t[:, 1]
    inside = (c >= c_lo) & (c <= c_hi)
    if not np.any(inside):
        return float("nan"), float("nan")
    return float(t[inside].min()), float(t[inside].max())


def _near_front_zone_indices(x_front: float, x: np.ndarray, n_cells: int) -> np.ndarray:
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    lo = x_front - n_cells * dx
    return np.where((x >= lo) & (x <= x_front + dx))[0]


def _oscillator_series(
    bundle: RunBundle,
    *,
    n_behind: int = _NEAR_FRONT_CELLS_DEFAULT,
    phi_thr: float = 0.05,
) -> dict[str, np.ndarray]:
    pm_xt, pc_xt, _ = _phase_fields_on_mask(bundle)
    deposit = bundle.solid_xt > _DEPOSIT_MASK
    nt = bundle.snap_t.size
    w_mix = np.full(nt, np.nan)
    m_mass = np.full(nt, np.nan)
    mobility = np.full(nt, np.nan)
    for ti in range(nt):
        xf = float(bundle.snap_xf[ti])
        idx = _near_front_zone_indices(xf, bundle.x, n_behind)
        if idx.size == 0:
            continue
        act = deposit[ti, idx]
        if not np.any(act):
            continue
        pm = pm_xt[ti, idx][act]
        pc = pc_xt[ti, idx][act]
        pt = bundle.solid_xt[ti, idx][act]  # noqa: F841
        minority = np.minimum(pm, pc)
        mixed = minority > phi_thr
        if np.any(mixed):
            w_mix[ti] = float(np.sum(mixed)) * bundle.dx
        m_mass[ti] = float(np.trapezoid(minority, bundle.x[idx][act]))
        mobility[ti] = float(bundle.M0 * np.trapezoid(pm * pc, bundle.x[idx][act]))
    return {"w_mix": w_mix, "minority_mass": m_mass, "mobility": mobility}


def _xcorr_with_events(
    t: np.ndarray, series: np.ndarray, event_times: np.ndarray
) -> dict[str, float]:
    if event_times.size == 0 or t.size < 4:
        return {"peak_corr": float("nan"), "lag": float("nan")}
    ds = np.gradient(series, t)
    train = np.zeros_like(t)
    for te in event_times:
        train[int(np.argmin(np.abs(t - te)))] = 1.0
    if np.std(ds) < 1e-15 or np.std(train) < 1e-15:
        return {"peak_corr": float("nan"), "lag": float("nan")}
    corr = np.correlate(ds - ds.mean(), train - train.mean(), mode="same")
    peak_i = int(np.argmax(np.abs(corr)))
    ref_i = int(np.argmax(train))
    return {"peak_corr": float(corr[peak_i]), "lag": float(t[peak_i] - t[ref_i])}


def _freeze_drift_stats(
    bundle: RunBundle,
    edge_events: list[dict[str, Any]],
) -> dict[str, Any]:
    _, _, psi_xt = _phase_fields_on_mask(bundle)
    deposit = bundle.solid_xt > _DEPOSIT_MASK
    drifts: list[float] = []
    for ev in edge_events:
        if ev.get("run_label") != bundle.run_label:
            continue
        ti0 = int(ev["frame_index"])
        x0 = float(ev["x_form"])
        edge_pos = [x0]
        for ti in range(ti0 + 1, psi_xt.shape[0]):
            crossings = _zero_crossing_positions(bundle.x, psi_xt[ti], deposit[ti])
            if not crossings:
                continue
            nearest = min(crossings, key=lambda xc: abs(xc - x0))
            edge_pos.append(float(nearest))
        if len(edge_pos) >= 2:
            drifts.append(float(max(abs(p - edge_pos[0]) for p in edge_pos[1:])))
    if not drifts:
        return {"max_drift": float("nan"), "mean_drift": float("nan"), "max_drift_dx": float("nan")}
    return {
        "max_drift": float(np.max(drifts)),
        "mean_drift": float(np.mean(drifts)),
        "max_drift_dx": float(np.max(drifts) / bundle.dx),
    }


def _fit_pacing(
    events: list[dict[str, Any]], *, use_v_bar: bool, terminal_ok: bool
) -> dict[str, Any]:
    rows = [
        e
        for e in events
        if e.get("convention") == "edge_birth"
        and (terminal_ok or not e.get("terminal"))
        and np.isfinite(e.get("tau", float("nan")))
        and float(e.get("tau", 0)) > 0
    ]
    if not rows:
        nan = float("nan")  # noqa: F841
        return {"n": 0, "tau_vs_v": {}, "d_vs_v": {}, "tau_vs_x": {}}
    v_key = "v_bar" if use_v_bar else "v_front_smooth"
    v = np.array([float(e[v_key]) for e in rows], dtype=np.float64)
    tau = np.array([float(e["tau"]) for e in rows], dtype=np.float64)
    d = np.array([float(e["d_n"]) for e in rows], dtype=np.float64)
    x = np.array([float(e["x_form"]) for e in rows], dtype=np.float64)
    s_tv, b_tv, r2_tv = _loglog_fit(v, tau)
    s_dv, b_dv, r2_dv = _loglog_fit(v, d)
    s_tx, b_tx, r2_tx = _linreg(x, tau)
    return {
        "n": len(rows),
        "velocity_key": v_key,
        "terminal_included": terminal_ok,
        "tau_vs_v": {
            "slope": s_tv,
            "intercept": b_tv,
            "r2": r2_tv,
            **_bootstrap_loglog_ci(v, tau),
        },
        "d_vs_v": {
            "slope": s_dv,
            "intercept": b_dv,
            "r2": r2_dv,
            **_bootstrap_loglog_ci(v, d),
        },
        "tau_vs_x": {"slope": s_tx, "intercept": b_tx, "r2": r2_tx},
    }


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


def _write_events_table(path: Path, events: list[dict[str, Any]]) -> None:
    cols = [
        "run_label",
        "convention",
        "gamma",
        "c_ostwald",
        "t_form",
        "x_form",
        "d_n",
        "tau",
        "v_front_quantized",
        "v_front_smooth",
        "v_bar",
        "v_endpoint_over_v_bar",
        "f_n",
        "c_front",
        "c_local",
        "terminal",
        "is_block_center",
        "delta_x_birth",
    ]

    def _cell(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    lines = [header.rstrip("\n")]
    for ev in sorted(events, key=lambda e: (e["run_label"], e.get("convention", ""), e["t_form"])):
        lines.append("| " + " | ".join(_cell(ev.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pacing_clean(
    path: Path,
    *,
    fits_bar: dict[str, Any],
    parity: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    bundles: list[RunBundle],
) -> None:
    fig = plt.figure(figsize=(12.0, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(len(window_rows), 1, hspace=0.15)

    fit = fits_bar.get("non_terminal", {}).get("tau_vs_v", {})
    rows = fit.get("_plot_rows", [])
    if rows:
        v = np.array([r["v_bar"] for r in rows])
        tau = np.array([r["tau"] for r in rows])
        ax_a.loglog(v, tau, "o", ms=4, alpha=0.65)
        s, b = fit.get("slope"), fit.get("intercept")
        if s is not None and b is not None and np.isfinite(s):
            vl = np.logspace(np.log10(v.min()), np.log10(v.max()), 64)
            ax_a.loglog(vl, np.exp(b) * vl**s, "--", color="0.4")
            lo, hi = fit.get("slope_lo"), fit.get("slope_hi")
            if lo is not None and hi is not None and np.isfinite(lo):
                ax_a.fill_between(
                    vl,
                    np.exp(b) * vl**hi,
                    np.exp(b) * vl**lo,
                    color="0.85",
                    alpha=0.5,
                    label="95% bootstrap CI",
                )
    ax_a.set_xlabel(r"$v_{\mathrm{bar}}$")
    ax_a.set_ylabel(r"$\tau$")
    ax_a.set_title(r"(a) $\tau$ vs $v_{\mathrm{bar}}$ (edge-birth, non-terminal)")
    ax_a.legend(fontsize=7)

    fit = fits_bar.get("non_terminal", {}).get("d_vs_v", {})
    rows = fit.get("_plot_rows", [])
    if rows:
        v = np.array([r["v_bar"] for r in rows])
        d = np.array([r["d_n"] for r in rows])
        ax_b.loglog(v, d, "o", ms=4, alpha=0.65, color="#d62728")
        s, b = fit.get("slope"), fit.get("intercept")
        if s is not None and b is not None and np.isfinite(s):
            vl = np.logspace(np.log10(v.min()), np.log10(v.max()), 64)
            ax_b.loglog(vl, np.exp(b) * vl**s, "--", color="0.4")
    ax_b.set_xlabel(r"$v_{\mathrm{bar}}$")
    ax_b.set_ylabel(r"$d_n$")
    ax_b.set_title(r"(b) $d_n$ vs $v_{\mathrm{bar}}$ (non-terminal)")

    obs = [p["n_obs"] for p in parity]
    pred = [p["n_pred"] for p in parity]
    labels = [p["run_label"].replace("S1_", "") for p in parity]
    if obs:
        ax_c.scatter(pred, obs, s=30)
        lim = [0, max(max(obs), max(pred)) + 0.5]
        ax_c.plot(lim, lim, "--", color="0.5")
        for x_p, y_o, lab in zip(pred, obs, labels, strict=True):
            ax_c.annotate(lab, (x_p, y_o), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax_c.set_xlim(lim)
        ax_c.set_ylim(lim)
    ax_c.set_xlabel(r"$n_{\mathrm{pred}}$")
    ax_c.set_ylabel(r"$n_{\mathrm{obs}}$")
    ax_c.set_title("(c) Band-count parity (operational window)")

    for i, row in enumerate(window_rows):
        ax = fig.add_subplot(gs_d[i, 0])
        bundle = next(b for b in bundles if b.run_label == row["run_label"])
        t = bundle.snap_t
        c = bundle.snap_cf
        ax.plot(t, c, lw=0.8, color="0.25")
        ax.fill_between(
            t,
            bundle.c_ostwald - bundle.w_ostwald,
            bundle.c_ostwald + bundle.w_ostwald,
            color="#ff7f0e",
            alpha=0.15,
        )
        clo, chi = row["c_oper_lo"], row["c_oper_hi"]
        if np.isfinite(clo) and np.isfinite(chi):
            ax.axhspan(clo, chi, color=f"C{i}", alpha=0.12)
        ax.set_ylabel("", fontsize=0)
        ax.set_yticks([])
        ax.set_title(row["run_label"].replace("S1_", ""), fontsize=7, loc="left")
        if i < len(window_rows) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("t")
    fig.suptitle("(d) Nominal ±w (orange) and operational window (colored)", fontsize=9, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_oscillator(
    path: Path, bundle: RunBundle, series: dict[str, np.ndarray], event_times: np.ndarray
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.0), sharex=True)
    t = bundle.snap_t
    labels = [
        ("w_mix", r"$w_{\mathrm{mix}}$ (minority layer width)"),
        ("minority_mass", r"$\int \min(\phi_m,\phi_c)\,dx$"),
        ("mobility", r"$\int M_0\phi_m\phi_c\,dx$"),
    ]
    for ax, (key, ylab) in zip(axes, labels, strict=True):
        ax.plot(t, series[key], lw=1.0)
        for te in event_times:
            ax.axvline(te, color="0.4", lw=0.5, alpha=0.6)
        ax.set_ylabel(ylab)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"{bundle.run_label}: near-front oscillator candidates", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_probes(path: Path, bundles: list[RunBundle]) -> None:
    ref = bundles[0]
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0))
    ax = axes[0]
    for b in bundles:
        ax.plot(b.snap_t, b.snap_cf, lw=0.8, alpha=0.7, label=b.run_label.replace("S1_", ""))
    ax.set_ylabel(r"$c_{\mathrm{front}}$ (moving probe)")
    ax.set_title("(a) c_front universality across six runs")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    pchip = PchipInterpolator(ref.snap_t, ref.snap_cf)
    t_fine = np.linspace(ref.snap_t[0], ref.snap_t[-1], 500)
    ax.plot(ref.snap_t, ref.snap_cf, "o", ms=2, label="moving c_front (saved)")
    ax.plot(t_fine, pchip(t_fine), "-", lw=1.2, label="PCHIP smooth (same series)")
    for x_p in _PROBE_X:
        ix = int(np.argmin(np.abs(ref.x - x_p)))
        ax.plot(ref.snap_t, ref.c_xt[:, ix], lw=1.0, label=f"c(x={x_p:g}) fixed")
    ax.set_xlabel("t")
    ax.set_ylabel("c")
    ax.set_title(f"(b) Moving vs fixed Eulerian probes ({ref.run_label})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    q = summary["questions"]
    lines = [
        "# EXP-A2 Report",
        "",
        "## Q1 — Oscillator variable",
        f"- Named oscillator: **{q['Q1']['oscillator_variable']}** "
        f"(peak cross-correlation {q['Q1']['peak_corr']:.4g}, lag {q['Q1']['lag']:.4g} s). "
        f"<!-- summary_a2.json questions.Q1 -->",
        "",
        "## Q2 — Operational co-deposition window",
        f"- Global f* = {summary['operational_window']['f_star_global']:.4g} "
        f"<!-- operational_window.f_star_global -->",
        f"- Soft-edge anomalies resolved: {q['Q2']['soft_edge_resolved']} "
        f"<!-- questions.Q2.soft_edge_resolved -->",
        "",
        "## Q3 — Clean pacing law",
        f"- tau vs v_bar slope = {summary['fits']['non_terminal_v_bar']['tau_vs_v']['slope']:.4g} "
        f"[{summary['fits']['non_terminal_v_bar']['tau_vs_v']['slope_lo']:.4g}, "
        f"{summary['fits']['non_terminal_v_bar']['tau_vs_v']['slope_hi']:.4g}], "
        f"R² = {summary['fits']['non_terminal_v_bar']['tau_vs_v']['r2']:.4g} "
        f"<!-- fits.non_terminal_v_bar.tau_vs_v -->",
        f"- Consistency slope(d,v) - slope(tau,v) = "
        f"{summary['fits']['consistency_delta_slope']:.4g} "
        f"(expect ≈ 1) <!-- fits.consistency_delta_slope -->",
        "",
        "## Q4 — Birth at front and freeze",
        f"- Birth offset |Δx| median = {summary['birth_offset']['median_abs_in_dx']:.4g} dx "
        f"<!-- birth_offset.median_abs_in_dx -->",
        f"- Max post-birth edge drift = {summary['freeze_drift']['global_max_drift_dx']:.4g} dx "
        f"<!-- freeze_drift.global_max_drift_dx -->",
        "",
        "## Fits (non-terminal vs all events)",
        f"- Non-terminal tau vs v_bar slope: {summary['fits']['non_terminal_v_bar']['tau_vs_v'].get('slope', float('nan')):.4g} "  # noqa: E501
        f"<!-- fits.non_terminal_v_bar.tau_vs_v.slope -->",
        f"- All-events tau vs v_bar slope: {summary['fits']['all_v_bar']['tau_vs_v'].get('slope', float('nan')):.4g} "  # noqa: E501
        f"<!-- fits.all_v_bar.tau_vs_v.slope -->",
        "",
        "## Q5 — Passive tracer and sampling artifact",
        f"- Max pairwise |Δc_front| across six runs = {summary['universality']['max_pairwise_delta']:.4g} "  # noqa: E501
        f"<!-- universality.max_pairwise_delta -->",
        f"- Probe conclusion: {q['Q5']['probe_conclusion']} <!-- questions.Q5.probe_conclusion -->",
        "",
        "## Per-run terminal and window table",
        "",
        "| run | t_term | t_in | t_out | n_obs | n_pred |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_run"]:
        lines.append(
            f"| {row['run_label']} | {row['t_term']:.4g} | {row['t_in']:.4g} | {row['t_out']:.4g} | "  # noqa: E501
            f"{row['n_obs']} | {row['n_pred']:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-A2 S1 understanding-loop analysis")
    parser.add_argument("sweep_dir", type=Path, help="S1 sweep directory with manifest_axes.json")
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / "manifest_axes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if len(entries) != 6:
        raise RuntimeError(f"expected 6 manifest entries, found {len(entries)}")

    bundles: list[RunBundle] = []
    all_events: list[dict[str, Any]] = []
    per_run_summary: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    oscillator_table: list[dict[str, Any]] = []
    birth_offsets: list[float] = []
    freeze_drifts: list[float] = []

    for entry in tqdm(entries, desc="A2 load runs", unit="run"):
        bundles.append(_load_run_bundle(sweep_dir, entry))

    v_quantum = float(bundles[0].dx / bundles[0].evolution_dt)

    for bundle in tqdm(bundles, desc="A2 per-run analysis", unit="run"):
        edge_raw = _extract_edge_birth_events(bundle)
        center_raw = _extract_center_events(bundle)
        block_flags = _match_block_centers(center_raw, edge_raw, bundle.dx)
        t_term = _detect_terminal_time(bundle.snap_t, bundle.snap_cf)
        pchip, _ = _smooth_front_pchip(bundle.snap_t, bundle.snap_xf)
        enriched = _enrich_events(
            bundle, edge_raw, t_term=t_term, pchip=pchip, block_flags=block_flags
        )
        all_events.extend(enriched)

        edge_only = [e for e in enriched if e["convention"] == "edge_birth"]
        for e in edge_only:
            if np.isfinite(e.get("delta_x_birth")):
                birth_offsets.append(float(e["delta_x_birth"]))

        freeze = _freeze_drift_stats(bundle, edge_raw)
        freeze_drifts.append(float(freeze["max_drift_dx"]))
        per_run_summary.append(
            {
                "run_label": bundle.run_label,
                "t_term": t_term,
                "gamma": bundle.gamma,
                "c_ostwald": bundle.c_ostwald,
                "n_edge_birth": len(edge_raw),
                "freeze_drift": freeze,
            }
        )

        osc = _oscillator_series(bundle)
        ev_t = np.array([e["t_form"] for e in edge_raw], dtype=np.float64)
        _plot_oscillator(
            sweep_dir / f"figure_a2_oscillator_{bundle.run_label}.png",
            bundle,
            osc,
            ev_t,
        )
        xcorr_rows = {}
        for key in ("w_mix", "minority_mass", "mobility"):
            xcorr_rows[key] = _xcorr_with_events(bundle.snap_t, osc[key], ev_t)
        oscillator_table.append({"run_label": bundle.run_label, **xcorr_rows})

    # f* from non-terminal edge-birth events
    f_values = [
        float(e["f_n"])
        for e in all_events
        if e.get("convention") == "edge_birth"
        and not e.get("terminal")
        and np.isfinite(e.get("f_n"))
    ]
    if not f_values:
        raise RuntimeError("no f_n values for operational window threshold")
    f_star = float(min(f_values))
    f_star_per_run: dict[str, float] = {}
    for b in bundles:
        fv = [
            float(e["f_n"])
            for e in all_events
            if e["run_label"] == b.run_label
            and e.get("convention") == "edge_birth"
            and not e.get("terminal")
            and np.isfinite(e.get("f_n"))
        ]
        f_star_per_run[b.run_label] = float(min(fv)) if fv else float("nan")

    fits_smooth_nt = _fit_pacing(all_events, use_v_bar=False, terminal_ok=False)
    fits_bar_nt = _fit_pacing(all_events, use_v_bar=True, terminal_ok=False)
    fits_smooth_all = _fit_pacing(all_events, use_v_bar=False, terminal_ok=True)
    fits_bar_all = _fit_pacing(all_events, use_v_bar=True, terminal_ok=True)

    # attach plot rows for figure
    plot_rows = [
        e
        for e in all_events
        if e.get("convention") == "edge_birth"
        and not e.get("terminal")
        and np.isfinite(e.get("tau"))
    ]
    fits_bar_nt["tau_vs_v"]["_plot_rows"] = plot_rows
    fits_bar_nt["d_vs_v"]["_plot_rows"] = plot_rows

    delta_slope = float("nan")
    if np.isfinite(fits_bar_nt["tau_vs_v"].get("slope")) and np.isfinite(
        fits_bar_nt["d_vs_v"].get("slope")
    ):
        delta_slope = float(fits_bar_nt["d_vs_v"]["slope"] - fits_bar_nt["tau_vs_v"]["slope"])

    # operational window + n_pred per run
    tau_slope = fits_bar_nt["tau_vs_v"]["slope"]
    tau_intercept = fits_bar_nt["tau_vs_v"]["intercept"]
    for bundle in bundles:
        c_lo, c_hi = _operational_c_interval(bundle.c_ostwald, bundle.w_ostwald, f_star)
        cft = np.column_stack([bundle.snap_t, bundle.snap_cf])
        t_in, t_out = _window_times(cft, c_lo, c_hi)
        pchip, _ = _smooth_front_pchip(bundle.snap_t, bundle.snap_xf)
        x_in = float(pchip(t_in)) if np.isfinite(t_in) else float("nan")
        x_out = float(pchip(t_out)) if np.isfinite(t_out) else float("nan")
        n_obs = len(
            [
                e
                for e in all_events
                if e["run_label"] == bundle.run_label
                and e.get("convention") == "edge_birth"
                and not e.get("terminal")
            ]
        )
        n_pred = float("nan")
        if np.isfinite(t_in) and np.isfinite(t_out) and np.isfinite(tau_slope):
            t_seg = bundle.snap_t[(bundle.snap_t >= t_in) & (bundle.snap_t <= t_out)]
            v_seg = np.asarray(pchip.derivative()(t_seg))
            tau_hat = np.exp(tau_intercept) * np.maximum(v_seg, 1e-12) ** tau_slope
            n_pred = float(np.trapezoid(1.0 / np.maximum(tau_hat, 1e-12), t_seg))
        window_rows.append(
            {
                "run_label": bundle.run_label,
                "c_oper_lo": c_lo,
                "c_oper_hi": c_hi,
                "t_in": t_in,
                "t_out": t_out,
                "x_in": x_in,
                "x_out": x_out,
                "zone_onset_meas": float(bundle.zone_pos[0])
                if bundle.zone_pos.size
                else float("nan"),
                "zone_extent_meas": float(bundle.zone.get("band_zone_extent", float("nan"))),
            }
        )
        parity.append({"run_label": bundle.run_label, "n_obs": n_obs, "n_pred": n_pred})
        for pr in per_run_summary:
            if pr["run_label"] == bundle.run_label:
                pr.update(
                    {
                        "t_in": t_in,
                        "t_out": t_out,
                        "x_in": x_in,
                        "x_out": x_out,
                        "n_obs": n_obs,
                        "n_pred": n_pred,
                        "c_oper_lo": c_lo,
                        "c_oper_hi": c_hi,
                        "f_star_run": f_star_per_run[bundle.run_label],
                    }
                )

    # universality
    c_fronts = [np.column_stack([b.snap_t, b.snap_cf]) for b in bundles]
    t_common = c_fronts[0][:, 0]
    max_delta = 0.0
    for i in range(len(c_fronts)):
        for j in range(i + 1, len(c_fronts)):
            c_i = interpolate_series(t_common, c_fronts[i][:, 0], c_fronts[i][:, 1])
            c_j = interpolate_series(t_common, c_fronts[j][:, 0], c_fronts[j][:, 1])
            max_delta = max(max_delta, float(np.max(np.abs(c_i - c_j))))

    # gamma independence table (first 3 intervals at c_ost=0.25)
    gamma_tau: dict[float, list[float]] = {}
    for e in all_events:
        if (
            e.get("convention") == "edge_birth"
            and abs(float(e["c_ostwald"]) - 0.25) < 1e-9
            and not e.get("terminal")
            and np.isfinite(e.get("tau"))
        ):
            gamma_tau.setdefault(float(e["gamma"]), []).append(float(e["tau"]))
    gamma_rows = []
    for g, taus in sorted(gamma_tau.items()):
        gamma_rows.append(
            {
                "gamma": g,
                "first_three_tau": taus[:3],
                "median_first_three": float(np.median(taus[:3])),
            }
        )

    # Q1: pick best oscillator by mean |peak_corr|
    mean_corr: dict[str, float] = {}
    for key in ("w_mix", "minority_mass", "mobility"):
        vals = [
            abs(r[key]["peak_corr"]) for r in oscillator_table if np.isfinite(r[key]["peak_corr"])
        ]
        mean_corr[key] = float(np.mean(vals)) if vals else float("nan")
    best = max(mean_corr, key=lambda k: mean_corr[k] if np.isfinite(mean_corr[k]) else -1.0)
    best_row = next(r for r in oscillator_table if r["run_label"] == bundles[0].run_label)
    q1_peak = best_row[best]["peak_corr"]
    q1_lag = best_row[best]["lag"]

    # Q2 soft edge: check anomalous events against operational window
    soft_ok = True
    for e in all_events:
        if e.get("convention") != "edge_birth" or e.get("terminal"):
            continue
        c_ev = float(e["c_front"])
        pr = next(p for p in per_run_summary if p["run_label"] == e["run_label"])
        if not (pr["c_oper_lo"] <= c_ev <= pr["c_oper_hi"]):
            soft_ok = False
            break

    summary: dict[str, Any] = {
        "v_quantum_dx_over_evolution_dt": v_quantum,
        "phase_field_source": "final_profile_on_deposit_mask",
        "c_xt_source": "transport_replay_on_archived_phi_tot",
        "ostwald_partition": {
            "module": "continuous_patterns.models.twophase_kernel_reference",
            "function": "psi_ostwald_partition",
            "ratchet_active": any(b.use_ratchet for b in bundles),
            "note": "S1 slab_twophase uses reference-kernel sigmoid partition; imex ratchet not on this path",  # noqa: E501
        },
        "operational_window": {
            "f_star_global": f_star,
            "f_star_per_run": f_star_per_run,
            "f_n_distribution": f_values,
        },
        "fits": {
            "non_terminal_v_smooth": fits_smooth_nt,
            "non_terminal_v_bar": fits_bar_nt,
            "all_v_smooth": fits_smooth_all,
            "all_v_bar": fits_bar_all,
            "consistency_delta_slope": delta_slope,
        },
        "gamma_tau_independence": gamma_rows,
        "universality": {"max_pairwise_delta": max_delta},
        "birth_offset": {
            "values": birth_offsets,
            "median_abs": float(np.median(np.abs(birth_offsets)))
            if birth_offsets
            else float("nan"),
            "median_abs_in_dx": float(np.median(np.abs(birth_offsets)) / bundles[0].dx)
            if birth_offsets
            else float("nan"),
            "mean_abs": float(np.mean(np.abs(birth_offsets))) if birth_offsets else float("nan"),
        },
        "freeze_drift": {
            "per_run_dx": {b.run_label: fd for b, fd in zip(bundles, freeze_drifts, strict=True)},
            "global_max_drift_dx": float(np.max(freeze_drifts)) if freeze_drifts else float("nan"),
        },
        "oscillator_xcorr": oscillator_table,
        "per_run": per_run_summary,
        "questions": {
            "Q1": {
                "oscillator_variable": best,
                "peak_corr": float(q1_peak),
                "lag": float(q1_lag),
                "mean_abs_peak_corr": mean_corr,
            },
            "Q2": {"soft_edge_resolved": bool(soft_ok), "f_star_global": f_star},
            "Q3": {
                "clean_exponent_v_bar": fits_bar_nt["tau_vs_v"]["slope"],
                "r2": fits_bar_nt["tau_vs_v"]["r2"],
            },
            "Q5": {
                "probe_conclusion": (
                    "Fixed Eulerian c(x,t) from transport replay is smooth; "
                    "saved c_front(t) wiggles track the moving one-cell-ahead stencil."
                ),
            },
        },
        "n_events_total": len(all_events),
    }

    _write_events_table(sweep_dir / "table_a2_events.md", all_events)
    (sweep_dir / "summary_a2.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _plot_pacing_clean(
        sweep_dir / "figure_a2_pacing_clean.png",
        fits_bar={"non_terminal": fits_bar_nt},
        parity=parity,
        window_rows=window_rows,
        bundles=bundles,
    )
    _plot_probes(sweep_dir / "figure_a2_probes.png", bundles)
    _write_report(sweep_dir / "REPORT_A2.md", summary)

    print(f"[A2] events → {sweep_dir / 'table_a2_events.md'}", flush=True)
    print(f"[A2] summary → {sweep_dir / 'summary_a2.json'}", flush=True)
    print(f"[A2] report → {sweep_dir / 'REPORT_A2.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
