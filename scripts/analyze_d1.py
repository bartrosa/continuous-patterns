#!/usr/bin/env python3
"""EXP-D1 discrimination analysis — shared band protocol, P1–P5 scoring, master figure."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.models.twophase_kernel_reference import psi_ostwald_partition  # noqa: E402

_KERNEL_CITE = "src/continuous_patterns/models/twophase_kernel_d1.py:_ch_exchange_step (verbatim twophase_kernel_reference.py:174-185)"  # noqa: E501
_F_STAR = 0.111
_VALIDITY_MASS_BC = 1e-4
_VALIDITY_CLIP_BC = 1e-6
_VALIDITY_PHI_A = 1e-6
_ANCHOR_LABEL = "D1A_v0p020_f30"
_ANCHOR_LAMBDA = (12.0, 22.0)
_DEPOSIT_MASK = 0.5


def _summary_float(summary: dict[str, Any], key: str, *, label: str) -> float:
    """Read a numeric diagnostic; fail loudly if missing/null (NaN sanitized to null in JSON)."""
    val = summary.get(key)
    if val is None:
        raise RuntimeError(
            f"{label}: summary.json[{key!r}] is null — simulation produced NaN/Inf; re-run required"
        )
    out = float(val)
    if not np.isfinite(out):
        raise RuntimeError(f"{label}: summary.json[{key!r}]={out!r} is not finite")
    return out


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    if np.unique(xv).size < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(np.log(xv), np.log(yv))
    return float(res.slope), float(res.intercept), float(res.rvalue**2)


def _bootstrap_loglog_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    nan = float("nan")
    if xv.size < 3:
        return {"slope_lo": nan, "slope_hi": nan}
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, xv.size, size=xv.size)
        if np.unique(xv[idx]).size < 2:
            continue
        s, _, _ = _loglog_fit(xv[idx], yv[idx])
        if np.isfinite(s):
            slopes.append(s)
    if not slopes:
        return {"slope_lo": nan, "slope_hi": nan}
    return {
        "slope_lo": float(np.percentile(slopes, 2.5)),
        "slope_hi": float(np.percentile(slopes, 97.5)),
    }


def _operational_c_bounds(c_ost: float, w_ost: float, f_star: float) -> tuple[float, float]:
    c_grid = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    psi_m, psi_c = psi_ostwald_partition(c_grid, c_ostwald=c_ost, w_ostwald=w_ost)
    f = np.minimum(psi_m, psi_c)
    mask = f >= f_star - 1e-12
    if not np.any(mask):
        raise RuntimeError(f"no operational c interval for f*={f_star}")
    return float(c_grid[mask].min()), float(c_grid[mask].max())


@dataclass
class D1Bundle:
    run_label: str
    block: str
    run_dir: Path
    cfg: dict[str, Any]
    L: float
    n: int
    dx: float
    x: np.ndarray
    evolution_dt: float
    snap_t: np.ndarray
    solid_xt: np.ndarray
    snap_xf: np.ndarray
    snap_c_dep: np.ndarray
    phi_m_final: np.ndarray
    phi_c_final: np.ndarray
    psi_final: np.ndarray
    diagnostics: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


def _block_from_label(label: str) -> str:
    if label.startswith("D1A_"):
        return "A"
    if label.startswith("D1B_"):
        return "B"
    if label.startswith("D1C_"):
        return "C"
    raise ValueError(label)


def _validate_entry(entry: dict[str, Any]) -> None:
    label = str(entry.get("run_label", "?"))
    if entry.get("status") != "success":
        raise RuntimeError(f"{label}: status={entry.get('status')!r}")
    if not entry.get("relative_path"):
        raise RuntimeError(f"{label}: missing relative_path")


def _load_snapshots(run_dir: Path, label: str) -> dict[str, np.ndarray]:
    path = run_dir / "snapshots.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing {path}")
    snap = np.load(path)
    required = ("snapshots_t", "snapshots_solid", "snapshots_x_front")
    missing = [k for k in required if k not in snap.files]
    if missing:
        raise RuntimeError(f"{label}: snapshots.npz missing {missing}")
    return {k: np.asarray(snap[k]) for k in snap.files}


def _load_d1_bundle(sweep_dir: Path, entry: dict[str, Any]) -> D1Bundle:
    _validate_entry(entry)
    label = str(entry["run_label"])
    block = _block_from_label(label)
    run_dir = (sweep_dir / str(entry["relative_path"])).resolve()
    cfg = load_run_config(run_dir / "config.yaml")
    fin = np.load(run_dir / "final_state.npz")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    snap = _load_snapshots(run_dir, label)

    L = float(cfg["geometry"]["L"])
    n = int(cfg["geometry"]["n"])
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    snap_t = np.asarray(snap["snapshots_t"], dtype=np.float64).ravel()
    solid = np.asarray(snap["snapshots_solid"], dtype=np.float64)
    snap_xf = np.asarray(snap["snapshots_x_front"], dtype=np.float64).ravel()
    if "snapshots_c_dep" in snap:
        snap_c_dep = np.asarray(snap["snapshots_c_dep"], dtype=np.float64).ravel()
    else:
        snap_c_dep = np.zeros_like(snap_t)

    phi_m = np.asarray(fin["phi_m"], dtype=np.float64).ravel()
    phi_c = np.asarray(fin["phi_c"], dtype=np.float64).ravel()

    if block == "A":
        raw = summary.get("phi_ledger_residual")
        if raw is None:
            mass = summary.get("mass_residual_pct")
            if mass is None:
                raise RuntimeError(f"{label}: missing phi_ledger_residual and mass_residual_pct")
            resid = float(mass) / 100.0
        else:
            resid = float(raw)
        if resid >= _VALIDITY_PHI_A:
            raise RuntimeError(f"{label}: phi_ledger_residual={resid:.3e} >= {_VALIDITY_PHI_A}")
    else:
        mass = _summary_float(summary, "mass_residual_pct", label=label)
        clip = _summary_float(summary, "clipped_mass_frac", label=label)
        if mass >= _VALIDITY_MASS_BC:
            raise RuntimeError(f"{label}: mass_residual_pct={mass:.3e} >= {_VALIDITY_MASS_BC}")
        if clip >= _VALIDITY_CLIP_BC:
            raise RuntimeError(f"{label}: clipped_mass_frac={clip:.3e} >= {_VALIDITY_CLIP_BC}")

    extra: dict[str, Any] = {}
    if "snapshots_S" in snap:
        extra["snapshots_S"] = [str(s) for s in snap["snapshots_S"].tolist()]

    return D1Bundle(
        run_label=label,
        block=block,
        run_dir=run_dir,
        cfg=cfg,
        L=L,
        n=n,
        dx=dx,
        x=x,
        evolution_dt=float(cfg.get("output", {}).get("evolution_dt", 50.0)),
        snap_t=snap_t,
        solid_xt=solid,
        snap_xf=snap_xf,
        snap_c_dep=snap_c_dep,
        phi_m_final=phi_m,
        phi_c_final=phi_c,
        psi_final=phi_m - phi_c,
        diagnostics=summary,
        extra=extra,
    )


def _phase_fields_on_mask(bundle: D1Bundle) -> np.ndarray:
    nt, _ = bundle.solid_xt.shape
    psi_xt = np.zeros_like(bundle.solid_xt)
    deposit_final = (bundle.phi_m_final + bundle.phi_c_final) > _DEPOSIT_MASK  # noqa: F841
    psi_final = bundle.psi_final
    for ti in range(nt):
        mask = bundle.solid_xt[ti] > _DEPOSIT_MASK
        psi_xt[ti, mask] = psi_final[mask]
        psi_xt[ti, ~mask] = np.nan
    return psi_xt


def _extract_edge_births(bundle: D1Bundle) -> list[dict[str, Any]]:
    a2 = _load_module("a2_mod", _SCRIPTS / "analyze_a2_understanding.py")
    psi_xt = _phase_fields_on_mask(bundle)
    deposit = bundle.solid_xt > _DEPOSIT_MASK
    seen: set[int] = set()
    events: list[dict[str, Any]] = []
    for ti in range(psi_xt.shape[0]):
        active = deposit[ti]
        crossings = a2._zero_crossing_positions(bundle.x, psi_xt[ti], active)
        xf = float(bundle.snap_xf[ti])
        for xc in crossings:
            key = int(round(xc / bundle.dx))
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "t_form": float(bundle.snap_t[ti]),
                    "x_form": float(xc),
                    "x_front": xf,
                    "frame_index": int(ti),
                    "birth_offset_dx": abs(float(xc - xf)) / bundle.dx,
                }
            )
    return events


def _pair_bands_a21(
    events: list[dict[str, Any]], bundle: D1Bundle, *, t_term: float
) -> tuple[list[dict], list[dict]]:
    a21 = _load_module("a21_mod", _SCRIPTS / "analyze_a21_patch.py")
    pchip = PchipInterpolator(bundle.snap_t, bundle.snap_xf)
    phys = bundle.cfg.get("physics", {})
    c_ost = float(phys.get("c_ostwald", 0.30))
    w_ost = float(phys.get("w_ostwald", 0.05))
    f_fixed = phys.get("f_chalcedony")
    v_prescribed = phys.get("v_front")

    class _Proxy:
        run_label = bundle.run_label
        snap_t = bundle.snap_t
        c_ostwald = c_ost
        w_ostwald = w_ost

    c_dep = bundle.snap_c_dep if np.any(bundle.snap_c_dep) else np.zeros_like(bundle.snap_t)
    edge_events = [
        {
            "t_form": e["t_form"],
            "x_form": e["x_form"],
            "frame_index": e["frame_index"],
            "run_label": bundle.run_label,
        }
        for e in events
    ]
    bands, _ = a21._pair_bands(
        edge_events, t_term=t_term, pchip=pchip, c_dep=c_dep, bundle=_Proxy()
    )
    for b in bands:
        if v_prescribed is not None:
            b["v_front_smooth_at_open"] = float(v_prescribed)
        if f_fixed is not None:
            b["f_n"] = float(f_fixed)
        b["birth_offset_dx"] = float(
            np.interp(
                b["t_open"], [e["t_form"] for e in events], [e["birth_offset_dx"] for e in events]
            )
            if events
            else float("nan")
        )
    cycles = a21._complete_cycles(bands, t_term=t_term)
    return bands, cycles


def _parse_a21_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| S1_"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 14:
            continue
        lam = parts[8]
        tau = parts[9]
        v = parts[11]
        if lam in ("—", "None", "") or tau in ("—", "None", "") or v in ("—", "None", ""):
            continue
        try:
            lam_f, tau_f, v_f = float(lam), float(tau), float(v)
        except ValueError:
            continue
        rows.append(
            {
                "run_label": parts[0],
                "source": "S1_A21",
                "lambda": lam_f,
                "tau_cycle": tau_f,
                "v": v_f,
                "block": "S1",
            }
        )
    return rows


def _score_p1(cycles_a: list[dict[str, Any]]) -> dict[str, Any]:
    f30 = [c for c in cycles_a if "f30" in c["run_label"] and not c.get("terminal")]
    v = np.asarray([c["v_open_k"] for c in f30], dtype=np.float64)
    lam = np.asarray([c["lambda_k"] for c in f30], dtype=np.float64)
    slope, intercept, r2 = _loglog_fit(v, lam)
    ci = _bootstrap_loglog_ci(v, lam)
    candidates = {"const": 0.0, "v_mhalf": -0.5, "v_inv": -1.0}
    errs = {name: abs(slope - exp) for name, exp in candidates.items()}
    best = min(errs, key=errs.get)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "slope_ci_lo": ci["slope_lo"],
        "slope_ci_hi": ci["slope_hi"],
        "best_law": best,
        "candidate_errors": errs,
        "n_points": int(v.size),
    }


def _score_p2(all_cycles_a: list[dict[str, Any]]) -> dict[str, Any]:
    offsets = [float(c.get("birth_offset_dx", float("nan"))) for c in all_cycles_a]
    offsets = [o for o in offsets if np.isfinite(o)]
    med = float(np.median(offsets)) if offsets else float("nan")
    mx = float(np.max(offsets)) if offsets else float("nan")
    return {"median_dx": med, "max_dx": mx, "n": len(offsets), "threshold_dx": 3.0}


def _score_p3(cycles_a: list[dict[str, Any]]) -> dict[str, Any]:
    by_f: dict[float, list[float]] = {}
    collapse_x, collapse_y = [], []
    for c in cycles_a:
        label = c["run_label"]
        m_f = re.search(r"_f(\d+)", label)
        m_m0 = re.search(r"M0p(\d+)", label)
        f = float(m_f.group(1)) / 100.0 if m_f else 0.30
        m0 = float(m_m0.group(1)) if m_m0 else 1.0
        by_f.setdefault(f, []).append(float(c["lambda_k"]))
        v = float(c["v_open_k"])
        if v > 0:
            collapse_x.append(m0 / v)
            collapse_y.append(float(c["lambda_k"]))
    f_slopes = {}
    for f, lams in by_f.items():
        f_slopes[str(f)] = {"median_lambda": float(np.median(lams)), "n": len(lams)}
    cs, ci, cr2 = _loglog_fit(np.asarray(collapse_x), np.asarray(collapse_y))
    return {"lambda_by_f": f_slopes, "collapse_slope": cs, "collapse_r2": cr2}


def _score_p4(bundle_b: D1Bundle, cycles: list[dict[str, Any]]) -> dict[str, Any]:
    phys = bundle_b.cfg["physics"]
    c1, c2 = float(phys["c1"]), float(phys["c2"])
    t = bundle_b.snap_t  # noqa: F841
    cdep = bundle_b.snap_c_dep
    gap_comp = c1 - c2
    gap_phys = bundle_b.L - bundle_b.snap_xf  # noqa: F841
    n_osc = len(cycles)
    if n_osc < 2:
        return {
            "n_cycles": n_osc,
            "oscillates": False,
            "c_dep_min": float(np.min(cdep)) if cdep.size else float("nan"),
            "c_dep_max": float(np.max(cdep)) if cdep.size else float("nan"),
            "gap_comp": gap_comp,
        }
    tau = np.asarray([c["tau_cycle_k"] for c in cycles], dtype=np.float64)
    gaps = np.asarray([c["gap_k"] for c in cycles], dtype=np.float64)
    v = np.asarray([c["v_open_k"] for c in cycles], dtype=np.float64)
    slope_tau_v, _, _ = _loglog_fit(v, tau)
    slope_gap_v, _, _ = _loglog_fit(v, gaps)
    return {
        "n_cycles": n_osc,
        "oscillates": True,
        "tau_vs_v_slope": slope_tau_v,
        "gap_vs_v_slope": slope_gap_v,
        "gap_comp": gap_comp,
        "clock_decoupled": abs(slope_tau_v) < 0.3,
        "clock_slaved": slope_tau_v < -0.7,
    }


def _score_p5(bundle: D1Bundle) -> dict[str, Any]:
    c_ost = float(bundle.cfg["physics"]["c_ostwald"])
    w_ost = float(bundle.cfg["physics"]["w_ostwald"])
    c_lo, c_hi = _operational_c_bounds(c_ost, w_ost, _F_STAR)
    psi = bundle.psi_final
    phi_tot = bundle.phi_m_final + bundle.phi_c_final
    deposit = phi_tot > _DEPOSIT_MASK
    x = bundle.x
    # classify thirds by psi sign dominance
    zones: list[dict[str, Any]] = []
    if np.any(deposit):
        xd = x[deposit]
        psid = psi[deposit]
        rind = xd < np.percentile(xd, 20)
        core = xd > np.percentile(xd, 80)
        mid = ~(rind | core)
        zones = [
            {
                "name": "rind",
                "median_psi": float(np.median(psid[rind])) if np.any(rind) else float("nan"),
            },
            {
                "name": "zone",
                "median_psi": float(np.median(psid[mid])) if np.any(mid) else float("nan"),
            },
            {
                "name": "core",
                "median_psi": float(np.median(psid[core])) if np.any(core) else float("nan"),
            },
        ]
    cdep = bundle.snap_c_dep
    crossings = []
    for level, name in ((c_lo, "window_lo"), (c_hi, "window_hi")):
        for i in range(1, cdep.size):
            if (cdep[i - 1] - level) * (cdep[i] - level) < 0:
                crossings.append({"level": name, "t": float(bundle.snap_t[i])})
    x_front_final = float(bundle.snap_xf[-1]) if bundle.snap_xf.size else 0.0
    return {
        "operational_c_lo": c_lo,
        "operational_c_hi": c_hi,
        "f_star": _F_STAR,
        "zones": zones,
        "c_dep_crossings": crossings,
        "x_front_final": x_front_final,
        "front_arrested_early": x_front_final < 0.35 * bundle.L,
    }


def _write_bands_table(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "run_label",
        "block",
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
        "v_at_open",
        "f_n",
        "birth_offset_dx",
        "terminal",
    ]

    def cell(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return "—" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_master(
    path: Path,
    *,
    cycles_a: list[dict],
    cycles_b: list[dict],
    cycles_c: list[dict],
    s1_pts: list[dict],
    e1e3_pts: list[dict],
) -> None:
    fig, (ax_lam, ax_tau) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for ax in (ax_lam, ax_tau):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("v")
        for exp, lbl in ((0, "slope 0"), (-0.5, "slope -1/2"), (-1, "slope -1")):
            xv = np.logspace(-3, 0, 50)
            ax.plot(
                xv, 10 * xv**exp, "k--", alpha=0.25, lw=0.8, label=lbl if ax is ax_lam else None
            )

    def _scatter_cycles(ax, cycles, **kw):
        if not cycles:
            return
        v = np.asarray(
            [c["v_open_k"] for c in cycles if np.isfinite(c.get("v_open_k", float("nan")))]
        )
        if ax is ax_lam:
            y = np.asarray([c["lambda_k"] for c in cycles[: v.size]])
        else:
            y = np.asarray([c["tau_cycle_k"] for c in cycles[: v.size]])
        m = (v > 0) & (y > 0)
        ax.scatter(v[m], y[m], **kw)

    # Block A: color by f, marker by M0
    for c in cycles_a:
        f = float(c.get("f_n", 0.3))
        m0 = 1.0
        if "M0p5" in c["run_label"]:
            m0 = 0.5
        elif "M2" in c["run_label"]:
            m0 = 2.0
        mk = "s" if m0 < 1 else "o" if m0 == 1 else "^"
        col = {0.15: "C0", 0.30: "C1", 0.50: "C2"}.get(round(f, 2), "C3")
        for ax, key in ((ax_lam, "lambda_k"), (ax_tau, "tau_cycle_k")):
            val = c.get(key)
            v = c.get("v_open_k")
            if val and v and val > 0 and v > 0:
                ax.scatter([v], [val], c=col, marker=mk, s=36, alpha=0.85)

    _scatter_cycles(ax_lam, cycles_b, marker="D", c="C4", label="Block B")
    _scatter_cycles(ax_tau, cycles_b, marker="D", c="C4")
    _scatter_cycles(ax_lam, cycles_c, marker="P", c="C5", label="Block C")
    _scatter_cycles(ax_tau, cycles_c, marker="P", c="C5")

    for pt in s1_pts:
        ax_lam.scatter(
            pt["v"], pt["lambda"], facecolors="none", edgecolors="C6", s=40, label="S1 (A2.1)"
        )
        if pt.get("tau_cycle"):
            ax_tau.scatter(pt["v"], pt["tau_cycle"], facecolors="none", edgecolors="C6", s=40)

    for pt in e1e3_pts:
        ax_lam.scatter(pt["v"], pt["lambda"], marker="X", c="C7", s=50, label="E1/E3 (fixed-gap)")
        if pt.get("tau_cycle"):
            ax_tau.scatter(pt["v"], pt["tau_cycle"], marker="X", c="C7", s=50)

    ax_lam.set_ylabel("lambda")
    ax_tau.set_ylabel("tau_cycle")
    ax_lam.legend(fontsize=7, loc="best")
    ax_lam.set_title("D1 master: lambda(v) and tau(v)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_block_b(path: Path, bundle: D1Bundle) -> None:
    phys = bundle.cfg["physics"]
    c1, c2 = float(phys["c1"]), float(phys["c2"])
    t = bundle.snap_t
    cdep = bundle.snap_c_dep
    fig, (ax_c, ax_s) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_c.plot(t, cdep, lw=0.9, label="c_dep(t)")
    ax_c.axhline(c1, color="C1", ls="--", label=f"c1={c1}")
    ax_c.axhline(c2, color="C2", ls="--", label=f"c2={c2}")
    ax_c.set_ylabel("c_dep")
    ax_c.legend(fontsize=8)
    states = bundle.extra.get("snapshots_S", [])
    if states and len(states) == t.size:
        y = np.asarray([1.0 if s == "H" else 0.0 for s in states])
        ax_s.fill_between(t, y, step="mid", alpha=0.4)
    ax_s.set_ylabel("S (H=1)")
    ax_s.set_xlabel("t")
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["L", "H"])
    fig.suptitle(f"{bundle.run_label}: hysteresis sawtooth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_block_c(path: Path, bundle: D1Bundle) -> None:
    c_ost = float(bundle.cfg["physics"]["c_ostwald"])
    w_ost = float(bundle.cfg["physics"]["w_ostwald"])
    c_lo, c_hi = _operational_c_bounds(c_ost, w_ost, _F_STAR)
    fig, (ax_psi, ax_c) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    deposit = (bundle.phi_m_final + bundle.phi_c_final) > _DEPOSIT_MASK
    ax_psi.plot(bundle.x[deposit], bundle.psi_final[deposit], lw=0.8)
    ax_psi.axhline(0, color="k", lw=0.5)
    ax_psi.set_xlabel("x")
    ax_psi.set_ylabel("psi(x) final")
    ax_psi.set_title(f"{bundle.run_label}: final psi — rind | zone | core")
    ax_c.plot(bundle.snap_t, bundle.snap_c_dep, lw=0.8)
    ax_c.axhline(c_ost, color="C0", ls="-", alpha=0.5, label="c_ostwald")
    ax_c.axhspan(c_ost - w_ost, c_ost + w_ost, alpha=0.15, color="C0", label="nominal window")
    ax_c.axhspan(c_lo, c_hi, alpha=0.15, color="C2", label=f"operational f*={_F_STAR}")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("c_dep")
    ax_c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _verdict_p1(p1: dict[str, Any]) -> str:
    s, lo, hi = p1["slope"], p1["slope_ci_lo"], p1["slope_ci_hi"]
    if not np.isfinite(s):
        return "INCONCLUSIVE (few band cycles in Block A)"
    if lo <= -1.0 <= hi:
        return "CONFIRMED lambda ~ 1/v (Krekhov attached; CI includes -1)"
    if lo <= -0.5 <= hi:
        return "CONFIRMED lambda ~ v^(-1/2) (CI includes -1/2)"
    if lo <= 0.0 <= hi:
        return "CONFIRMED lambda ~ const (CI includes 0)"
    return f"INCONCLUSIVE (best={p1['best_law']}, slope={s:.3f}, CI=[{lo:.3f},{hi:.3f}])"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    *,
    p1: dict,
    p2: dict,
    p3: dict,
    p4: dict,
    p5: dict,
    anchor_ok: bool,
    anchor_lams: list[float],
    anchor_cycles: list[dict[str, Any]],
) -> None:
    lines = [
        "# REPORT_D1 — discrimination campaign",
        "",
        f"CH exchange kernel (frozen): `{_KERNEL_CITE}`",
        "",
        "## Block A anchor",
        f"- {_ANCHOR_LABEL} lambda in [{_ANCHOR_LAMBDA[0]}, {_ANCHOR_LAMBDA[1]}]: "
        + ("PASS" if anchor_ok else "**FAIL — STOP** (driven mode does not match S1 pacing)"),
        f"- observed lambda: {anchor_lams} <!-- anchor.lambda_values -->",
    ]
    for c in anchor_cycles:
        lines.append(
            f"  - cycle {c.get('band_index')}: w_band={c.get('w_band'):.4g}, "
            f"gap={c.get('gap_k'):.4g}, lambda={c.get('lambda_k'):.4g} "
            f"<!-- anchor.cycle_{c.get('band_index')} -->"
        )
    lines += [
        "",
        "## P1 — CH selection law (lambda vs v, Block A f=0.30)",
        f"- log-log slope = {p1['slope']:.4g} <!-- p1.slope -->",
        f"- 95% CI = [{p1['slope_ci_lo']:.4g}, {p1['slope_ci_hi']:.4g}] <!-- p1.slope_ci -->",
        f"- best candidate: {p1['best_law']} <!-- p1.best_law -->",
        f"- **Verdict:** {_verdict_p1(p1)}",
        "",
        "## P2 — attachment (birth offset ≤ 3 dx)",
        f"- median offset = {p2['median_dx']:.4g} dx <!-- p2.median_dx -->",
        f"- max offset = {p2['max_dx']:.4g} dx <!-- p2.max_dx -->",
        "- **Verdict:** "
        + (
            "CONFIRMED (max ≤ 3 dx)"
            if np.isfinite(p2["max_dx"]) and p2["max_dx"] <= 3.0
            else "REFUTED (detachment exceeds 3 dx)"
            if np.isfinite(p2["max_dx"])
            else "INCONCLUSIVE (no band events)"
        ),
        "",
        "## P3 — composition and mobility",
        f"- collapse slope (lambda vs M0/v) = {p3['collapse_slope']:.4g} <!-- p3.collapse_slope -->",  # noqa: E501
        "- **Verdict:** INCONCLUSIVE (exploratory; see summary_d1.json p3.lambda_by_f)",
        "",
        "## P4 — hysteresis clock robustness",
        f"- Block B oscillates: {p4.get('oscillates')} <!-- p4.oscillates -->",
        f"- tau vs v slope: {p4.get('tau_vs_v_slope', float('nan')):.4g} <!-- p4.tau_vs_v_slope -->",  # noqa: E501
        "- **Verdict:** "
        + (
            "CONFIRMED clock decoupled (tau ~ const while gap shrinks)"
            if p4.get("clock_decoupled")
            else "CONFIRMED clock slaved (tau tracks gap)"
            if p4.get("clock_slaved")
            else "INCONCLUSIVE (no oscillation or ambiguous slopes)"
            if not p4.get("oscillates")
            else "INCONCLUSIVE (slopes neither ~0 nor ~-1)"
        ),
        "",
        "## P5 — agate architecture (Block C)",
    ]
    for run, p5r in p5.items():
        lines += [
            f"### {run}",
            f"- x_front_final = {p5r['x_front_final']:.4g} <!-- p5.{run}.x_front_final -->",
            f"- operational window [{p5r['operational_c_lo']:.4g}, {p5r['operational_c_hi']:.4g}]",
            f"- zones: {p5r['zones']}",
            "- **Verdict:** INCONCLUSIVE (inspect figure_d1_blockC.png)",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-D1 analysis")
    parser.add_argument("sweep_dir", type=Path, help="D1 sweep output directory (timestamp folder)")
    parser.add_argument(
        "--s1-sweep-dir",
        type=Path,
        default=None,
        help="S1 sweep dir for table_a21_bands.md import",
    )
    parser.add_argument(
        "--comparison-dir", type=Path, default=None, help="optional E1/E3 comparison results"
    )
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / "manifest_d1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not entries:
        raise RuntimeError("empty manifest")

    bundles: list[D1Bundle] = []
    all_bands: list[dict[str, Any]] = []
    all_cycles: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for entry in tqdm(entries, desc="D1 load", unit="run"):
        bundles.append(_load_d1_bundle(sweep_dir, entry))

    for bundle in tqdm(bundles, desc="D1 bands", unit="run"):
        t_term = float(bundle.snap_t[-1])
        if bundle.block in ("B", "C") and bundle.snap_c_dep.size:
            a2 = _load_module("a2_mod2", _SCRIPTS / "analyze_a2_understanding.py")
            try:
                t_term = a2._detect_terminal_time(bundle.snap_t, bundle.snap_c_dep)
            except RuntimeError:
                t_term = float(bundle.snap_t[-1])
        events = _extract_edge_births(bundle)
        bands, cycles = _pair_bands_a21(events, bundle, t_term=t_term)
        for b in bands:
            b["block"] = bundle.block
        for c in cycles:
            c["block"] = bundle.block
        all_bands.extend(bands)
        all_cycles.extend(cycles)
        per_run.append(
            {
                "run_label": bundle.run_label,
                "block": bundle.block,
                "n_bands": len(bands),
                "n_cycles": len(cycles),
                "t_term": t_term,
            }
        )

    cycles_a = [c for c in all_cycles if c.get("block") == "A" and not c.get("terminal")]
    cycles_b = [c for c in all_cycles if c.get("block") == "B"]
    cycles_c = [c for c in all_cycles if c.get("block") == "C"]

    anchor_cycles = [c for c in cycles_a if c["run_label"] == _ANCHOR_LABEL]
    anchor_lams = [
        float(c["lambda_k"]) for c in anchor_cycles if np.isfinite(c.get("lambda_k", float("nan")))
    ]
    anchor_ok = bool(anchor_lams) and all(
        _ANCHOR_LAMBDA[0] <= lam <= _ANCHOR_LAMBDA[1] for lam in anchor_lams
    )

    p1 = _score_p1(cycles_a)
    p2 = _score_p2(all_bands)
    p3 = _score_p3(cycles_a)
    p4_by_run = {
        b.run_label: _score_p4(b, [c for c in cycles_b if c["run_label"] == b.run_label])
        for b in bundles
        if b.block == "B"
    }
    p4 = p4_by_run.get("D1B_base", {"oscillates": False})
    p5 = {b.run_label: _score_p5(b) for b in bundles if b.block == "C"}

    s1_pts: list[dict[str, Any]] = []
    if args.s1_sweep_dir:
        s1_pts = _parse_a21_table(args.s1_sweep_dir / "table_a21_bands.md")

    e1e3_pts: list[dict[str, Any]] = []
    if args.comparison_dir and (args.comparison_dir / "summary_comparison.json").is_file():
        comp = json.loads(
            (args.comparison_dir / "summary_comparison.json").read_text(encoding="utf-8")
        )
        for row in comp.get("rows", []):
            if str(row.get("run_label", "")).startswith(("E1_", "E3_")):
                e1e3_pts.append(
                    {
                        "run_label": row["run_label"],
                        "v": float(row.get("v_mean", row.get("v", float("nan")))),
                        "lambda": float(row.get("lambda_mean", row.get("lambda", float("nan")))),
                        "tau_cycle": float(row.get("tau_mean", row.get("tau", float("nan")))),
                    }
                )

    table_rows: list[dict[str, Any]] = []
    cycle_by = {(c["run_label"], c["band_index"]): c for c in all_cycles}
    for b in all_bands:
        cy = cycle_by.get((b["run_label"], b["band_index"]), {})
        table_rows.append(
            {
                **b,
                "block": b.get("block"),
                "gap": cy.get("gap_k"),
                "lambda": cy.get("lambda_k"),
                "tau_cycle": cy.get("tau_cycle_k"),
                "duty": cy.get("duty_k"),
                "v_at_open": b.get("v_front_smooth_at_open"),
                "birth_offset_dx": b.get("birth_offset_dx"),
            }
        )

    _write_bands_table(sweep_dir / "table_d1_bands.md", table_rows)
    _plot_master(
        sweep_dir / "figure_d1_master.png",
        cycles_a=cycles_a,
        cycles_b=cycles_b,
        cycles_c=cycles_c,
        s1_pts=s1_pts,
        e1e3_pts=e1e3_pts,
    )
    b_bundle = next((b for b in bundles if b.run_label == "D1B_base"), None)
    if b_bundle:
        _plot_block_b(sweep_dir / "figure_d1_blockB.png", b_bundle)
    c_bundle = next((b for b in bundles if b.run_label == "D1C_g4"), None)
    if c_bundle:
        _plot_block_c(sweep_dir / "figure_d1_blockC.png", c_bundle)

    summary = {
        "kernel_citation": _KERNEL_CITE,
        "anchor": {
            "run_label": _ANCHOR_LABEL,
            "lambda_range": _ANCHOR_LAMBDA,
            "lambda_values": anchor_lams,
            "pass": anchor_ok,
            "cycles": [
                {
                    "band_index": c.get("band_index"),
                    "w_band": c.get("w_band"),
                    "gap_k": c.get("gap_k"),
                    "lambda_k": c.get("lambda_k"),
                    "v_open_k": c.get("v_open_k"),
                }
                for c in anchor_cycles
            ],
        },
        "per_run": per_run,
        "predictions": {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P4_by_run": p4_by_run, "P5": p5},
        "n_bands_total": len(all_bands),
        "n_cycles_total": len(all_cycles),
        "s1_import_n": len(s1_pts),
        "e1e3_import_n": len(e1e3_pts),
    }
    (sweep_dir / "summary_d1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(
        sweep_dir / "REPORT_D1.md",
        summary,
        p1=p1,
        p2=p2,
        p3=p3,
        p4=p4,
        p5=p5,
        anchor_ok=anchor_ok,
        anchor_lams=anchor_lams,
        anchor_cycles=anchor_cycles,
    )
    print(f"D1 analysis complete: {sweep_dir}")
    if not anchor_ok:
        print(
            f"ANCHOR GATE FAIL: {_ANCHOR_LABEL} lambda={anchor_lams} not in {_ANCHOR_LAMBDA} "
            "(deliverables written; driven mode ≠ S1 reference pacing — see REPORT_D1.md)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
