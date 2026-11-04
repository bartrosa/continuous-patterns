#!/usr/bin/env python3
"""Run a fresh multi-model banding comparison from a YAML config."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from continuous_patterns.core.banding_metrics import (  # noqa: E402
    BandingRecord,
    _has_terminal_band_spike,
    compute_clip_rate,
    compute_liesegang_laws,
    deposit_extent,
    front_origin_for_record,
    front_position_vs_time,
    record_from_slab_kinetic_result,
    record_from_slab_reactive_outputs,
    record_from_slab_twophase_result,
    s1_twophase_validity,
)
from continuous_patterns.core.io import (  # noqa: E402
    config_hash,
    load_run_config,
    validate_run_config_dict,
)
from continuous_patterns.experiments.checkpoint import (  # noqa: E402
    atomic_write_json,
    is_run_complete,
    load_comparison_progress,
    load_sim_result_from_run_dir,
    mark_comparison_run_complete,
    mark_comparison_run_failed,
)
from continuous_patterns.experiments.run import run_one  # noqa: E402
from continuous_patterns.models.slab_twophase import estimate_integration_steps  # noqa: E402

_LABEL_COLORS = {
    "H": "#d62728",
    "L": "#1f77b4",
    "m": "#d62728",
    "c": "#1f77b4",
    "solid": "#7f7f7f",
}

_LABEL_LEGEND = [
    ("SiO2 phase A", "#d62728"),
    ("SiO2 phase B", "#1f77b4"),
    ("mixture", "#7f7f7f"),
]

# Rows whose mechanism_class contains this token get the Heaney section header.
_HEANEY_SECTION_TOKEN = "Heaney"
_CH_SECTION_TOKEN = "CH"

_TIER1_HEANEY = (
    "HEANEY MODEL — kinetic hysteresis clock | liquid concentration oscillates at the "
    "reaction surface; sets the PERIOD"
)
_TIER1_CH = (
    "CAHN-HILLIARD MODEL — phase-field demixing | composition separates in the solid "
    "behind the front; sets the WAVELENGTH"
)

_PANEL_TITLES = ("Wall dynamics", "Kymograph", "Band pattern", "Spacing laws")
_FONT_TITLE = 11
_FONT_LABEL = 10
_FONT_TICK = 8
_FONT_CAPTION = 8
_MAX_KYMO_ROWS = 160
_MAX_WALL_PTS = 4000


def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Coerce diagnostic scalars; ``None`` and bad values map to *default*."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


_REGRESSION_CHECKS: dict[str, dict[str, Any]] = {
    "Heaney_osc_Da1": {
        "n_transitions": {"target": 114.0, "tol": 5.0},
        "mean_dwell_period": {"target": 0.35, "rel_tol": 0.15},
    },
    "Heaney_stuck_Da1": {
        "n_transitions_max": 1,
        "n_bands_max": 2,
    },
    "CH_open_legacy": {
        "band_count_axial": {"target": 15, "exact": True},
    },
}

_BASELINE_ESTABLISHING = frozenset({"CH_rim_only", "CH_closed_seed"})

_E3_DEFAULT_PAIRS: list[dict[str, Any]] = [
    {"c1": 0.70, "c2": 0.66, "label": "0.70/0.66"},
    {"c1": 0.60, "c2": 0.56, "label": "0.60/0.56"},
    {"c1": 0.50, "c2": 0.46, "label": "0.50/0.46"},
]

_E3_HEANEY_CLOSED = "E3_Heaney_closed"


def _find_e3_closed_root(out_dir: Path) -> Path:
    return out_dir / "runs" / _E3_HEANEY_CLOSED


def _newest_complete_run_for_label(out_dir: Path, run_label: str, invocation_id: str) -> Path:
    return _newest_complete_run(out_dir / "runs" / run_label, invocation_id)


def _twophase_is_driven(cfg: dict[str, Any]) -> bool:
    phys = cfg.get("physics") if isinstance(cfg.get("physics"), dict) else {}
    if str(phys.get("run_mode", "")).lower() == "driven":
        return True
    return str(phys.get("precipitation", "")).lower() == "driven"


def _status_draws_overlay(status: str) -> bool:
    return status in ("NUMERICALLY_INVALID", "RUN_INVALID")


def _status_is_honest_negative(status: str, invalid_reasons: list[str]) -> bool:
    return status == "OK" and any("no sustained oscillation" in str(r) for r in invalid_reasons)


def _verify_physics_overrides(entry: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Ensure every ``physics.*`` comparison override reached the solver config."""
    overrides = entry.get("overrides", {})
    if not isinstance(overrides, dict):
        return
    phys = cfg.get("physics")
    if not isinstance(phys, dict):
        raise ValueError(f"{entry['run_label']}: missing physics block after merge")
    for dotted, value in overrides.items():
        key_path = str(dotted)
        if not key_path.startswith("physics."):
            continue
        key = key_path.split(".", 1)[1]
        if key not in phys:
            raise ValueError(
                f"{entry['run_label']}: physics override {key_path!r} missing from merged config"
            )
        got = phys[key]
        if isinstance(value, float) and isinstance(got, (int, float)):
            if not np.isclose(float(got), float(value), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"{entry['run_label']}: override {key_path}={value!r} not applied (got {got!r})"
                )
        elif got != value:
            raise ValueError(
                f"{entry['run_label']}: override {key_path}={value!r} not applied (got {got!r})"
            )


def _closed_solid_budget(cfg: dict[str, Any]) -> float:
    """Maximum deposit mass (solution + solid) for closed rows (+0.5% tolerance)."""
    g = cfg.get("geometry") or {}
    ini = cfg.get("initial") or {}
    phys = cfg.get("physics") or {}
    L = float(g.get("L", 100.0))
    n = int(g["n"])
    seed = max(1, int(g.get("seed_width_px", 2)))
    c_init = float(ini.get("c_init", 0.3))
    rho = float(phys.get("rho", 1.0))
    x = np.linspace(0.0, L, n, dtype=np.float64)
    phi = np.zeros(n, dtype=np.float64)
    phi[:seed] = 0.5
    phi_w = np.maximum(1.0 - phi, 0.0)
    m_sol = float(np.trapezoid(phi_w * c_init, x))
    m_dep = float(rho * np.trapezoid(phi, x))
    return (m_sol + m_dep) * 1.005


def _row_integrity_status(
    model: str,
    cfg: dict[str, Any],
    diag: dict[str, Any],
    *,
    run_label: str | None = None,
    n_bands: int = 0,
    mean_spacing: float | None = None,
) -> tuple[str, list[str]]:
    """Return ``(status, reasons)`` with ``OK``, ``WARN``, ``RUN_INVALID``, or ``NUMERICALLY_INVALID``."""  # noqa: E501
    reasons: list[str] = []
    status = "OK"
    supply = _supply_regime(cfg)

    if model == "slab_twophase":
        tier = s1_twophase_validity(diag)
        if tier == "NUMERICALLY_INVALID":
            mass = diag.get("mass_residual_pct")
            min_pw = diag.get("min_phi_w_wetted", diag.get("min_phi_w"))
            if mass is not None and float(mass) >= 1e-1:
                reasons.append(f"mass_residual_pct={mass}")
            if min_pw is not None and float(min_pw) < -1e-6:
                reasons.append(f"min_phi_w={min_pw}")
            status = "NUMERICALLY_INVALID"
        elif tier == "WARN":
            mass = diag.get("mass_residual_pct")
            reasons.append(f"mass_residual_pct={mass} (soft WARN tier)")
            if status == "OK":
                status = "WARN"

        lam = _safe_float(diag.get("lambda_ch"))  # noqa: F841
        if run_label == "E8_TP_gamma4_CH_matchedOstwald":
            if n_bands < 4:
                reasons.append(f"n_bands={n_bands} (expected >=4)")
                status = "RUN_INVALID"

        if run_label in ("E4_TP_gamma4_open", "E5_TP_gamma0_open"):
            if bool(diag.get("front_frozen")):
                reasons.append("front_frozen")
                status = "RUN_INVALID"
            elif _safe_float(diag.get("mass_residual_pct")) > 1.0:
                reasons.append(f"mass_residual_pct={diag.get('mass_residual_pct')}")
                status = "RUN_INVALID"
            else:
                status = "OK"
                reasons.clear()

    if supply == "closed":
        flux = diag.get("max_abs_right_flux")
        if flux is not None and float(flux) > 1e-12:
            reasons.append(f"max_abs_right_flux={flux}")
            if status == "OK":
                status = "RUN_INVALID"
        mass = diag.get("mass_residual_pct")
        if mass is not None and float(mass) > 1.0:
            reasons.append(f"mass_residual_pct={mass}")
            if status == "OK":
                status = "RUN_INVALID"
        budget = _closed_solid_budget(cfg)
        m_dep = diag.get("M_deposit_final")
        if m_dep is not None and float(m_dep) > budget:
            reasons.append(f"M_deposit_final={float(m_dep):.4g} > budget={budget:.4g}")
            if status == "OK":
                status = "RUN_INVALID"

    return status, reasons


def _plot_invalid_cell(ax: plt.Axes, status: str, reasons: list[str]) -> None:
    ax.set_facecolor("#ffe6e6")
    msg = status.replace("_", " ")
    detail = "; ".join(reasons[:4]) if reasons else "integrity check failed"
    ax.text(
        0.5,
        0.55,
        msg,
        ha="center",
        va="center",
        fontsize=_FONT_LABEL + 2,
        fontweight="bold",
        color="#b00020",
    )
    ax.text(
        0.5,
        0.35,
        detail,
        ha="center",
        va="center",
        fontsize=_FONT_TICK,
        color="#b00020",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b00020")
        spine.set_linewidth(2.0)


def _run_e3_heaney_closed(
    entry: dict[str, Any],
    repo_root: Path,
    out_dir: Path,
    *,
    show_progress: bool = True,
    invocation_id: str,
) -> tuple[Any, dict[str, Any], str, dict[str, Any]]:
    """Run three closed-cavity Heaney threshold pairs; return best result and pair summary."""
    pairs_raw = entry.get("e3_pairs", _E3_DEFAULT_PAIRS)
    if not isinstance(pairs_raw, list) or not pairs_raw:
        pairs_raw = _E3_DEFAULT_PAIRS

    pair_results: list[tuple[str, float, Any, dict[str, Any]]] = []
    e3_root = _find_e3_closed_root(out_dir)

    for pair in pairs_raw:
        if not isinstance(pair, dict):
            continue
        label = str(pair.get("label", f"{pair.get('c1')}/{pair.get('c2')}"))
        safe_label = label.replace("/", "_")
        exp_name = f"{_E3_HEANEY_CLOSED}_{safe_label}"
        pair_root = e3_root / safe_label
        cfg = _build_run_config(entry, repo_root)
        cfg.setdefault("physics", {})
        cfg["physics"]["c1"] = float(pair["c1"])
        cfg["physics"]["c2"] = float(pair["c2"])
        cfg["physics"]["bc_right"] = "noflux"
        cfg.setdefault("initial", {})
        cfg["initial"]["profile"] = "uniform"
        cfg["initial"]["c_init"] = 1.0
        cfg.setdefault("time", {})
        cfg["time"]["T"] = 40.0
        cfg["experiment"]["name"] = exp_name
        result = run_one(
            cfg,
            results_root=pair_root,
            write_artifacts=True,
            show_progress=show_progress,
            log_level=str(cfg.get("output", {}).get("log_level", "INFO")),
            summary_extra={
                "invocation_id": invocation_id,
                "config_hash": config_hash(cfg),
            },
        )
        diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        n_tr = float(diag.get("n_transitions", 0))
        pair_results.append((label, n_tr, result, cfg))

    if not pair_results:
        raise ValueError(f"{_E3_HEANEY_CLOSED}: no valid e3_pairs configured")

    best_label, best_n, best_result, best_cfg = max(pair_results, key=lambda item: item[1])
    summary_parts = [f"{lbl}→{int(n) if n == int(n) else n:g}" for lbl, n, _, _ in pair_results]
    pair_summary = f"E3 pairs: {', '.join(summary_parts)} (shown: {best_label}→{int(best_n) if best_n == int(best_n) else best_n:g})"  # noqa: E501
    pair_counts = {lbl: n for lbl, n, _, _ in pair_results}
    extra_meta = {
        "e3_pair_summary": pair_summary,
        "e3_pair_counts": pair_counts,
        "e3_shown_pair": best_label,
    }
    return best_result, best_cfg, best_label, extra_meta


def _n_bands_from_record(rec: BandingRecord) -> int:
    return int(np.asarray(rec.band_positions, dtype=np.float64).size)


def _supply_regime(cfg: dict[str, Any]) -> str:
    bc = str((cfg.get("physics") or {}).get("bc_right", "dirichlet")).lower()
    return "closed" if bc == "noflux" else "open"


def _crop_kymograph_time(
    field: np.ndarray,
    times: np.ndarray,
    *,
    margin_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).ravel()
    if field.shape[0] < 2 or times.size < 2:
        return field, times
    delta = np.max(np.abs(np.diff(field, axis=0)), axis=1)
    thresh = 1e-4
    for i in range(len(delta)):
        if np.all(delta[i:] < thresh):
            tail = max(1, int(margin_frac * (len(times) - i)))
            end_idx = min(len(times) - 1, i + tail)
            return field[: end_idx + 1], times[: end_idx + 1]
    return field, times


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: dict[str, Any] = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = copy.deepcopy(value)


def _prune_slab_twophase_physics(cfg: dict[str, Any]) -> None:
    """Drop mode-incompatible ``physics`` keys after comparison overrides."""
    if str(cfg.get("experiment", {}).get("model", "")) != "slab_twophase":
        return
    phys = cfg.get("physics")
    if not isinstance(phys, dict):
        return
    demixing = str(phys.get("demixing", "ac")).lower()
    if demixing == "ch":
        for key in ("kappa_s", "L_s", "W"):
            phys.pop(key, None)
    else:
        for key in ("W", "kappa", "M0"):
            phys.pop(key, None)


def _build_run_config(entry: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    base_path = repo_root / str(entry["base_config"])
    cfg = load_run_config(base_path)
    overrides = entry.get("overrides", {})
    if isinstance(overrides, dict):
        for dotted, value in overrides.items():
            _set_dotted(cfg, str(dotted), value)
    run_label = str(entry["run_label"])
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = run_label
    cfg["experiment"]["model"] = str(entry["model"])
    if str(entry["model"]) == "slab_twophase":
        _prune_slab_twophase_physics(cfg)
        return validate_run_config_dict(cfg)
    return cfg


def _is_ingest_entry(entry: dict[str, Any]) -> bool:
    return bool(entry.get("ingest_from") or entry.get("ingest"))


def _newest_timestamp_dir(root: Path) -> Path:
    subs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subs:
        raise FileNotFoundError(f"no timestamp directories under {root}")
    return subs[-1]


def _manifest_run_dir(manifest_path: Path, run_label: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for ent in manifest.get("entries", []):
        if str(ent.get("run_label")) == run_label:
            rel = ent.get("relative_path")
            if not rel:
                raise RuntimeError(f"{run_label}: missing relative_path in {manifest_path}")
            return (manifest_path.parent / str(rel)).resolve()
    raise KeyError(f"{run_label} not found in {manifest_path}")


def _resolve_ingest_run_dir(entry: dict[str, Any], comp: dict[str, Any], repo_root: Path) -> Path:
    """Resolve on-disk run directory for ingest rows (hard-fail if missing)."""
    if "ingest_from" in entry:
        p = Path(str(entry["ingest_from"]))
        run_dir = p if p.is_absolute() else repo_root / p
    else:
        mode = str(entry.get("ingest", ""))
        label = str(entry.get("ingest_run", entry["run_label"]))
        if mode == "d1":
            d1_root = repo_root / str(comp["d1_sweep_dir"])
            run_dir = _manifest_run_dir(d1_root / "manifest_d1.json", label)
        elif mode == "d2h":
            d2_root = repo_root / str(comp["d2h_sweep_dir"])
            if (d2_root / "manifest_d2h.json").is_file():
                manifest_path = d2_root / "manifest_d2h.json"
            else:
                manifest_path = _newest_timestamp_dir(d2_root) / "manifest_d2h.json"
            run_dir = _manifest_run_dir(manifest_path, label)
        elif mode == "d2h_best":
            d2_root = repo_root / str(comp["d2h_sweep_dir"])
            verdict_path = comp.get("d2_verdict_path")
            if verdict_path:
                vp = Path(str(verdict_path))
                vp = vp if vp.is_absolute() else repo_root / vp
            else:
                vp = _newest_timestamp_dir(d2_root) / "d2_part_a_verdict.json"
            if not vp.is_file():
                raise FileNotFoundError(f"D2H verdict missing: {vp} (run analyze_d2.py first)")
            verdict = json.loads(vp.read_text(encoding="utf-8"))
            best = str(verdict["master_comparison_run"])
            sweep_dir = Path(str(verdict.get("d2h_sweep_dir", "")))
            if sweep_dir.is_dir():
                manifest_path = sweep_dir / "manifest_d2h.json"
            elif (d2_root / "manifest_d2h.json").is_file():
                manifest_path = d2_root / "manifest_d2h.json"
            else:
                manifest_path = _newest_timestamp_dir(d2_root) / "manifest_d2h.json"
            run_dir = _manifest_run_dir(manifest_path, best)
        elif mode == "d3h_best":
            d3_root = repo_root / str(comp["d3h_sweep_dir"])
            verdict_path = comp.get("d3_verdict_path")
            if verdict_path:
                vp = Path(str(verdict_path))
                vp = vp if vp.is_absolute() else repo_root / vp
            else:
                vp = _newest_timestamp_dir(d3_root) / "d3_part_b_verdict.json"
            if not vp.is_file():
                raise FileNotFoundError(
                    f"D3H verdict missing: {vp} — run D3H sweep + analyze_d3.py first"
                )
            verdict = json.loads(vp.read_text(encoding="utf-8"))
            best = str(verdict["master_comparison_run"])
            sweep_dir = Path(str(verdict.get("d3h_sweep_dir", "")))
            if sweep_dir.is_dir():
                manifest_path = sweep_dir / "manifest_d3h.json"
            elif (d3_root / "manifest_d3h.json").is_file():
                manifest_path = d3_root / "manifest_d3h.json"
            else:
                manifest_path = _newest_timestamp_dir(d3_root) / "manifest_d3h.json"
            run_dir = _manifest_run_dir(manifest_path, best)
        elif mode == "d5h_best":
            d5_root = repo_root / str(comp["d5h_sweep_dir"])
            verdict_path = comp.get("d5_verdict_path")
            if verdict_path:
                vp = Path(str(verdict_path))
                vp = vp if vp.is_absolute() else repo_root / vp
            else:
                vp = _newest_timestamp_dir(d5_root) / "d5_part_a_verdict.json"
            if not vp.is_file():
                raise FileNotFoundError(
                    f"D5H verdict missing: {vp} — run D5H sweep + analyze_d5.py first"
                )
            verdict = json.loads(vp.read_text(encoding="utf-8"))
            best = str(verdict["master_comparison_run"])
            sweep_dir = Path(str(verdict.get("d5h_sweep_dir", "")))
            if sweep_dir.is_dir():
                manifest_path = sweep_dir / "manifest_d5h.json"
            elif (d5_root / "manifest_d5h.json").is_file():
                manifest_path = d5_root / "manifest_d5h.json"
            else:
                manifest_path = _newest_timestamp_dir(d5_root) / "manifest_d5h.json"
            run_dir = _manifest_run_dir(manifest_path, best)
        elif mode == "d6h_best":
            d6_root = Path(str(comp["d6h_sweep_dir"]))
            if not d6_root.is_absolute():
                d6_root = (repo_root / d6_root).resolve()
            else:
                d6_root = d6_root.resolve()
            verdict_path = comp.get("d6_verdict_path")
            if verdict_path:
                vp = Path(str(verdict_path))
                if not vp.is_absolute():
                    vp = (repo_root / vp).resolve()
                else:
                    vp = vp.resolve()
            else:
                vp = _newest_timestamp_dir(d6_root) / "d6_part_c_verdict.json"
            if not vp.is_file():
                raise FileNotFoundError(
                    f"D6H verdict missing: {vp} — run D6H sweep + analyze_d6.py first"
                )
            verdict = json.loads(vp.read_text(encoding="utf-8"))
            best = str(verdict["master_comparison_run"])
            sweep_dir = Path(str(verdict.get("d6h_sweep_dir", "")))
            if sweep_dir.is_dir():
                manifest_path = sweep_dir / "manifest_d6h.json"
            elif (d6_root / "manifest_d6h.json").is_file():
                manifest_path = d6_root / "manifest_d6h.json"
            else:
                manifest_path = _newest_timestamp_dir(d6_root) / "manifest_d6h.json"
            run_dir = _manifest_run_dir(manifest_path, best)
        else:
            raise ValueError(f"unknown ingest mode {mode!r} for {entry.get('run_label')}")
    if not is_run_complete(run_dir):
        raise FileNotFoundError(f"ingest source incomplete or missing: {run_dir}")
    return Path(run_dir)


def _preflight_comparison_experiments(
    experiments: list[dict[str, Any]],
    repo_root: Path,
    comp: dict[str, Any] | None = None,
    *,
    max_ac_steps: int = 5_000_000,
    max_ch_steps: int = 50_000_000,
) -> None:
    """Validate every row; print CH dt/n_steps budget; reject impossible runs."""
    import tempfile

    from continuous_patterns.core.io import save_run_config

    errors: list[str] = []
    for entry in experiments:
        if not isinstance(entry, dict):
            errors.append("experiment entry must be a mapping")
            continue
        if _is_ingest_entry(entry):
            label = str(entry.get("run_label", "?"))
            try:
                _resolve_ingest_run_dir(entry, comp or {}, repo_root)
            except Exception as exc:
                errors.append(f"{label}: ingest preflight — {exc}")
            continue
        label = str(entry.get("run_label", "?"))
        model = str(entry.get("model", ""))
        try:
            cfg = _build_run_config(entry, repo_root)
            _verify_physics_overrides(entry, cfg)
            with tempfile.TemporaryDirectory() as td:
                save_run_config(Path(td) / "config.yaml", cfg)
            if model == "slab_twophase":
                dt, n_steps = estimate_integration_steps(cfg)
                demixing = str(cfg.get("physics", {}).get("demixing", "ac")).lower()
                limit = max_ch_steps if demixing == "ch" else max_ac_steps
                cavity_row = bool(entry.get("cavity_row"))
                if n_steps > limit and not cavity_row:
                    errors.append(
                        f"{label}: slab_twophase would need {n_steps} steps "
                        f"(dt={dt:.4e}, limit={limit}); fix before running"
                    )
                elif cavity_row and n_steps > limit:
                    print(
                        f"[preflight] {label}: cavity_row long run n_steps={n_steps} "
                        f"(limit={limit} waived)",
                        flush=True,
                    )
                if demixing == "ch":
                    if dt > 1e-3:
                        errors.append(
                            f"{label}: CH dt={dt:.4e} looks like O(dx^2) CFL; "
                            f"expected O(dx^4) limit ~1.5e-4"
                        )
                    print(
                        f"[preflight] {label}: CH dt={dt:.6e} n_steps={n_steps} T={cfg['time']['T']}",  # noqa: E501
                        flush=True,
                    )
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    if errors:
        raise ValueError("Comparison preflight failed:\n  - " + "\n  - ".join(errors))


def _record_from_result(result: Any, entry: dict[str, Any]) -> BandingRecord:
    run_label = str(entry["run_label"])
    model = str(entry["model"])
    if model == "slab_kinetic":
        return record_from_slab_kinetic_result(result, run_label)
    if model == "slab_reactive":
        assert result.paths is not None
        side = str(entry.get("nucleation_side", "left"))
        cfg = result.config_resolved if isinstance(result.config_resolved, dict) else {}
        return record_from_slab_reactive_outputs(
            result.paths.final_state_npz,
            cfg,
            run_label,
            nucleation_side="right" if side == "right" else "left",
        )
    if model == "slab_twophase":
        return record_from_slab_twophase_result(result, run_label)
    raise ValueError(f"Unsupported model for banding comparison: {model!r}")


def _process_comparison_entry(
    entry: dict[str, Any],
    repo_root: Path,
    out_dir: Path,
    *,
    show_progress: bool,
    invocation_id: str,
    comp: dict[str, Any] | None = None,
) -> tuple[
    Any,
    dict[str, Any],
    BandingRecord,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Run one comparison row fresh or ingest from disk."""
    run_label = str(entry["run_label"])
    model = str(entry["model"])
    e3_extra: dict[str, Any] = {}

    if _is_ingest_entry(entry):
        run_dir = _resolve_ingest_run_dir(entry, comp or {}, repo_root)
        cfg = load_run_config(run_dir / "config.yaml")
        result = load_sim_result_from_run_dir(run_dir, model)
        row_bundle = _build_comparison_row(entry, result, cfg, e3_extra)
        # stamp ingest provenance on metrics
        _, cfg_o, rec, row, entry_metrics, fo, regression = row_bundle
        entry_metrics["run_dir"] = str(run_dir)
        entry_metrics["ingest_source"] = str(run_dir)
        return result, cfg_o, rec, row, entry_metrics, fo, regression

    if run_label == _E3_HEANEY_CLOSED and model == "slab_kinetic":
        result, cfg, _, e3_extra = _run_e3_heaney_closed(
            entry,
            repo_root,
            out_dir,
            show_progress=show_progress,
            invocation_id=invocation_id,
        )
    else:
        cfg = _build_run_config(entry, repo_root)
        _verify_physics_overrides(entry, cfg)
        cfg.setdefault("output", {})
        cfg["output"]["save_evolution"] = True
        cfg_hash = config_hash(cfg)
        result = run_one(
            cfg,
            results_root=out_dir / "runs",
            write_artifacts=True,
            show_progress=show_progress,
            log_level=str(cfg.get("output", {}).get("log_level", "INFO")),
            summary_extra={
                "invocation_id": invocation_id,
                "config_hash": cfg_hash,
            },
        )

    return _build_comparison_row(entry, result, cfg, e3_extra)


def _build_comparison_row(
    entry: dict[str, Any],
    result: Any,
    cfg: dict[str, Any],
    e3_extra: dict[str, Any],
) -> tuple[
    Any,
    dict[str, Any],
    BandingRecord,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    run_label = str(entry["run_label"])
    model = str(entry["model"])
    rec = _record_from_result(result, entry)
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    n_bands_rec = _n_bands_from_record(rec)
    laws_preview = compute_liesegang_laws(rec)
    d_sp = laws_preview.get("d_spacings")
    mean_sp = (
        float(np.mean(np.asarray(d_sp, dtype=np.float64)))
        if isinstance(d_sp, list) and d_sp
        else None
    )
    row_status, invalid_reasons = _row_integrity_status(
        model,
        cfg,
        diag,
        run_label=str(entry.get("run_label")),
        n_bands=n_bands_rec,
        mean_spacing=mean_sp,
    )
    laws: dict[str, Any]
    if row_status in ("RUN_INVALID", "NUMERICALLY_INVALID"):
        laws = {
            "n_bands": n_bands_rec,
            "n_segments": 0,
            "p_fit": None,
            "Q_cv": None,
            "Q_cv_tail": None,
            "p_fit_excl_last": None,
            "Q_cv_tail_excl_last": None,
            "laws_reason": "row_invalid",
            "time_law": {"alpha": None},
        }
        if n_bands_rec >= 6:
            laws = compute_liesegang_laws(rec)
            laws["laws_reason"] = f"integrity_{row_status.lower()}"
    else:
        laws = compute_liesegang_laws(rec)
    if row_status == "OK" and int(laws["n_bands"]) != n_bands_rec:
        raise ValueError(
            f"{run_label}: n_bands mismatch laws={laws['n_bands']} record={n_bands_rec}"
        )
    regression = _run_regression(
        run_label,
        laws,
        diag,
        entry_spec=entry.get("regression") if isinstance(entry.get("regression"), dict) else None,
    )
    run_path = result.paths.root if result.paths else None
    snapshots = _load_run_snapshots(run_path)
    history = _load_run_history(run_path)
    fo = front_origin_for_record(rec, snapshots=snapshots, history=history)
    mass_pct = _mass_residual_pct(model, diag)
    min_phi_w = diag.get("min_phi_w") if model == "slab_twophase" else None
    n_proj = diag.get("n_projection_events") if model == "slab_twophase" else None
    clip_rate = diag.get("clip_rate") if model == "slab_twophase" else None
    if clip_rate is None and model == "slab_twophase":
        clip_rate = compute_clip_rate(
            diag.get("n_projection_events"),
            diag.get("n_steps"),
            (cfg.get("geometry") or {}).get("n"),
        )
    supply = _supply_regime(cfg)
    row = {
        "run_label": run_label,
        "model": model,
        "supply": supply,
        "row_status": row_status,
        "invalid_reasons": invalid_reasons,
        "n_bands": n_bands_rec,
        "n_segments": laws.get("n_segments"),
        "p_fit": laws["p_fit"],
        "CV(Q)_tail": laws.get("Q_cv_tail", laws.get("Q_cv")),
        "p_fit_excl_last": laws.get("p_fit_excl_last"),
        "CV(Q)_tail_excl_last": laws.get("Q_cv_tail_excl_last"),
        "front_origin": fo.get("origin"),
        "alpha_time": laws["time_law"]["alpha"],
        "mass_residual_pct": mass_pct,
        "min_phi_w": min_phi_w,
        "n_projection_events": n_proj,
        "clip_rate": clip_rate,
        "max_abs_right_flux": diag.get("max_abs_right_flux"),
        "front_frozen": diag.get("front_frozen"),
        "lambda_ch": diag.get("lambda_ch"),
        "dt": diag.get("dt"),
        "n_steps": diag.get("n_steps"),
        "regression": regression["status"],
    }
    entry_metrics = {
        **laws,
        "run_label": run_label,
        "model": model,
        "row_status": row_status,
        "invalid_reasons": invalid_reasons,
        "front_origin": fo,
        "regression": regression,
        "diagnostics": {
            k: diag[k]
            for k in (
                "n_transitions",
                "mean_dwell_period",
                "thickness_final",
                "wall_time_s",
                "dt",
                "n_steps",
                "lambda_ch",
                "front_frozen",
                "max_abs_right_flux",
            )
            if k in diag
        },
        "mass_residual_pct": mass_pct,
        "min_phi_w": min_phi_w,
        "n_projection_events": n_proj,
        "clip_rate": clip_rate,
        "max_abs_right_flux": diag.get("max_abs_right_flux"),
        "front_frozen": diag.get("front_frozen"),
        "lambda_ch": diag.get("lambda_ch"),
        "supply": supply,
        "run_dir": str(result.paths.root) if result.paths else None,
        "config_hash": config_hash(cfg),
        **e3_extra,
    }
    return result, cfg, rec, row, entry_metrics, fo, regression


def _write_metrics_checkpoint(
    out_dir: Path,
    *,
    comp: dict[str, Any],
    config_stem: str,
    table_rows: list[dict[str, Any]],
    metrics_all: dict[str, Any],
    invocation_id: str,
    wall_s: float,
    completed: int,
    total: int,
) -> None:
    """Persist metrics/table after each row so export failures never lose the full day."""
    metrics_doc = {
        "comparison_name": comp.get("name", config_stem),
        "invocation_id": invocation_id,
        "wall_time_s": wall_s,
        "completed_rows": completed,
        "total_rows": total,
        "export_complete": completed >= total,
        "experiments": metrics_all,
    }
    atomic_write_json(out_dir / "metrics.json", metrics_doc)
    _write_comparison_table(out_dir / "table_comparison.md", table_rows)


def _write_comparison_outputs(
    out_dir: Path,
    *,
    comp: dict[str, Any],
    config_stem: str,
    experiments: list[dict[str, Any]],
    records: list[BandingRecord],
    table_rows: list[dict[str, Any]],
    metrics_all: dict[str, Any],
    cfgs: list[dict[str, Any]],
    front_origins: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    wall_s: float,
    invocation_id: str,
) -> None:
    metrics_doc = {
        "comparison_name": comp.get("name", config_stem),
        "invocation_id": invocation_id,
        "wall_time_s": wall_s,
        "completed_rows": len(experiments),
        "total_rows": len(experiments),
        "export_complete": True,
        "experiments": metrics_all,
    }
    atomic_write_json(out_dir / "metrics.json", metrics_doc)
    _write_comparison_table(out_dir / "table_comparison.md", table_rows)

    npz_payload: dict[str, Any] = {}
    for rec in records:
        serialized = _serialize_record(rec)
        for key, val in serialized.items():
            npz_payload[f"{rec.run_label}/{key}"] = val
    np.savez_compressed(out_dir / "records.npz", **npz_payload)

    metrics_list = [metrics_all[str(entry["run_label"])] for entry in experiments]
    figure_name = str(comp.get("figure_out", "figure_comparison.png"))
    footer_note = str(comp.get("footer_note", ""))
    _write_comparison_figure(
        out_dir / figure_name,
        records,
        metrics_list,
        cfgs,
        front_origins,
        regressions,
        table_rows,
        experiments,
        invocation_id=invocation_id,
        footer_note=footer_note,
        comp=comp,
    )


def _print_regression_summary(
    experiments: list[dict[str, Any]], regressions: list[dict[str, Any]]
) -> None:
    print("[banding_comparison] Regression summary:", flush=True)
    for entry, reg in zip(experiments, regressions, strict=True):
        label = str(entry["run_label"])
        status = reg.get("status", "SKIP")
        print(f"  {label}: {status}", flush=True)
        for chk in reg.get("checks", []):
            mark = "PASS" if chk.get("pass") else "FAIL"
            print(
                f"    - {chk.get('name')}: {mark} (observed={chk.get('observed')}, "
                f"target={chk.get('target')})",
                flush=True,
            )


def _mass_residual_pct(model: str, diagnostics: dict[str, Any]) -> float | None:
    if model in ("slab_kinetic", "slab_twophase"):
        val = diagnostics.get("mass_residual_pct")
        return float(val) if val is not None else None
    dmb = diagnostics.get("dirichlet_mass_balance", {})
    if isinstance(dmb, dict) and "residual_pct" in dmb:
        return float(dmb["residual_pct"])
    return None


def _run_regression(
    run_label: str,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    entry_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if run_label in _BASELINE_ESTABLISHING:
        return {"status": "n/a", "checks": [], "note": "baseline-establishing"}

    spec = (
        entry_spec
        if isinstance(entry_spec, dict) and entry_spec
        else _REGRESSION_CHECKS.get(run_label)
    )
    if spec is None:
        return {"status": "SKIP", "checks": []}

    checks: list[dict[str, Any]] = []
    all_pass = True

    if "n_transitions" in spec:
        bounds = spec["n_transitions"]
        if isinstance(bounds, list) and len(bounds) == 2:
            lo, hi = float(bounds[0]), float(bounds[1])
            observed = float(diagnostics.get("n_transitions", float("nan")))
            ok = lo <= observed <= hi
            target = f"[{lo}, {hi}]"
        else:
            target = float(spec["n_transitions"]["target"])
            tol = float(spec["n_transitions"]["tol"])
            observed = float(diagnostics.get("n_transitions", float("nan")))
            ok = abs(observed - target) <= tol
            target = f"{target} ± {tol}"
        checks.append(
            {
                "name": "n_transitions",
                "observed": observed,
                "target": target,
                "pass": ok,
            }
        )
        all_pass = all_pass and ok

    if "mean_dwell_period" in spec:
        target = float(spec["mean_dwell_period"]["target"])
        rel_tol = float(spec["mean_dwell_period"]["rel_tol"])
        observed = float(diagnostics.get("mean_dwell_period", float("nan")))
        ok = abs(observed - target) <= rel_tol * target
        checks.append(
            {
                "name": "mean_dwell_period",
                "observed": observed,
                "target": f"{target} ± {rel_tol * 100:.0f}%",
                "pass": ok,
            }
        )
        all_pass = all_pass and ok

    if "n_transitions_max" in spec:
        observed = int(diagnostics.get("n_transitions", 0))
        limit = int(spec["n_transitions_max"])
        ok = observed <= limit
        checks.append(
            {
                "name": "n_transitions_max",
                "observed": observed,
                "target": f"<= {limit}",
                "pass": ok,
            }
        )
        all_pass = all_pass and ok

    if "n_bands_max" in spec:
        observed = int(metrics.get("n_bands", 0))
        limit = int(spec["n_bands_max"])
        ok = observed <= limit
        checks.append(
            {
                "name": "n_bands_max",
                "observed": observed,
                "target": f"<= {limit}",
                "pass": ok,
            }
        )
        all_pass = all_pass and ok

    if "band_count_axial" in spec:
        target = int(spec["band_count_axial"]["target"])
        observed = int(metrics.get("n_peaks", metrics.get("n_bands", 0)))
        ok = observed == target
        checks.append(
            {
                "name": "band_count_axial",
                "observed": observed,
                "target": str(target),
                "pass": ok,
            }
        )
        all_pass = all_pass and ok

    return {"status": "PASS" if all_pass else "FAIL", "checks": checks}


def _json_numpy_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _serialize_record(rec: BandingRecord) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": rec.source,
        "model": rec.model,
        "run_label": rec.run_label,
        "band_positions": np.asarray(rec.band_positions, dtype=np.float64),
        "band_widths": np.asarray(rec.band_widths, dtype=np.float64),
        "band_labels": np.asarray(rec.band_labels, dtype=object),
        "domain_length": float(rec.domain_length),
        "extra_json": json.dumps(rec.extra, default=_json_numpy_default),
    }
    if rec.band_times is not None:
        out["band_times"] = np.asarray(rec.band_times, dtype=np.float64)
    if rec.front_pos_t is not None:
        out["front_pos_t"] = np.asarray(rec.front_pos_t, dtype=np.float64)
    if rec.c_front_t is not None:
        out["c_front_t"] = np.asarray(rec.c_front_t, dtype=np.float64)
    return out


def _load_run_snapshots(run_dir: Path | None) -> dict[str, np.ndarray] | None:
    if run_dir is None:
        return None
    snap_path = Path(run_dir) / "snapshots.npz"
    if not snap_path.is_file():
        return None
    with np.load(snap_path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _load_run_history(run_dir: Path | None) -> dict[str, np.ndarray] | None:
    if run_dir is None:
        return None
    hist_path = Path(run_dir) / "history.npz"
    if not hist_path.is_file():
        return None
    with np.load(hist_path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _downsample_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] <= max_rows:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_rows, dtype=int)
    return arr[idx]


def _downsample_series(t: np.ndarray, *series: np.ndarray, max_pts: int) -> tuple[np.ndarray, ...]:
    t = np.asarray(t, dtype=np.float64).ravel()
    arrs = [np.asarray(s).ravel() for s in series]
    if t.size <= max_pts:
        return (t, *arrs)
    idx = np.linspace(0, t.size - 1, max_pts, dtype=int)
    return (t[idx], *(a[idx] for a in arrs))


def _fmt_law(v: Any, metrics: dict[str, Any]) -> str:
    if metrics.get("laws_reason") == "insufficient_bands":
        return "n/a (n<6)"
    if v is None:
        return "n/a"
    if isinstance(v, float) and not np.isfinite(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _plot_wall_dynamics(
    ax: plt.Axes,
    rec: BandingRecord,
    *,
    c1: float | None,
    c2: float | None,
    history: dict[str, np.ndarray] | None,
    snapshots: dict[str, np.ndarray] | None,
    front_origin: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> None:
    ax.tick_params(labelsize=_FONT_TICK)
    driven = rec.model == "slab_twophase" and cfg is not None and _twophase_is_driven(cfg)
    ax.set_title(
        "prescribed front (no c field)" if driven else _PANEL_TITLES[0],
        fontsize=_FONT_TITLE,
    )

    if rec.model == "slab_kinetic" and history is not None:
        t, c0, st, th = _downsample_series(
            history["t_history"],
            history["c_at_x0_history"],
            history["state_history"],
            history["thickness_history"],
            max_pts=_MAX_WALL_PTS,
        )
        dt_med = float(np.median(np.diff(t))) if t.size > 1 else 1.0
        i = 0
        while i < t.size:
            j = i
            while j + 1 < t.size and int(st[j + 1]) == int(st[i]):
                j += 1
            col = _LABEL_COLORS["H" if int(st[i]) == 1 else "L"]
            ax.axvspan(
                float(t[i]) - 0.5 * dt_med,
                float(t[j]) + 0.5 * dt_med,
                facecolor=col,
                alpha=0.18,
                linewidth=0,
            )
            i = j + 1
        ax.plot(t, c0, color="0.1", lw=1.0, label=r"$c(x=0,t)$")
        if c1 is not None and np.isfinite(c1):
            ax.axhline(c1, color=_LABEL_COLORS["H"], ls="--", lw=0.9, label=f"c1={c1:g}")
        if c2 is not None and np.isfinite(c2):
            ax.axhline(c2, color=_LABEL_COLORS["L"], ls="--", lw=0.9, label=f"c2={c2:g}")
        ax.set_xlabel("time t", fontsize=_FONT_LABEL)
        ax.set_ylabel(r"$c$ at $x=0$", fontsize=_FONT_LABEL)
        ax2 = ax.twinx()
        ax2.plot(t, th, color="#2ca02c", lw=1.0, label="front thickness h(t)")
        ax2.set_ylabel("deposit thickness h(t)", fontsize=_FONT_LABEL, color="#2ca02c")
        ax2.tick_params(axis="y", labelsize=_FONT_TICK, colors="#2ca02c")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=_FONT_TICK, loc="upper right")
        ax.grid(True, alpha=0.25)
        return

    if rec.model == "slab_reactive" and snapshots is not None:
        times = np.asarray(snapshots.get("snapshots_t"), dtype=np.float64).ravel()
        c_mean = snapshots.get("snapshots_c_mean")
        if c_mean is not None and times.size:
            c_mean = np.asarray(c_mean, dtype=np.float64).ravel()
            t, cm = _downsample_series(times, c_mean, max_pts=_MAX_WALL_PTS)
            ax.plot(t, cm, color="0.1", lw=1.0, label="domain-mean c(t)")
        if "snapshots_solid" in snapshots and times.size:
            solid = np.asarray(snapshots["snapshots_solid"], dtype=np.float64)
            domain = float(snapshots.get("domain_L", rec.domain_length))
            n_x = solid.shape[1]
            x = (np.arange(n_x, dtype=np.float64) + 0.5) * (domain / n_x)
            ft, fp = front_position_vs_time(solid, times, x)
            ft, fp = _downsample_series(ft, fp, max_pts=_MAX_WALL_PTS)
            ax2 = ax.twinx()
            ax2.plot(ft, fp, color="#2ca02c", lw=1.0, label="solid front position")
            ax2.set_ylabel("solid front x(t)", fontsize=_FONT_LABEL, color="#2ca02c")
            ax2.tick_params(axis="y", labelsize=_FONT_TICK, colors="#2ca02c")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=_FONT_TICK, loc="upper left")
        ax.set_xlabel("time t", fontsize=_FONT_LABEL)
        ax.set_ylabel("domain-mean c", fontsize=_FONT_LABEL)
        ax.grid(True, alpha=0.25)
        return

    if rec.model == "slab_twophase" and snapshots is not None:
        times = np.asarray(snapshots.get("snapshots_t"), dtype=np.float64).ravel()
        xf_series = snapshots.get("snapshots_x_front")
        if driven and xf_series is not None and times.size:
            xf_arr = np.asarray(xf_series, dtype=np.float64).ravel()
            t, xf = _downsample_series(times, xf_arr, max_pts=_MAX_WALL_PTS)
            ax.plot(t, xf, color="#2ca02c", lw=1.0, label="prescribed front x(t)")
            ax.set_xlabel("time t", fontsize=_FONT_LABEL)
            ax.set_ylabel("solid front x(t)", fontsize=_FONT_LABEL, color="#2ca02c")
            ax.legend(fontsize=_FONT_TICK, loc="upper left")
            ax.grid(True, alpha=0.25)
            return

        cdep = snapshots.get("snapshots_c_dep")
        states = snapshots.get("snapshots_S")
        c1_h = c2_h = None
        if cdep is not None and times.size and not driven:
            cdep_arr = np.asarray(cdep, dtype=np.float64).ravel()
            t, cd = _downsample_series(times, cdep_arr, max_pts=_MAX_WALL_PTS)
            ax.plot(t, cd, color="0.1", lw=1.0, label=r"$c_{\mathrm{dep}}(t)$")
            phys = rec.extra.get("physics") if isinstance(rec.extra.get("physics"), dict) else {}
            c1_h = phys.get("c1")
            c2_h = phys.get("c2")
            if c1_h is None and c1 is not None:
                c1_h = c1
            if c2_h is None and c2 is not None:
                c2_h = c2
            if c1_h is not None and np.isfinite(float(c1_h)):
                ax.axhline(
                    float(c1_h),
                    color=_LABEL_COLORS["H"],
                    ls="--",
                    lw=0.9,
                    label=f"c1={float(c1_h):g}",
                )
            if c2_h is not None and np.isfinite(float(c2_h)):
                ax.axhline(
                    float(c2_h),
                    color=_LABEL_COLORS["L"],
                    ls="--",
                    lw=0.9,
                    label=f"c2={float(c2_h):g}",
                )
            if states is not None and states.size == times.size:
                st = np.asarray(states)
                dt_med = float(np.median(np.diff(t))) if t.size > 1 else 1.0
                for i in range(t.size):
                    col = _LABEL_COLORS["H" if str(st[i]) in ("H", "b'H'") else "L"]
                    ax.axvspan(
                        float(t[i]) - 0.5 * dt_med,
                        float(t[i]) + 0.5 * dt_med,
                        color=col,
                        alpha=0.12,
                    )
        c0_series = snapshots.get("snapshots_c0")
        xf_series = snapshots.get("snapshots_x_front")
        if cdep is None and c0_series is not None and times.size:
            c0_arr = np.asarray(c0_series, dtype=np.float64).ravel()
            t, c0 = _downsample_series(times, c0_arr, max_pts=_MAX_WALL_PTS)
            ax.plot(t, c0, color="0.1", lw=1.0, label=r"$c(x=0,t)$")
        if xf_series is not None and times.size:
            xf_arr = np.asarray(xf_series, dtype=np.float64).ravel()
            t2, xf = _downsample_series(times, xf_arr, max_pts=_MAX_WALL_PTS)
            ax2 = ax.twinx()
            ax2.plot(t2, xf, color="#2ca02c", lw=1.0, label="solid front x(t)")
            ax2.set_ylabel("solid front x(t)", fontsize=_FONT_LABEL, color="#2ca02c")
            ax2.tick_params(axis="y", labelsize=_FONT_TICK, colors="#2ca02c")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=_FONT_TICK, loc="upper left")
        ax.set_xlabel("time t", fontsize=_FONT_LABEL)
        ylab = r"$c_{\mathrm{dep}}(t)$" if cdep is not None else r"$c$ at $x=0$"
        ax.set_ylabel(ylab, fontsize=_FONT_LABEL)
        ax.grid(True, alpha=0.25)
        return

    ax.text(0.5, 0.5, "NO WALL DYNAMICS DATA", ha="center", va="center", color="red")
    ax.axis("off")


def _plot_kymograph(
    ax: plt.Axes,
    rec: BandingRecord,
    snapshots: dict[str, np.ndarray] | None,
) -> None:
    ax.set_title(_PANEL_TITLES[1], fontsize=_FONT_TITLE)
    ax.tick_params(labelsize=_FONT_TICK)

    if snapshots is None:
        ax.text(
            0.5,
            0.5,
            "NO EVOLUTION DATA",
            ha="center",
            va="center",
            color="red",
            fontsize=_FONT_LABEL,
            fontweight="bold",
        )
        ax.axis("off")
        return

    times = np.asarray(snapshots.get("snapshots_t"), dtype=np.float64).ravel()
    x_lo, x_hi = 0.0, float(rec.domain_length)
    if rec.model == "slab_kinetic" and "snapshots_c" in snapshots:
        field = np.asarray(snapshots["snapshots_c"], dtype=np.float64)
        n_x = field.shape[1]
        x_lo, x_hi = 0.0, 1.0
        x_label = "x, solution domain (dimensionless)"
        cbar_label = r"$c(x,t)$"
    elif "snapshots_solid" in snapshots:
        field = np.asarray(snapshots["snapshots_solid"], dtype=np.float64)
        domain = float(snapshots.get("domain_L", rec.domain_length))
        n_x = field.shape[1]  # noqa: F841
        x_lo, x_hi = 0.0, domain
        x_label = "x, distance from left wall"
        cbar_label = r"$\phi_m+\phi_c$ (solid indicator)"
    else:
        ax.text(0.5, 0.5, "NO EVOLUTION DATA", ha="center", va="center", color="red")
        ax.axis("off")
        return

    field, times = _crop_kymograph_time(field, times)
    field = _downsample_rows(field, _MAX_KYMO_ROWS)
    if field.shape[0] != times.size:
        idx = np.linspace(0, times.size - 1, field.shape[0], dtype=int)
        times = times[idx]

    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(x_lo, x_hi, float(times[0]), float(times[-1])),
        cmap="viridis",
    )
    ax.axvline(x_lo, color="white", ls="--", lw=0.8, alpha=0.8)
    ax.set_xlabel(x_label, fontsize=_FONT_LABEL)
    ax.set_ylabel("time t", fontsize=_FONT_LABEL)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=_FONT_TICK)
    cbar.ax.tick_params(labelsize=_FONT_TICK)


def _plot_integrity_overlay(ax: plt.Axes, status: str, reasons: list[str]) -> None:
    if not _status_draws_overlay(status):
        return
    detail = "; ".join(reasons[:2]) if reasons else status.replace("_", " ")
    ax.text(
        0.02,
        0.98,
        f"⚠ {status.replace('_', ' ')}\n{detail}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_FONT_TICK,
        color="#b00020",
        bbox={"facecolor": "#ffe6e6", "edgecolor": "#b00020", "alpha": 0.85, "pad": 2},
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#b00020")
        spine.set_linewidth(1.5)


def _plot_band_strip(
    ax: plt.Axes,
    rec: BandingRecord,
    *,
    row_status: str = "OK",
    invalid_reasons: list[str] | None = None,
) -> None:
    ax.set_title(_PANEL_TITLES[2], fontsize=_FONT_TITLE)
    ax.set_yticks([])
    ax.tick_params(labelsize=_FONT_TICK)

    w = np.asarray(rec.band_widths, dtype=np.float64)
    if w.size == 0:
        if row_status != "OK":
            _plot_invalid_cell(ax, row_status, list(invalid_reasons or []))
        else:
            ax.text(0.5, 0.5, "no bands detected", ha="center", va="center")
            ax.axis("off")
        return

    extent = deposit_extent(rec.band_positions, w)
    if rec.model == "slab_kinetic":
        ax.set_xlim(0.0, max(extent, 1e-9))
        ax.set_xlabel("deposit thickness (dimensionless)", fontsize=_FONT_LABEL)
        left = 0.0
        for label, width in zip(rec.band_labels, w, strict=False):
            color = _LABEL_COLORS.get(label, "#aaaaaa")
            ax.barh(
                0.5, float(width), left=left, height=0.65, color=color, edgecolor="k", linewidth=0.2
            )
            left += float(width)
    else:
        side = str(rec.extra.get("nucleation_side", "left"))
        if side == "right":
            ax.set_xlabel("x, distance from right nucleation wall", fontsize=_FONT_LABEL)
        else:
            ax.set_xlabel("x, distance from nucleation wall (domain units)", fontsize=_FONT_LABEL)
        ax.set_xlim(0.0, max(float(rec.domain_length), extent))
        if rec.source == "label_history":
            left = 0.0
            for label, width in zip(rec.band_labels, w, strict=False):
                lbl = str(label)
                if lbl.startswith("m") or lbl == "H":
                    color = _LABEL_COLORS["m"]
                elif lbl.startswith("c") or lbl == "L":
                    color = _LABEL_COLORS["c"]
                else:
                    color = _LABEL_COLORS.get(lbl, "#aaaaaa")
                ax.barh(
                    0.5,
                    float(width),
                    left=left,
                    height=0.65,
                    color=color,
                    edgecolor="k",
                    linewidth=0.2,
                )
                left += float(width)
        else:
            for pos, width, label in zip(rec.band_positions, w, rec.band_labels, strict=False):
                seg_left = float(pos) - 0.5 * float(width)
                lbl = str(label)
                if lbl.startswith("m") or lbl == "H":
                    color = _LABEL_COLORS["m"]
                elif lbl.startswith("c") or lbl == "L":
                    color = _LABEL_COLORS["c"]
                else:
                    color = _LABEL_COLORS.get(lbl, "#aaaaaa")
                ax.barh(
                    0.5,
                    float(width),
                    left=seg_left,
                    height=0.65,
                    color=color,
                    edgecolor="k",
                    linewidth=0.2,
                )

    _plot_integrity_overlay(ax, row_status, list(invalid_reasons or []))


def _split_axes_1x2(ax: plt.Axes, *, wspace: float = 0.35) -> tuple[plt.Axes, plt.Axes]:
    """Split one axes into two side-by-side panels (GridSpec or add_axes parent)."""
    fig = ax.figure
    subplotspec = ax.get_subplotspec()
    if subplotspec is not None:
        gs = subplotspec.subgridspec(1, 2, wspace=wspace)
        ax_d = fig.add_subplot(gs[0, 0])
        ax_q = fig.add_subplot(gs[0, 1])
        ax.remove()
        return ax_d, ax_q
    pos = ax.get_position()
    ax.remove()
    gap = wspace * pos.width / 2.0
    child_w = (pos.width - gap) / 2.0
    ax_d = fig.add_axes([pos.x0, pos.y0, child_w, pos.height])
    ax_q = fig.add_axes([pos.x0 + child_w + gap, pos.y0, child_w, pos.height])
    return ax_d, ax_q


def _plot_spacing_laws(
    ax: plt.Axes,
    metrics: dict[str, Any],
    *,
    row_status: str = "OK",
    run_label: str = "",
) -> list[plt.Axes]:
    ax.set_title(_PANEL_TITLES[3], fontsize=_FONT_TITLE)
    ax.tick_params(labelsize=_FONT_TICK)

    reasons = list(metrics.get("invalid_reasons") or [])
    n_bands = int(metrics.get("n_bands") or 0)

    if n_bands == 0 and _status_draws_overlay(row_status):
        _plot_invalid_cell(ax, row_status, reasons)
        return [ax]

    if metrics.get("laws_reason") == "insufficient_bands" or n_bands < 6:
        ax.text(
            0.5,
            0.5,
            "n/a (n<6 bands)",
            ha="center",
            va="center",
            fontsize=_FONT_LABEL,
        )
        ax.axis("off")
        return [ax]

    d_sp = np.asarray(metrics.get("d_spacings", []), dtype=np.float64)
    q_rat = np.asarray(metrics.get("Q_ratios", []), dtype=np.float64)
    p_fit = metrics.get("p_fit")
    p_fit_excl = metrics.get("p_fit_excl_last")
    q_cv_tail = metrics.get("Q_cv_tail", metrics.get("Q_cv"))
    q_cv_excl = metrics.get("Q_cv_tail_excl_last")
    alpha = metrics.get("time_law", {}).get("alpha")
    use_excl = (
        _has_terminal_band_spike(metrics)
        and p_fit_excl is not None
        and np.isfinite(float(p_fit_excl))
    )
    p_show = float(p_fit_excl) if use_excl else p_fit
    q_show = q_cv_excl if use_excl else q_cv_tail

    ax_d, ax_q = _split_axes_1x2(ax, wspace=0.35)

    if d_sp.size:
        ax_d.plot(np.arange(1, d_sp.size + 1), d_sp, "o-", ms=3, lw=0.9, color="#1f77b4")
        if d_sp.max() / max(d_sp.min(), 1e-30) > 10.0:
            ax_d.set_yscale("log")
    ax_d.set_xlabel("band index n", fontsize=_FONT_TICK)
    ax_d.set_ylabel(r"spacing $d_n$", fontsize=_FONT_TICK)
    ax_d.tick_params(labelsize=_FONT_TICK)
    ax_d.grid(True, alpha=0.25)

    if q_rat.size:
        idx = np.arange(1, q_rat.size + 1)
        if use_excl and q_rat.size >= 2:
            ax_q.plot(idx[:-1], q_rat[:-1], "o-", ms=3, lw=0.9, color="#ff7f0e")
            ax_q.plot(idx[-1], q_rat[-1], "*", ms=10, color="#b00020", label="terminal band")
            y_cap = max(float(np.max(q_rat[:-1])), float(np.median(q_rat[:-1])) * 3.0)
            ax_q.set_ylim(0.0, y_cap * 1.15)
        else:
            ax_q.plot(idx, q_rat, "o-", ms=3, lw=0.9, color="#ff7f0e")
        if p_show is not None and np.isfinite(float(p_show)):
            ax_q.axhline(1.0 + float(p_show), color="gray", ls="--", lw=0.8)
    ax_q.set_xlabel("band index n", fontsize=_FONT_TICK)
    ax_q.set_ylabel(r"position ratio $Q_n$", fontsize=_FONT_TICK)
    ax_q.tick_params(labelsize=_FONT_TICK)
    ax_q.grid(True, alpha=0.25)
    if use_excl:
        ax_q.legend(fontsize=7, loc="upper right")

    excl_note = "  excl. terminal band" if use_excl else ""
    summary = (
        f"p={_fmt_law(p_show, metrics)}{excl_note}  "
        f"CV(Q)_tail={_fmt_law(q_show, metrics)}  "
        f"alpha_time={_fmt_law(alpha, metrics)}"
    )
    ax_q.text(
        0.02,
        0.98,
        summary,
        transform=ax_q.transAxes,
        va="top",
        ha="left",
        fontsize=_FONT_TICK,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    if _status_draws_overlay(row_status):
        _plot_integrity_overlay(ax_d, row_status, reasons)
    return [ax_d, ax_q]


def _write_comparison_figure(
    path: Path,
    records: list[BandingRecord],
    metrics_list: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    front_origins: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    *,
    invocation_id: str,
    footer_note: str = "",
    comp: dict[str, Any] | None = None,
) -> None:
    from matplotlib.gridspec import GridSpec

    comp = comp or {}
    use_banners = bool(comp.get("figure_banners", False))
    figure_subtitle = str(comp.get("figure_subtitle", ""))
    banner_layout = str(comp.get("figure_banner_layout", "overlay"))

    if use_banners and banner_layout == "dedicated":
        from scripts.comparison_figure_layout import write_comparison_figure_dedicated

        write_comparison_figure_dedicated(
            path,
            records=records,
            metrics_list=metrics_list,
            configs=configs,
            front_origins=front_origins,
            regressions=regressions,
            table_rows=table_rows,
            experiments=experiments,
            invocation_id=invocation_id,
            footer_note=footer_note,
            figure_subtitle=figure_subtitle,
            label_legend=_LABEL_LEGEND,
            panel_titles=_PANEL_TITLES,
            plot_wall_dynamics=_plot_wall_dynamics,
            plot_kymograph=_plot_kymograph,
            plot_band_strip=_plot_band_strip,
            plot_spacing_laws=_plot_spacing_laws,
            load_run_snapshots=_load_run_snapshots,
            load_run_history=_load_run_history,
            supply_regime=_supply_regime,
            fmt_law=_fmt_law,
            status_draws_overlay=_status_draws_overlay,
        )
        return

    n_rows = len(records)
    top_margin = 0.965 if use_banners and figure_subtitle else 0.97
    fig = plt.figure(figsize=(22.0, 4.6 * n_rows + (1.2 if use_banners else 0.8)))
    gs = GridSpec(
        n_rows * 2,
        4,
        figure=fig,
        height_ratios=[1, 0.12] * n_rows,
        hspace=0.55,
        wspace=0.25,
        top=top_margin,
        bottom=0.07,
    )

    if use_banners and figure_subtitle:
        fig.text(
            0.5,
            0.992,
            figure_subtitle,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="normal",
            transform=fig.transFigure,
        )

    for row, (rec, metrics, cfg, fo, reg, trow, entry) in enumerate(
        zip(
            records,
            metrics_list,
            configs,
            front_origins,
            regressions,
            table_rows,
            experiments,
            strict=True,
        )
    ):
        panel_row = row * 2
        caption_row = panel_row + 1
        axes = [fig.add_subplot(gs[panel_row, col]) for col in range(4)]
        cap_ax = fig.add_subplot(gs[caption_row, :])
        cap_ax.axis("off")
        row_status = str(trow.get("row_status", "OK"))

        c1 = c2 = None
        if rec.model == "slab_kinetic":
            phys = cfg.get("physics", {})
            c1 = float(phys.get("c1", float("nan")))
            c2 = float(phys.get("c2", float("nan")))
        elif rec.model == "slab_twophase":
            phys = cfg.get("physics", {})
            if str(phys.get("precipitation", "")).lower() == "hysteresis":
                c1 = float(phys.get("c1", float("nan")))
                c2 = float(phys.get("c2", float("nan")))

        run_dir = metrics.get("run_dir")
        run_path = Path(run_dir) if run_dir else None
        snapshots = _load_run_snapshots(run_path)
        history = _load_run_history(run_path)

        _plot_wall_dynamics(
            axes[0],
            rec,
            c1=c1,
            c2=c2,
            history=history,
            snapshots=snapshots,
            front_origin=fo,
            cfg=cfg,
        )
        _plot_kymograph(axes[1], rec, snapshots)
        _plot_band_strip(
            axes[2],
            rec,
            row_status=row_status,
            invalid_reasons=list(trow.get("invalid_reasons") or []),
        )
        _plot_spacing_laws(
            axes[3], metrics, row_status=row_status, run_label=str(entry.get("run_label", ""))
        )

        supply = _supply_regime(cfg)
        mass_pct = metrics.get("mass_residual_pct")
        mass_str = "n/a" if mass_pct is None else f"{float(mass_pct):.3g}%"
        if row_status not in ("OK",):
            laws_str = f"FLAG ({row_status})"
        elif str(metrics.get("laws_reason") or "").startswith("integrity_"):
            laws_str = f"p={_fmt_law(metrics.get('p_fit'), metrics)} (integrity flag)"
        elif metrics.get("laws_reason") == "insufficient_bands":
            laws_str = "laws: n/a (n<6)"
        elif supply == "closed":
            laws_str = (
                f"p={_fmt_law(metrics.get('p_fit'), metrics)} / "
                f"excl_last={_fmt_law(metrics.get('p_fit_excl_last'), metrics)}"
            )
        else:
            laws_str = f"p={_fmt_law(metrics.get('p_fit'), metrics)}"
        min_pw = metrics.get("min_phi_w")
        min_pw_str = "n/a" if min_pw is None else f"{float(min_pw):.3g}"
        n_proj = metrics.get("n_projection_events")  # noqa: F841
        clip_rate = metrics.get("clip_rate")
        clip_str = "n/a" if clip_rate is None else f"{float(clip_rate):.3e}"
        flux = metrics.get("max_abs_right_flux")
        flux_str = "n/a" if flux is None else f"{float(flux):.3g}"
        e3_note = metrics.get("e3_pair_summary")
        display = str(entry.get("display_label", rec.run_label))
        mech = str(entry.get("mechanism_class", ""))
        ingest_note = metrics.get("ingest_source")
        caption = (
            f"{display} | {rec.model} | {supply} | status={row_status} | "
            f"n_bands={metrics.get('n_bands')} | {laws_str} | "
            f"front_origin={fo.get('origin', 'NONE')} | mass_res={mass_str} | "
            f"min_phi_w={min_pw_str} | clip_rate={clip_str} | flux_R={flux_str} | "
            f"regression={reg.get('status', 'SKIP')}"
        )
        if mech:
            caption = f"{caption} | class={mech}"
        if ingest_note:
            caption = f"{caption} | ingest={ingest_note}"
        if e3_note:
            caption = f"{caption} | {e3_note}"
        cap_ax.text(
            0.0,
            0.5,
            caption,
            fontsize=_FONT_CAPTION - (1 if use_banners else 0),
            family="monospace",
            va="center",
            ha="left",
            transform=cap_ax.transAxes,
            color="0.35" if use_banners else "0.15",
        )

    legend_ax = fig.add_axes([0.08, 0.005, 0.84, 0.025])
    legend_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="k", linewidth=0.3)
        for _lbl, col in _LABEL_LEGEND
    ]
    legend_ax.legend(
        handles,
        [lbl for lbl, _ in _LABEL_LEGEND],
        loc="center",
        ncol=3,
        fontsize=_FONT_TICK,
        frameon=False,
    )

    heaney_rows = [
        i
        for i, entry in enumerate(experiments)
        if _HEANEY_SECTION_TOKEN in str(entry.get("mechanism_class", ""))
    ]
    ch_rows = [
        i
        for i, entry in enumerate(experiments)
        if _HEANEY_SECTION_TOKEN not in str(entry.get("mechanism_class", ""))
    ]

    def _row_panel_top(row_idx: int) -> float:
        return 1.0 - (2 * row_idx) / (n_rows * 2)

    if use_banners:
        if heaney_rows:
            y = _row_panel_top(heaney_rows[0]) + 0.012
            fig.text(
                0.5,
                y,
                _TIER1_HEANEY,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                transform=fig.transFigure,
            )
        if ch_rows:
            y = _row_panel_top(ch_rows[0]) + 0.012
            fig.text(
                0.5,
                y,
                _TIER1_CH,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                transform=fig.transFigure,
            )
        for row_idx, entry in enumerate(experiments):
            vb = entry.get("variant_banner")
            if not vb:
                continue
            y = _row_panel_top(row_idx) + 0.004
            fig.text(
                0.5,
                y,
                str(vb),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                transform=fig.transFigure,
            )
    else:

        def _section_y_center(rows: list[int]) -> float | None:
            if not rows:
                return None
            top = 1.0 - (2 * rows[0]) / (n_rows * 2)
            bot = 1.0 - (2 * rows[-1] + 2) / (n_rows * 2)
            return 0.5 * (top + bot)

        y_heaney = _section_y_center(heaney_rows)
        y_ch = _section_y_center(ch_rows)
        if y_heaney is not None:
            fig.text(
                0.01,
                y_heaney,
                "Heaney model",
                rotation=90,
                va="center",
                ha="center",
                fontsize=13,
                fontweight="bold",
            )
        if y_ch is not None:
            fig.text(
                0.01,
                y_ch,
                "Cahn-Hilliard model",
                rotation=90,
                va="center",
                ha="center",
                fontsize=13,
                fontweight="bold",
            )

    # Legacy rotated side headers only when figure_banners is false (see else branch above).

    footer = f"invocation_id={invocation_id}  (fresh compute — all rows from this timestamp)"
    if footer_note:
        footer = f"{footer}  |  {footer_note}"
    fig.text(
        0.5,
        0.002,
        footer,
        ha="center",
        va="bottom",
        fontsize=_FONT_CAPTION,
        family="monospace",
    )

    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _write_comparison_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_label",
        "model",
        "supply",
        "row_status",
        "n_bands",
        "p_fit",
        "CV(Q)_tail",
        "p_fit_excl_last",
        "CV(Q)_tail_excl_last",
        "front_origin",
        "alpha_time",
        "mass_residual_pct",
        "min_phi_w",
        "clip_rate",
        "n_projection_events",
        "max_abs_right_flux",
        "regression",
    ]

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if not np.isfinite(v):
                return "n/a"
            return f"{v:.4g}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_movingfront_display_labels(
    experiments: list[dict[str, Any]], comp: dict[str, Any], repo_root: Path
) -> None:
    """Set E9 moving-front caption and variant banner from D3H/D5H verdict."""
    for ingest_key, verdict_key in (
        ("d2h_best", "d2_verdict_path"),
        ("d3h_best", "d3_verdict_path"),
        ("d5h_best", "d5_verdict_path"),
        ("d6h_best", "d6_verdict_path"),
    ):
        vp = comp.get(verdict_key)
        if not vp:
            continue
        vp_path = Path(str(vp))
        vp_path = vp_path if vp_path.is_absolute() else repo_root / vp_path
        if not vp_path.is_file():
            continue
        verdict = json.loads(vp_path.read_text(encoding="utf-8"))
        best = str(verdict.get("master_comparison_run", "moving_front_best"))
        any_osc = bool(verdict.get("any_oscillates", False))
        root_cause = str(verdict.get("root_cause_verdict", ""))
        for entry in experiments:
            if str(entry.get("ingest", "")) == ingest_key:
                if any_osc:
                    entry["display_label"] = f"{best} (Heaney moving front — oscillates)"
                    entry["variant_banner"] = "Moving front (this work) | oscillates"
                elif root_cause:
                    entry["display_label"] = f"{best} — no sustained oscillation ({root_cause})"
                    entry["variant_banner"] = "Moving front (this work) | honest negative"
                else:
                    entry["display_label"] = f"{best} — no sustained oscillation"
                    entry["variant_banner"] = "Moving front (this work) | honest negative"


def _assert_fresh_invocation(
    invocation_id: str,
    experiments: list[dict[str, Any]],
    metrics_all: dict[str, Any],
) -> None:
    """Hard-fail before figure export if any row summary lacks the current invocation_id."""
    mismatches: list[str] = []
    for entry in experiments:
        label = str(entry["run_label"])
        if _is_ingest_entry(entry):
            continue
        run_dir = metrics_all.get(label, {}).get("run_dir")
        if not run_dir:
            if label == _E3_HEANEY_CLOSED:
                continue
            mismatches.append(f"{label}: missing run_dir in metrics")
            continue
        summary_path = Path(str(run_dir)) / "summary.json"
        if not summary_path.is_file():
            mismatches.append(f"{label}: no summary.json at {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        got = summary.get("invocation_id")
        if got != invocation_id:
            mismatches.append(f"{label}: invocation_id={got!r} (expected {invocation_id!r})")
    if mismatches:
        raise RuntimeError(
            "Fresh-compute integrity check failed — stale or mixed invocation:\n  "
            + "\n  ".join(mismatches)
        )


def _complete_runs_under(root: Path, invocation_id: str) -> list[Path]:
    """Return complete run dirs under ``root`` whose summary matches ``invocation_id``."""
    found: list[Path] = []
    if not root.is_dir():
        return found
    for summary_path in root.rglob("summary.json"):
        run_dir = summary_path.parent
        if not is_run_complete(run_dir):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("invocation_id") != invocation_id:
            continue
        found.append(run_dir)
    return found


def _newest_complete_run(root: Path, invocation_id: str) -> Path:
    runs = _complete_runs_under(root, invocation_id)
    if not runs:
        raise FileNotFoundError(
            f"no complete run with invocation_id={invocation_id!r} under {root}"
        )
    return max(runs, key=lambda p: p.name)


def _load_e3_heaney_closed_from_disk(
    entry: dict[str, Any],
    out_dir: Path,
    invocation_id: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Pick the best E3 pair run already on disk (same logic as live sweep)."""
    pairs_raw = entry.get("e3_pairs", _E3_DEFAULT_PAIRS)
    if not isinstance(pairs_raw, list) or not pairs_raw:
        pairs_raw = _E3_DEFAULT_PAIRS

    pair_results: list[tuple[str, float, Any, dict[str, Any]]] = []
    e3_root = _find_e3_closed_root(out_dir)

    for pair in pairs_raw:
        if not isinstance(pair, dict):
            continue
        label = str(pair.get("label", f"{pair.get('c1')}/{pair.get('c2')}"))
        safe_label = label.replace("/", "_")
        run_dir = _newest_complete_run(e3_root / safe_label, invocation_id)
        result = load_sim_result_from_run_dir(run_dir, "slab_kinetic")
        cfg = result.config_resolved if isinstance(result.config_resolved, dict) else {}
        diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
        n_tr = float(diag.get("n_transitions", 0))
        pair_results.append((label, n_tr, result, cfg))

    if not pair_results:
        raise FileNotFoundError(f"{_E3_HEANEY_CLOSED}: no pair runs under {e3_root}")

    best_label, best_n, best_result, best_cfg = max(pair_results, key=lambda item: item[1])
    summary_parts = [f"{lbl}→{int(n) if n == int(n) else n:g}" for lbl, n, _, _ in pair_results]
    pair_summary = (
        f"E3 pairs: {', '.join(summary_parts)} "
        f"(shown: {best_label}→{int(best_n) if best_n == int(best_n) else best_n:g})"
    )
    extra_meta = {
        "e3_pair_summary": pair_summary,
        "e3_pair_counts": {lbl: n for lbl, n, _, _ in pair_results},
        "e3_shown_pair": best_label,
    }
    return best_result, best_cfg, extra_meta


def _load_comparison_entry_from_disk(
    entry: dict[str, Any],
    out_dir: Path,
    invocation_id: str,
    *,
    repo_root: Path,
    comp: dict[str, Any],
    completed_run_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    run_label = str(entry["run_label"])
    model = str(entry["model"])
    if _is_ingest_entry(entry):
        run_dir = (
            completed_run_dir
            if completed_run_dir is not None
            else _resolve_ingest_run_dir(entry, comp, repo_root)
        )
        cfg = load_run_config(run_dir / "config.yaml")
        result = load_sim_result_from_run_dir(run_dir, model)
        return result, cfg, {}
    if run_label == _E3_HEANEY_CLOSED and model == "slab_kinetic":
        return _load_e3_heaney_closed_from_disk(entry, out_dir, invocation_id)
    run_dir = (
        completed_run_dir
        if completed_run_dir is not None
        else _newest_complete_run_for_label(out_dir, run_label, invocation_id)
    )
    result = load_sim_result_from_run_dir(run_dir, model)
    cfg = result.config_resolved if isinstance(result.config_resolved, dict) else {}
    return result, cfg, {}


def finalize_comparison_outputs(out_dir: Path) -> Path:
    """Write metrics/table/figure from completed runs when export failed after compute."""
    out_dir = out_dir.resolve()
    config_path = out_dir / "comparison_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing comparison config: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "comparison" not in raw:
        raise ValueError("config must contain a top-level 'comparison' block")

    comp = raw["comparison"]
    experiments = comp.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("comparison.experiments must be a nonempty list")

    _patch_movingfront_display_labels(experiments, comp, _REPO)

    invocation_id = out_dir.name
    records: list[BandingRecord] = []
    metrics_all: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    cfgs: list[dict[str, Any]] = []
    front_origins: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    wall_s = 0.0

    print(
        f"[banding_comparison] Finalize outputs invocation_id={invocation_id} -> {out_dir}",
        flush=True,
    )

    for entry in experiments:
        result, cfg, e3_extra = _load_comparison_entry_from_disk(
            entry,
            out_dir,
            invocation_id,
            repo_root=_REPO,
            comp=comp,
        )
        _, cfg, rec, row, entry_metrics, fo, regression = _build_comparison_row(
            entry, result, cfg, e3_extra
        )
        if _is_ingest_entry(entry):
            run_dir = _resolve_ingest_run_dir(entry, comp, _REPO)
            entry_metrics["run_dir"] = str(run_dir)
            entry_metrics["ingest_source"] = str(run_dir)
        diag = entry_metrics.get("diagnostics", {})
        if isinstance(diag, dict) and "wall_time_s" in diag:
            wall_s += float(diag["wall_time_s"])

        cfgs.append(cfg)
        table_rows.append(row)
        metrics_all[str(entry["run_label"])] = entry_metrics
        records.append(rec)
        front_origins.append(fo)
        regressions.append(regression)

        print(
            f"  {entry['run_label']}: status={row['row_status']} n_bands={row['n_bands']} "
            f"front_origin={fo.get('origin')} regression={regression['status']}",
            flush=True,
        )

    _assert_fresh_invocation(invocation_id, experiments, metrics_all)
    _write_comparison_outputs(
        out_dir,
        comp=comp,
        config_stem=config_path.stem,
        experiments=experiments,
        records=records,
        table_rows=table_rows,
        metrics_all=metrics_all,
        cfgs=cfgs,
        front_origins=front_origins,
        regressions=regressions,
        wall_s=wall_s,
        invocation_id=invocation_id,
    )
    print(f"[banding_comparison] Finalized -> {out_dir}", flush=True)
    _print_regression_summary(experiments, regressions)
    return out_dir


def _execute_comparison_rows(
    *,
    config_path: Path,
    out_dir: Path,
    comp: dict[str, Any],
    experiments: list[dict[str, Any]],
    invocation_id: str,
    repo_root: Path,
    resume: bool = False,
) -> Path:
    """Run or resume all comparison rows, checkpointing after each."""
    _preflight_comparison_experiments(experiments, repo_root, comp)
    _patch_movingfront_display_labels(experiments, comp, repo_root)

    progress = load_comparison_progress(out_dir) if resume else {"completed": {}, "failed": {}}
    completed = progress.get("completed", {})

    records: list[BandingRecord] = []
    metrics_all: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    cfgs: list[dict[str, Any]] = []
    front_origins: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for entry in tqdm(experiments, desc="banding_comparison", unit="run"):
        label = str(entry["run_label"])
        try:
            if resume and label in completed:
                if label == _E3_HEANEY_CLOSED and str(entry["model"]) == "slab_kinetic":
                    result, cfg, e3_extra = _load_e3_heaney_closed_from_disk(
                        entry, out_dir, invocation_id
                    )
                else:
                    saved_dir = Path(str(completed[label]["run_dir"]))
                    result, cfg, e3_extra = _load_comparison_entry_from_disk(
                        entry,
                        out_dir,
                        invocation_id,
                        repo_root=repo_root,
                        comp=comp,
                        completed_run_dir=saved_dir,
                    )
                _, cfg, rec, row, entry_metrics, fo, regression = _build_comparison_row(
                    entry, result, cfg, e3_extra
                )
                if _is_ingest_entry(entry):
                    entry_metrics["run_dir"] = str(saved_dir)
                    entry_metrics["ingest_source"] = str(saved_dir)
                tqdm.write(f"  {label}: resumed from checkpoint")
            else:
                result, cfg, rec, row, entry_metrics, fo, regression = _process_comparison_entry(
                    entry,
                    repo_root,
                    out_dir,
                    show_progress=True,
                    invocation_id=invocation_id,
                    comp=comp,
                )
                run_dir = entry_metrics.get("run_dir")
                if run_dir:
                    diag = entry_metrics.get("diagnostics", {})
                    wall_row = (
                        _safe_float(diag.get("wall_time_s"), 0.0) if isinstance(diag, dict) else 0.0
                    )
                    mark_comparison_run_complete(
                        out_dir,
                        run_label=label,
                        run_dir=Path(str(run_dir)),
                        model=str(entry["model"]),
                        wall_time_s=wall_row,
                    )
        except Exception as exc:
            mark_comparison_run_failed(out_dir, run_label=label, error=str(exc))
            raise

        cfgs.append(cfg)
        table_rows.append(row)
        metrics_all[label] = entry_metrics
        records.append(rec)
        front_origins.append(fo)
        regressions.append(regression)

        partial_wall = time.perf_counter() - t0
        _write_metrics_checkpoint(
            out_dir,
            comp=comp,
            config_stem=config_path.stem,
            table_rows=table_rows,
            metrics_all=metrics_all,
            invocation_id=invocation_id,
            wall_s=partial_wall,
            completed=len(table_rows),
            total=len(experiments),
        )

        if not (resume and label in completed):
            tqdm.write(
                f"  {label}: status={row['row_status']} n_bands={row['n_bands']} "
                f"front_origin={fo.get('origin')} regression={regression['status']}"
            )

    _assert_fresh_invocation(invocation_id, experiments, metrics_all)

    wall_s = time.perf_counter() - t0
    try:
        _write_comparison_outputs(
            out_dir,
            comp=comp,
            config_stem=config_path.stem,
            experiments=experiments,
            records=records,
            table_rows=table_rows,
            metrics_all=metrics_all,
            cfgs=cfgs,
            front_origins=front_origins,
            regressions=regressions,
            wall_s=wall_s,
            invocation_id=invocation_id,
        )
    except Exception as export_exc:
        print(
            f"[banding_comparison] Final export failed ({export_exc}); "
            f"rebuilding figure/npz from runs/ ...",
            flush=True,
        )
        finalize_comparison_outputs(out_dir)

    print(f"[banding_comparison] Finished in {wall_s:.1f}s -> {out_dir}", flush=True)
    _print_regression_summary(experiments, regressions)
    return out_dir


def run_comparison(config_path: Path, output_root: Path) -> Path:
    """Execute a banding comparison (fresh compute only; invocation_id assertion enforced)."""
    repo_root = _REPO
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    invocation_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / invocation_id
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out_dir / "comparison_config.yaml")
    print(
        f"[banding_comparison] Fresh run invocation_id={invocation_id} -> {out_dir}",
        flush=True,
    )

    print(
        f"[banding_comparison] Recovery (no re-run): "
        f"uv run python scripts/run_banding_comparison.py --finalize-outputs {out_dir}",
        flush=True,
    )

    if not isinstance(raw, dict) or "comparison" not in raw:
        raise ValueError("config must contain a top-level 'comparison' block")

    comp = raw["comparison"]
    experiments = comp.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("comparison.experiments must be a nonempty list")

    return _execute_comparison_rows(
        config_path=config_path,
        out_dir=out_dir,
        comp=comp,
        experiments=experiments,
        invocation_id=invocation_id,
        repo_root=repo_root,
        resume=False,
    )


def continue_comparison(out_dir: Path) -> Path:
    """Resume an interrupted comparison in the same output directory."""
    out_dir = out_dir.resolve()
    config_path = out_dir / "comparison_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing comparison config: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "comparison" not in raw:
        raise ValueError("config must contain a top-level 'comparison' block")

    comp = raw["comparison"]
    experiments = comp.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("comparison.experiments must be a nonempty list")

    invocation_id = out_dir.name
    progress = load_comparison_progress(out_dir)
    n_done = len(progress.get("completed", {}))
    print(
        f"[banding_comparison] Continue invocation_id={invocation_id} "
        f"({n_done}/{len(experiments)} rows checkpointed) -> {out_dir}",
        flush=True,
    )
    print(
        f"[banding_comparison] Recovery (export only): "
        f"uv run python scripts/run_banding_comparison.py --finalize-outputs {out_dir}",
        flush=True,
    )
    return _execute_comparison_rows(
        config_path=config_path,
        out_dir=out_dir,
        comp=comp,
        experiments=experiments,
        invocation_id=invocation_id,
        repo_root=_REPO,
        resume=True,
    )


def _record_from_npz_prefix(z: np.lib.npyio.NpzFile, prefix: str) -> BandingRecord:
    extra = json.loads(str(z[f"{prefix}/extra_json"]))
    labels = [str(x) for x in np.asarray(z[f"{prefix}/band_labels"]).ravel()]
    bt_key = f"{prefix}/band_times"
    fp_key = f"{prefix}/front_pos_t"
    cf_key = f"{prefix}/c_front_t"
    return BandingRecord(
        source=str(z[f"{prefix}/source"]),
        model=str(z[f"{prefix}/model"]),
        run_label=str(z[f"{prefix}/run_label"]),
        band_positions=np.asarray(z[f"{prefix}/band_positions"], dtype=np.float64),
        band_widths=np.asarray(z[f"{prefix}/band_widths"], dtype=np.float64),
        band_labels=labels,
        band_times=np.asarray(z[bt_key], dtype=np.float64) if bt_key in z.files else None,
        front_pos_t=np.asarray(z[fp_key], dtype=np.float64) if fp_key in z.files else None,
        c_front_t=np.asarray(z[cf_key], dtype=np.float64) if cf_key in z.files else None,
        domain_length=float(z[f"{prefix}/domain_length"]),
        extra=extra,
    )


def _config_for_figure_row(
    entry: dict[str, Any],
    metrics: dict[str, Any],
    *,
    comp: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Load resolved run config from disk (ingest rows have no base_config in YAML)."""
    run_dir = metrics.get("run_dir")
    if run_dir:
        cfg_path = Path(str(run_dir)) / "config.yaml"
        if cfg_path.is_file():
            return load_run_config(cfg_path)
    if _is_ingest_entry(entry):
        ingest_dir = _resolve_ingest_run_dir(entry, comp, repo_root)
        return load_run_config(ingest_dir / "config.yaml")
    return _build_run_config(entry, repo_root)


def _load_comparison_bundle(
    out_dir: Path,
    *,
    comp: dict[str, Any] | None = None,
) -> tuple[
    list[BandingRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config_path = out_dir / "comparison_config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    comp = comp or raw.get("comparison", {})
    experiments = comp.get("experiments", [])
    metrics_doc = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics_by_label = metrics_doc.get("experiments", {})

    records: list[BandingRecord] = []
    metrics_list: list[dict[str, Any]] = []
    cfgs: list[dict[str, Any]] = []
    front_origins: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    with np.load(out_dir / "records.npz", allow_pickle=True) as z:
        for entry in experiments:
            label = str(entry["run_label"])
            records.append(_record_from_npz_prefix(z, label))
            m = dict(metrics_by_label[label])
            metrics_list.append(m)
            cfgs.append(_config_for_figure_row(entry, m, comp=comp, repo_root=_REPO))
            front_origins.append(dict(m.get("front_origin", {})))
            regressions.append(dict(m.get("regression", {})))
    return records, metrics_list, cfgs, front_origins, regressions


def _load_table_rows(out_dir: Path, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_path = out_dir / "table_comparison.md"
    if not table_path.is_file():
        return [{"run_label": str(e["run_label"]), "row_status": "OK"} for e in experiments]
    metrics_doc = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in experiments:
        label = str(entry["run_label"])
        m = dict(metrics_doc.get("experiments", {}).get(label, {}))
        rows.append(
            {
                "run_label": label,
                "row_status": m.get("row_status", "OK"),
                "invalid_reasons": m.get("invalid_reasons", []),
                "n_bands": m.get("n_bands"),
            }
        )
    return rows


def regenerate_comparison_figure(out_dir: Path) -> Path:
    """Rebuild comparison figure from saved comparison artifacts (no compute)."""
    out_dir = out_dir.resolve()
    config_path = out_dir / "comparison_config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    comp = raw.get("comparison", {})
    experiments = comp.get("experiments", [])
    _patch_movingfront_display_labels(experiments, comp, _REPO)

    records, metrics_list, cfgs, front_origins, regressions = _load_comparison_bundle(
        out_dir, comp=comp
    )
    metrics_doc = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    invocation_id = str(metrics_doc.get("invocation_id", "unknown"))
    table_rows = _load_table_rows(out_dir, experiments)
    figure_name = str(comp.get("figure_out", "figure_comparison.png"))
    fig_path = out_dir / figure_name
    _write_comparison_figure(
        fig_path,
        records,
        metrics_list,
        cfgs,
        front_origins,
        regressions,
        table_rows,
        experiments,
        invocation_id=invocation_id,
        footer_note=str(comp.get("footer_note", "")),
        comp=comp,
    )
    print(f"[banding_comparison] Wrote {fig_path}", flush=True)
    return fig_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-model banding comparison with checkpoint/recovery."
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=None,
        help="Comparison YAML config (not needed with --regenerate-figure).",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=None,
        help="Output root directory for a new comparison run.",
    )
    parser.add_argument(
        "--regenerate-figure",
        action="store_true",
        help="Rebuild figure_comparison.png from an existing comparison directory.",
    )
    parser.add_argument(
        "--finalize-outputs",
        action="store_true",
        help=(
            "Write metrics.json, table, and figure from completed runs under a comparison "
            "directory (does not re-run simulations)."
        ),
    )
    parser.add_argument(
        "--continue-out-dir",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted comparison in an existing timestamp directory "
            "(skips rows recorded in comparison_progress.json)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.continue_out_dir is not None:
            out_dir = (
                args.continue_out_dir
                if args.continue_out_dir.is_absolute()
                else _REPO / args.continue_out_dir
            )
            continue_comparison(out_dir)
            return 0

        if args.finalize_outputs:
            if args.config is None and args.output_root is None:
                parser.error("provide the comparison output directory as config or output_root")
            out_dir = args.output_root if args.output_root is not None else args.config
            out_dir = out_dir if out_dir.is_absolute() else _REPO / out_dir
            finalize_comparison_outputs(out_dir)
            return 0

        if args.regenerate_figure:
            if args.config is None and args.output_root is None:
                parser.error("provide the comparison output directory as config or output_root")
            out_dir = args.output_root if args.output_root is not None else args.config
            out_dir = out_dir if out_dir.is_absolute() else _REPO / out_dir
            regenerate_comparison_figure(out_dir)
            return 0

        if args.config is None:
            parser.error("config is required unless --regenerate-figure")
        config_path = args.config if args.config.is_absolute() else _REPO / args.config
        output_root = (
            args.output_root if args.output_root is not None else Path("results/banding_comparison")
        )
        output_root = output_root if output_root.is_absolute() else _REPO / output_root
        run_comparison(config_path, output_root)
    except Exception as exc:
        print(f"banding comparison failed: {exc}", file=sys.stderr)
        print(
            "If some rows finished, continue without re-running them:",
            file=sys.stderr,
        )
        print(
            "  uv run python scripts/run_banding_comparison.py "
            "--continue-out-dir results/master_comparison/<timestamp>",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
