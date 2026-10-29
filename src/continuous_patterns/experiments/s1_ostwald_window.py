"""S1 Ostwald-window sweep analysis for ``slab_twophase`` CH runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from continuous_patterns.core.banding_metrics import (
    alternating_zone_metrics,
    band_formation_times,
    front_velocity_series,
    interpolate_series,
    record_from_slab_twophase_result,
    s1_twophase_validity,
)
from continuous_patterns.core.io import load_run_config
from continuous_patterns.core.types import SimResult
from continuous_patterns.experiments.checkpoint import load_sim_result_from_run_dir


def _load_twophase_result(run_dir: Path) -> tuple[SimResult, dict[str, Any]]:
    cfg = load_run_config(run_dir / "config.yaml")
    result = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    return result, cfg


def analyze_s1_run(run_dir: Path, *, run_label: str | None = None) -> dict[str, Any]:
    """Compute S1 zone metrics and trajectories; merge into ``summary.json``."""
    run_dir = run_dir.resolve()
    result, cfg = _load_twophase_result(run_dir)
    label = run_label or str(cfg.get("experiment", {}).get("name", run_dir.name))
    rec = record_from_slab_twophase_result(result, label)

    zone = alternating_zone_metrics(rec.band_positions, rec.band_widths, rec.band_labels)
    diag = dict(result.diagnostics) if isinstance(result.diagnostics, dict) else {}
    validity = s1_twophase_validity(diag)

    snap_t = np.asarray(rec.extra.get("snapshots_t", []), dtype=np.float64)
    snap_cf = np.asarray(rec.extra.get("snapshots_c_front", []), dtype=np.float64)
    snap_xf = np.asarray(rec.extra.get("snapshots_x_front", []), dtype=np.float64)

    if snap_t.size and snap_cf.size == snap_t.size:
        c_front_t = np.column_stack([snap_t, snap_cf])
    else:
        c_front_t = None

    t_v, v_front = front_velocity_series(snap_t, snap_xf)
    v_front_t = np.column_stack([t_v, v_front]) if t_v.size else None

    zone_pos = np.asarray(zone.get("zone_band_positions", []), dtype=np.float64)
    form_t = band_formation_times(zone_pos, snap_t, snap_xf) if zone_pos.size else np.asarray([])
    spacing_pacing: list[dict[str, float]] = []
    if zone_pos.size >= 2 and form_t.size >= 2 and t_v.size:
        d_sp = np.diff(zone_pos)
        t_pair = 0.5 * (form_t[:-1] + form_t[1:])
        v_at = interpolate_series(t_pair, t_v, v_front)
        for i, d_val in enumerate(d_sp):
            spacing_pacing.append(
                {
                    "d_n": float(d_val),
                    "v_front": float(v_at[i]) if i < v_at.size else float("nan"),
                    "t_form": float(t_pair[i]) if i < t_pair.size else float("nan"),
                }
            )

    phys = cfg.get("physics") or {}
    payload: dict[str, Any] = {
        **diag,
        "run_label": label,
        "c_ostwald": float(phys.get("c_ostwald", float("nan"))),
        "w_ostwald": float(phys.get("w_ostwald", float("nan"))),
        "gamma": float(phys.get("gamma", float("nan"))),
        "n_bands": int(rec.band_positions.size),
        "n_bands_zone": int(zone["n_bands_zone"]),
        "band_zone_extent": float(zone["band_zone_extent"]),
        "mean_spacing_zone": float(zone["mean_spacing_zone"]),
        "median_spacing_zone": float(zone["median_spacing_zone"]),
        "mean_spacing_zone_excl_last": float(zone["mean_spacing_zone_excl_last"]),
        "std_spacing_zone": float(zone["std_spacing_zone"]),
        "row_validity": validity,
        "spacing_pacing": spacing_pacing,
    }
    if c_front_t is not None:
        payload["c_front_t"] = c_front_t.tolist()
    if v_front_t is not None:
        payload["v_front_t"] = v_front_t.tolist()

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def check_s1_anchor(entry: dict[str, Any]) -> tuple[bool, str]:
    """Regression anchor: ``S1_cost0p25_g4`` should match E8 band-zone metrics."""
    if str(entry.get("run_label")) != "S1_cost0p25_g4":
        return True, "SKIP (not anchor run)"
    if entry.get("status") != "success":
        return False, "FAIL anchor: run did not succeed"
    if entry.get("row_validity") != "OK":
        return False, f"FAIL anchor: validity={entry.get('row_validity')}"
    n_zone = int(entry.get("n_bands_zone", 0))
    extent = float(entry.get("band_zone_extent", float("nan")))
    ok_n = 4 <= n_zone <= 6
    ok_ext = 30.0 <= extent <= 45.0
    if ok_n and ok_ext:
        return True, f"PASS anchor: n_bands_zone={n_zone} band_zone_extent={extent:.2g}"
    return False, (
        f"FAIL anchor: n_bands_zone={n_zone} (expected 4-6), "
        f"band_zone_extent={extent:.2g} (expected 30-45)"
    )
