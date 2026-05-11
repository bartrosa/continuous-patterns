"""Run a generic axes-based Cartesian sweep."""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

from continuous_patterns.core.io import load_run_config
from continuous_patterns.experiments.run import run_one

_SCRIPTS_DIR = Path(__file__).resolve().parent


def analytic_strip_predict(Da: float, kappa: float, c1: float, c2: float) -> bool:
    """Strict analytic strip (open rectangle): lo < c2 < c1 < hi."""
    da = float(Da)
    ka = float(kappa)
    lo = 1.0 / (1.0 + ka * da)
    hi = 1.0 / (1.0 + da)
    fc1 = float(c1)
    fc2 = float(c2)
    return (fc2 > lo) and (fc1 < hi) and (fc1 > fc2)


def _diag_optional_float(diag: dict[str, Any], key: str) -> float | None:
    v = diag.get(key)
    if v is None:
        return None
    return float(v)


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur: dict[str, Any] = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _slug_value(v: Any) -> str:
    if isinstance(v, dict):
        parts: list[str] = []
        for k in sorted(v.keys(), key=lambda x: str(x)):
            parts.append(f"{k}_{_slug_value(v[k])}")
        return "__".join(parts)
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isfinite(v):
            s = f"{v:.6g}".replace("-", "m").replace(".", "p")
            return s
        return str(v).lower()
    return str(v).replace(" ", "_")


def _combo_name(combo: dict[str, Any]) -> str:
    parts = [f"{k}_{_slug_value(v)}" for k, v in combo.items()]
    return "__".join(parts)


def _apply_axis_to_cfg(
    cfg: dict[str, Any],
    axis_defs: list[dict[str, Any]],
    combo: dict[str, Any],
) -> None:
    """Apply sweep axes (``path``, ``merge_into``, ``param_overrides``) to ``cfg`` in-place."""
    for ax in axis_defs:
        ax_name = str(ax["name"])
        ax_value = combo[ax_name]
        merge_into = ax.get("merge_into")
        ax_path = ax.get("path")
        if merge_into:
            if not isinstance(ax_value, dict):
                raise ValueError(
                    f"axis {ax_name!r} uses merge_into but value is not a dict: {ax_value!r}"
                )
            tgt = cfg.setdefault(str(merge_into), {})
            if not isinstance(tgt, dict):
                raise ValueError(f"merge_into target {merge_into!r} must be a mapping")
            for kk, vv in ax_value.items():
                tgt[str(kk)] = copy.deepcopy(vv)
        elif isinstance(ax_path, str) and ax_path.strip():
            _set_dotted(cfg, ax_path, copy.deepcopy(ax_value))
        pov = ax.get("param_overrides")
        if isinstance(pov, dict):
            override_for_value = pov.get(ax_value)
            if override_for_value is None:
                override_for_value = pov.get(str(ax_value))
            if override_for_value is not None:
                if not isinstance(override_for_value, dict):
                    raise ValueError(
                        f"param_overrides for axis {ax_name!r} value {ax_value!r} must be a mapping"
                    )
                for dotted_key, v in override_for_value.items():
                    _set_dotted(cfg, dotted_key, copy.deepcopy(v))


def _front_origin_from_h5_or_final(
    run_dir: Path, *, rim_left_width_px: int, rim_width_px: int
) -> str:
    h5_path = run_dir / "snapshots.h5"
    phi_total: Any = None
    if h5_path.is_file():
        import h5py

        with h5py.File(h5_path, "r") as h5:
            keys = sorted([k for k in h5.keys() if k.startswith("snap_")])
            if keys:
                k_mid = keys[len(keys) // 2]
                g = h5[k_mid]
                phi_m = g["phi_m"][()]
                phi_c = g["phi_c"][()]
                phi_total = phi_m + phi_c
    if phi_total is None:
        arr = np.load(run_dir / "final_state.npz")
        phi_total = arr["phi_m"] + arr["phi_c"]

    n = int(phi_total.shape[0])
    left_hi = min(n, int(rim_left_width_px) + 10)
    right_lo = max(0, n - int(rim_width_px) - 10)
    left_mass = float(np.mean(phi_total[:left_hi, :]))
    right_mass = float(np.mean(phi_total[right_lo:, :]))
    if left_mass > right_mass + 0.1:
        return "LEFT"
    if right_mass > left_mass + 0.1:
        return "RIGHT"
    return "BOTH"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            "Usage: uv run python scripts/run_sweep_axes.py "
            "<sweep_yaml> <out_dir> [--chunk-size N]",
            file=sys.stderr,
        )
        return 2

    sweep_path = Path(args[0])
    out_root = Path(args[1])
    chunk_size = 2000
    if len(args) >= 4 and args[2] == "--chunk-size":
        chunk_size = int(args[3])

    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep config must be a mapping")
    sweep = raw.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("missing sweep section")
    axes = sweep.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("expected non-empty list in sweep.axes")

    base_cfg = load_run_config(Path(sweep["base_config"]))
    name = str(sweep.get("name", "sweep_axes"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    nested = bool(sweep.get("nested_results", False))
    sweep_dir = (out_root / stamp) if nested else out_root
    sweep_dir.mkdir(parents=True, exist_ok=True)

    overrides = raw.get("overrides", {})
    if isinstance(overrides, dict):
        for dotted_key, v in overrides.items():
            _set_dotted(base_cfg, dotted_key, copy.deepcopy(v))
    axis_defs: list[dict[str, Any]] = []
    for ax in axes:
        if not isinstance(ax, dict):
            raise ValueError("each axis must be a mapping")
        name_ax = str(ax.get("name", "")).strip()
        vals = list(ax.get("values", []))
        if not name_ax or not vals:
            raise ValueError("axis requires non-empty name and values")
        axis_defs.append(ax)

    entries: list[dict[str, Any]] = []
    axis_names = [str(ax["name"]) for ax in axis_defs]
    combos = [
        dict(zip(axis_names, vals, strict=True))
        for vals in itertools.product(*[ax["values"] for ax in axis_defs])
    ]

    total_wall = 0.0
    for combo in tqdm(combos, total=len(combos), desc=f"Sweep {name}", unit="run", leave=True):
        cfg = copy.deepcopy(base_cfg)
        run_label = _combo_name(combo)
        cfg["experiment"]["name"] = run_label
        _apply_axis_to_cfg(cfg, axis_defs, combo)

        exp_m = str(cfg.get("experiment", {}).get("model", ""))
        phys = cfg.get("physics", {})
        c1v = float(phys.get("c1", float("nan")))
        c2v = float(phys.get("c2", float("nan")))
        Da_v = float(phys.get("Da", float("nan")))
        ka_v = float(phys.get("kappa", float("nan")))
        Tv = float(cfg.get("time", {}).get("T", float("nan")))

        if exp_m == "slab_kinetic" and not (c1v > c2v):
            osc_p = analytic_strip_predict(Da_v, ka_v, c1v, c2v)
            entries.append(
                {
                    "run_label": run_label,
                    "parameters": combo,
                    "status": "skipped",
                    "relative_path": None,
                    "band_count_axial": None,
                    "residual_pct": None,
                    "front_origin": None,
                    "wall_time_s": None,
                    "error": None,
                    "Da": Da_v,
                    "kappa": ka_v,
                    "c1": c1v,
                    "c2": c2v,
                    "T": Tv,
                    "n_transitions": None,
                    "dwell_L_mean": None,
                    "dwell_H_mean": None,
                    "peak_period": None,
                    "thickness_final": None,
                    "osc_predicted": osc_p,
                    "osc_observed": False,
                    "osc_match": not osc_p,
                }
            )
            continue

        status = "success"
        rel_path: str | None = None
        error: str | None = None
        band_count: int | None = None
        residual_pct: float | None = None
        wall_time_s: float | None = None
        front_origin: str | None = None
        kinetic_extra: dict[str, Any] = {}
        try:
            t0 = time.perf_counter()
            result = run_one(
                cfg,
                results_root=sweep_dir,
                chunk_size=chunk_size,
                write_artifacts=True,
                show_progress=False,
                log_level="WARNING",
            )
            assert result.paths is not None
            rel_path = result.paths.root.relative_to(sweep_dir).as_posix()
            wall_time_s = float(time.perf_counter() - t0)
            total_wall += wall_time_s
            slab_ax = result.diagnostics.get("slab_axial", {})
            dmb = result.diagnostics.get("dirichlet_mass_balance", {})
            band_count = (
                int(slab_ax.get("band_count_axial", 0)) if isinstance(slab_ax, dict) else None
            )
            residual_pct = float(dmb.get("residual_pct", 0.0)) if isinstance(dmb, dict) else None
            model_nm = str(cfg.get("experiment", {}).get("model", ""))
            run_dir = sweep_dir / rel_path
            if model_nm == "slab_kinetic":
                front_origin = None
                diag = result.diagnostics
                phys = cfg.get("physics", {})
                Da = float(phys.get("Da", float("nan")))
                kappa = float(phys.get("kappa", float("nan")))
                c1 = float(phys.get("c1", float("nan")))
                c2 = float(phys.get("c2", float("nan")))
                osc_pred = analytic_strip_predict(Da, kappa, c1, c2)
                n_tr = int(diag.get("n_transitions", 0))
                osc_obs = n_tr >= 2

                kinetic_extra = {
                    "Da": Da,
                    "kappa": kappa,
                    "c1": c1,
                    "c2": c2,
                    "T": float(cfg.get("time", {}).get("T", float("nan"))),
                    "n_transitions": n_tr,
                    "dwell_L_mean": _diag_optional_float(diag, "dwell_L_mean"),
                    "dwell_H_mean": _diag_optional_float(diag, "dwell_H_mean"),
                    "peak_period": _diag_optional_float(diag, "peak_period"),
                    "mean_dwell_period": _diag_optional_float(diag, "mean_dwell_period"),
                    "mean_dwell_freq": _diag_optional_float(diag, "mean_dwell_freq"),
                    "thickness_final": _diag_optional_float(diag, "thickness_final"),
                    "osc_predicted": osc_pred,
                    "osc_observed": osc_obs,
                    "osc_match": bool(osc_pred == osc_obs),
                }
            else:
                rim_left = int(cfg.get("geometry", {}).get("rim_left_width_px", 0))
                rim_right = int(cfg.get("geometry", {}).get("rim_width_px", 0))
                front_origin = _front_origin_from_h5_or_final(
                    run_dir,
                    rim_left_width_px=rim_left,
                    rim_width_px=rim_right,
                )
        except ValueError as exc:
            msg = str(exc)
            if exp_m == "slab_kinetic" and "c1 must be > c2" in msg:
                osc_p = analytic_strip_predict(Da_v, ka_v, c1v, c2v)
                row_skipped = {
                    "run_label": run_label,
                    "parameters": combo,
                    "status": "skipped",
                    "relative_path": None,
                    "band_count_axial": None,
                    "residual_pct": None,
                    "front_origin": None,
                    "wall_time_s": None,
                    "error": None,
                    "Da": Da_v,
                    "kappa": ka_v,
                    "c1": c1v,
                    "c2": c2v,
                    "T": Tv,
                    "n_transitions": None,
                    "dwell_L_mean": None,
                    "dwell_H_mean": None,
                    "peak_period": None,
                    "thickness_final": None,
                    "osc_predicted": osc_p,
                    "osc_observed": False,
                    "osc_match": not osc_p,
                }
                entries.append(row_skipped)
                continue
            status = "failed"
            error = msg
            row = {
                "run_label": run_label,
                "parameters": combo,
                "status": status,
                "relative_path": None,
                "band_count_axial": None,
                "residual_pct": None,
                "front_origin": None,
                "wall_time_s": wall_time_s,
                "error": error,
            }
            row.update(kinetic_extra)
            entries.append(row)
            continue
        except Exception as exc:
            status = "failed"
            error = str(exc)

        row: dict[str, Any] = {
            "run_label": run_label,
            "parameters": combo,
            "status": status,
            "relative_path": rel_path,
            "band_count_axial": band_count,
            "residual_pct": residual_pct,
            "front_origin": front_origin,
            "wall_time_s": wall_time_s,
            "error": error,
        }
        row.update(kinetic_extra)
        entries.append(row)

    success_entries = [e for e in entries if e.get("status") == "success"]
    osc_matched = [e for e in success_entries if e.get("osc_match")]
    osc_rate = 100.0 * len(osc_matched) / max(len(success_entries), 1)

    manifest: dict[str, Any] = {
        "sweep_name": name,
        "timestamp": stamp,
        "source_sweep_yaml": str(sweep_path),
        "results_root": str(sweep_dir.resolve()),
        "nested_results": nested,
        "total_wall_time_s": float(total_wall),
        "osc_match_rate_pct": float(osc_rate),
        "entries": entries,
    }
    (sweep_dir / "manifest_axes.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    if name == "slab_kinetic_oscillation_window":
        import importlib.util

        mod_path = _SCRIPTS_DIR / "slab_kinetic_oscillation_window_outputs.py"
        spec = importlib.util.spec_from_file_location("oscillation_window_outputs", mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {mod_path}")
        ow_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ow_mod)
        ow_mod.write_oscillation_window_artifacts(sweep_dir, manifest)

    print(f"Sweep complete: {sweep_dir}")
    failed_only = [e for e in entries if e.get("status") == "failed"]
    if failed_only:
        print(f"WARNING: {len(failed_only)} runs failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
