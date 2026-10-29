"""Run a generic axes-based Cartesian sweep."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import shutil
import sys
import time
from concurrent.futures import as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

from continuous_patterns.core.io import (
    allocate_run_dir,
    load_run_config,
    save_run_config,
    save_summary,
    validate_run_config_dict,
)
from continuous_patterns.experiments.run import run_one, write_slab_kinetic_run_artifacts

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


def _prune_slab_twophase_physics(cfg: dict[str, Any]) -> None:
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
        for key in ("W", "kappa", "M0", "phi_gate"):
            phys.pop(key, None)


def _build_explicit_run_cfg(
    base_cfg: dict[str, Any],
    *,
    run_label: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = run_label
    for dotted, value in overrides.items():
        _set_dotted(cfg, str(dotted), copy.deepcopy(value))
    if str(cfg["experiment"].get("model", "")) == "slab_twophase":
        _prune_slab_twophase_physics(cfg)
        return validate_run_config_dict(cfg)
    return cfg


def _s1_manifest_from_analysis(
    *,
    run_label: str,
    rel_path: str,
    wall_time_s: float,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_label": run_label,
        "status": "success",
        "relative_path": rel_path,
        "wall_time_s": wall_time_s,
        "c_ostwald": analysis.get("c_ostwald"),
        "w_ostwald": analysis.get("w_ostwald"),
        "gamma": analysis.get("gamma"),
        "n_bands_zone": analysis.get("n_bands_zone"),
        "band_zone_extent": analysis.get("band_zone_extent"),
        "mean_spacing_zone": analysis.get("mean_spacing_zone"),
        "median_spacing_zone": analysis.get("median_spacing_zone"),
        "mean_spacing_zone_excl_last": analysis.get("mean_spacing_zone_excl_last"),
        "std_spacing_zone": analysis.get("std_spacing_zone"),
        "mass_residual_pct": analysis.get("mass_residual_pct"),
        "clipped_mass_frac": analysis.get("clipped_mass_frac"),
        "row_validity": analysis.get("row_validity"),
        "spacing_pacing": analysis.get("spacing_pacing"),
    }


def _execute_explicit_run(
    cfg: dict[str, Any],
    *,
    run_label: str,
    sweep_dir: Path,
    chunk_size: int,
    show_progress: bool,
    analyze_s1: bool,
) -> dict[str, Any]:
    from continuous_patterns.experiments.s1_ostwald_window import analyze_s1_run

    t0 = time.perf_counter()
    result = run_one(
        cfg,
        results_root=sweep_dir,
        chunk_size=chunk_size,
        write_artifacts=True,
        show_progress=show_progress,
        log_level="WARNING",
    )
    assert result.paths is not None
    rel_path = result.paths.root.relative_to(sweep_dir).as_posix()
    wall_time_s = float(time.perf_counter() - t0)
    if analyze_s1:
        analysis = analyze_s1_run(result.paths.root, run_label=run_label)
        return _s1_manifest_from_analysis(
            run_label=run_label,
            rel_path=rel_path,
            wall_time_s=wall_time_s,
            analysis=analysis,
        )
    return _entry_from_result(
        combo={"run_label": run_label},
        run_label=run_label,
        cfg=cfg,
        result=result,
        rel_path=rel_path,
        wall_time_s=wall_time_s,
    )


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


def _kinetic_skipped_entry(
    *,
    run_label: str,
    combo: dict[str, Any],
    Da_v: float,
    ka_v: float,
    c1v: float,
    c2v: float,
    Tv: float,
) -> dict[str, Any]:
    osc_p = analytic_strip_predict(Da_v, ka_v, c1v, c2v)
    return {
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


def _kinetic_manifest_extras(cfg: dict[str, Any], diag: dict[str, Any]) -> dict[str, Any]:
    phys = cfg.get("physics", {})
    Da = float(phys.get("Da", float("nan")))
    kappa = float(phys.get("kappa", float("nan")))
    c1 = float(phys.get("c1", float("nan")))
    c2 = float(phys.get("c2", float("nan")))
    osc_pred = analytic_strip_predict(Da, kappa, c1, c2)
    n_tr = int(diag.get("n_transitions", 0))
    osc_obs = n_tr >= 2
    return {
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


def _entry_from_result(
    *,
    combo: dict[str, Any],
    run_label: str,
    cfg: dict[str, Any],
    result: Any,
    rel_path: str,
    wall_time_s: float,
) -> dict[str, Any]:
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    slab_ax = diag.get("slab_axial", {})
    dmb = diag.get("dirichlet_mass_balance", {})
    band_count = int(slab_ax.get("band_count_axial", 0)) if isinstance(slab_ax, dict) else None
    residual_pct = float(dmb.get("residual_pct", 0.0)) if isinstance(dmb, dict) else None
    model_nm = str(cfg.get("experiment", {}).get("model", ""))
    row: dict[str, Any] = {
        "run_label": run_label,
        "parameters": combo,
        "status": "success",
        "relative_path": rel_path,
        "band_count_axial": band_count,
        "residual_pct": residual_pct,
        "front_origin": None,
        "wall_time_s": wall_time_s,
        "error": None,
    }
    if model_nm == "slab_kinetic":
        row.update(_kinetic_manifest_extras(cfg, diag))
    else:
        run_dir = Path(rel_path)
        rim_left = int(cfg.get("geometry", {}).get("rim_left_width_px", 0))
        rim_right = int(cfg.get("geometry", {}).get("rim_width_px", 0))
        row["front_origin"] = _front_origin_from_h5_or_final(
            run_dir,
            rim_left_width_px=rim_left,
            rim_width_px=rim_right,
        )
    return row


def _run_jax_kinetic_batch(
    *,
    valid_jobs: list[tuple[dict[str, Any], dict[str, Any], str]],
    sweep_dir: Path,
) -> tuple[list[dict[str, Any]], float, dict[str, float]]:
    from continuous_patterns.models.slab_kinetic_jax import BatchParams, simulate_batch

    if not valid_jobs:
        return [], 0.0, {"compile_s": 0.0, "exec_s": 0.0}

    base_cfg = copy.deepcopy(valid_jobs[0][1])
    base_cfg.setdefault("solver", {})
    base_cfg["solver"]["backend"] = "jax"

    batch = BatchParams(
        Da=np.asarray([j[1]["physics"]["Da"] for j in valid_jobs], dtype=np.float64),
        kappa=np.asarray([j[1]["physics"]["kappa"] for j in valid_jobs], dtype=np.float64),
        c1=np.asarray([j[1]["physics"]["c1"] for j in valid_jobs], dtype=np.float64),
        c2=np.asarray([j[1]["physics"]["c2"] for j in valid_jobs], dtype=np.float64),
    )
    results, timing = simulate_batch(base_cfg, batch)
    per_run_wall = float(timing["exec_s"]) / max(len(valid_jobs), 1)

    entries: list[dict[str, Any]] = []
    for (combo, cfg, run_label), result in zip(valid_jobs, results, strict=True):
        paths = allocate_run_dir(
            experiment_name=str(cfg["experiment"]["name"]),
            results_root=sweep_dir,
        )
        save_run_config(paths.config_yaml, cfg)
        save_summary(paths.summary_json, result.diagnostics)
        write_slab_kinetic_run_artifacts(result, cfg, paths, wall_s=per_run_wall)
        rel_path = paths.root.relative_to(sweep_dir).as_posix()
        entries.append(
            _entry_from_result(
                combo=combo,
                run_label=run_label,
                cfg=cfg,
                result=result,
                rel_path=rel_path,
                wall_time_s=per_run_wall,
            )
        )
    return entries, float(timing["exec_s"]), timing


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an axes-based Cartesian sweep.")
    parser.add_argument("sweep_yaml", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU workers (numpy/CH).")
    parser.add_argument(
        "--invocation-id",
        type=str,
        default=None,
        help="Fixed timestamp subdirectory (for parallel subset jobs into one sweep dir).",
    )
    parser.add_argument(
        "--run-labels",
        type=str,
        default=None,
        help="Comma-separated explicit run labels to execute (requires sweep.runs).",
    )
    parser.add_argument(
        "--finalize-s1",
        action="store_true",
        help="Write S1 table/figure from manifest only (slab_twophase_ostwald_window).",
    )
    return parser.parse_args(argv)


def _write_manifest_snapshot(sweep_dir: Path, manifest: dict[str, Any]) -> None:
    (sweep_dir / "manifest_axes.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def _run_explicit_sweep(
    *,
    sweep_path: Path,
    sweep: dict[str, Any],
    base_cfg: dict[str, Any],
    sweep_dir: Path,
    runs_spec: list[dict[str, Any]],
    chunk_size: int,
    workers: int,
    label_filter: set[str] | None,
    analyze_s1: bool,
) -> tuple[list[dict[str, Any]], float]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    for spec in runs_spec:
        if not isinstance(spec, dict):
            raise ValueError("each sweep.runs entry must be a mapping")
        run_label = str(spec["run_label"])
        if label_filter is not None and run_label not in label_filter:
            continue
        overrides = spec.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{run_label}: overrides must be a mapping")
        cfg = _build_explicit_run_cfg(base_cfg, run_label=run_label, overrides=overrides)
        jobs.append((run_label, cfg))

    if not jobs:
        raise ValueError("no runs selected (check --run-labels filter)")

    entries: list[dict[str, Any]] = []
    total_wall = 0.0
    name = str(sweep.get("name", "sweep_axes"))

    if workers > 1:
        payloads = [
            {
                "cfg": cfg,
                "run_label": run_label,
                "sweep_dir": str(sweep_dir.resolve()),
                "chunk_size": chunk_size,
                "analyze_s1": analyze_s1,
            }
            for run_label, cfg in jobs
        ]
        from continuous_patterns.experiments.process_pool import ManagedProcessPool
        from continuous_patterns.experiments.sweep_worker import (
            ensure_cpu_jax_backend,
            run_s1_combo,
        )

        mp_ctx = get_context("spawn")
        with ManagedProcessPool(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=ensure_cpu_jax_backend,
        ) as pool:
            futures = [pool.submit(run_s1_combo, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Sweep {name} (workers={workers})",
                unit="run",
            ):
                row = fut.result()
                entries.append(row)
                if row.get("status") == "success" and row.get("wall_time_s") is not None:
                    total_wall += float(row["wall_time_s"])
    else:
        for run_label, cfg in tqdm(jobs, desc=f"Sweep {name}", unit="run", leave=True):
            try:
                row = _execute_explicit_run(
                    cfg,
                    run_label=run_label,
                    sweep_dir=sweep_dir,
                    chunk_size=chunk_size,
                    show_progress=True,
                    analyze_s1=analyze_s1,
                )
            except Exception as exc:
                row = {
                    "run_label": run_label,
                    "status": "failed",
                    "relative_path": None,
                    "wall_time_s": None,
                    "error": str(exc),
                }
            entries.append(row)
            if row.get("status") == "success" and row.get("wall_time_s") is not None:
                total_wall += float(row["wall_time_s"])

    return entries, total_wall


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    sweep_path = ns.sweep_yaml
    out_root = ns.out_dir
    chunk_size = int(ns.chunk_size)
    backend = str(ns.backend)
    workers = max(1, int(ns.workers))

    raw = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep config must be a mapping")
    sweep = raw.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("missing sweep section")

    name = str(sweep.get("name", "sweep_axes"))
    nested = bool(sweep.get("nested_results", False))
    if ns.invocation_id:
        stamp = str(ns.invocation_id)
        sweep_dir = out_root / stamp
        sweep_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        sweep_dir = (out_root / stamp) if nested else out_root
        sweep_dir.mkdir(parents=True, exist_ok=False)

    if ns.finalize_s1:
        manifest_path = sweep_dir / "manifest_axes.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import importlib.util

        mod_path = _SCRIPTS_DIR / "slab_twophase_ostwald_window_outputs.py"
        spec = importlib.util.spec_from_file_location("s1_outputs", mod_path)
        assert spec is not None and spec.loader is not None
        s1_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s1_mod)
        s1_mod.write_s1_artifacts(sweep_dir, manifest)
        return 0

    base_cfg = load_run_config(Path(sweep["base_config"]))
    overrides = raw.get("overrides", {})
    if isinstance(overrides, dict):
        for dotted_key, v in overrides.items():
            _set_dotted(base_cfg, dotted_key, copy.deepcopy(v))

    label_filter: set[str] | None = None
    if ns.run_labels:
        label_filter = {s.strip() for s in str(ns.run_labels).split(",") if s.strip()}

    runs_spec = sweep.get("runs")
    explicit_mode = isinstance(runs_spec, list) and len(runs_spec) > 0

    if explicit_mode:
        shutil.copy2(sweep_path, sweep_dir / "sweep_config.yaml")
        analyze_s1 = name == "slab_twophase_ostwald_window"
        entries, total_wall = _run_explicit_sweep(
            sweep_path=sweep_path,
            sweep=sweep,
            base_cfg=base_cfg,
            sweep_dir=sweep_dir,
            runs_spec=runs_spec,
            chunk_size=chunk_size,
            workers=workers,
            label_filter=label_filter,
            analyze_s1=analyze_s1,
        )
        manifest_path = sweep_dir / "manifest_axes.json"
        if label_filter is not None and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            merged = {str(e.get("run_label")): e for e in prior.get("entries", [])}
            for row in entries:
                merged[str(row.get("run_label"))] = row
            entries = list(merged.values())
            total_wall = float(prior.get("total_wall_time_s", 0.0)) + float(total_wall)
        manifest: dict[str, Any] = {
            "sweep_name": name,
            "timestamp": stamp,
            "source_sweep_yaml": str(sweep_path),
            "results_root": str(sweep_dir.resolve()),
            "nested_results": nested,
            "total_wall_time_s": float(total_wall),
            "backend": "numpy",
            "workers": workers,
            "entries": entries,
        }
        _write_manifest_snapshot(sweep_dir, manifest)
        if analyze_s1 and label_filter is None:
            import importlib.util

            mod_path = _SCRIPTS_DIR / "slab_twophase_ostwald_window_outputs.py"
            spec = importlib.util.spec_from_file_location("s1_outputs", mod_path)
            assert spec is not None and spec.loader is not None
            s1_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(s1_mod)
            s1_mod.write_s1_artifacts(sweep_dir, manifest)
        print(f"Sweep complete: {sweep_dir}", flush=True)
        failed_only = [e for e in entries if e.get("status") == "failed"]
        if failed_only:
            print(f"WARNING: {len(failed_only)} runs failed", file=sys.stderr)
            return 1
        return 0

    axes = sweep.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("expected non-empty list in sweep.axes or sweep.runs")

    axis_defs: list[dict[str, Any]] = []
    for ax in axes:
        if not isinstance(ax, dict):
            raise ValueError("each axis must be a mapping")
        name_ax = str(ax.get("name", "")).strip()
        vals = list(ax.get("values", []))
        if not name_ax or not vals:
            raise ValueError("axis requires non-empty name and values")
        axis_defs.append(ax)

    axis_names = [str(ax["name"]) for ax in axis_defs]
    combos = [
        dict(zip(axis_names, vals, strict=True))
        for vals in itertools.product(*[ax["values"] for ax in axis_defs])
    ]

    exp_model = str(base_cfg.get("experiment", {}).get("model", ""))
    use_jax_batch = backend == "jax" and exp_model == "slab_kinetic"
    if backend == "jax" and exp_model != "slab_kinetic":
        print("WARNING: --backend jax ignored (only slab_kinetic supports JAX)", file=sys.stderr)

    entries: list[dict[str, Any]] = []
    total_wall = 0.0
    jax_timing: dict[str, float] = {}

    if use_jax_batch:
        valid_jobs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        for combo in combos:
            cfg = copy.deepcopy(base_cfg)
            run_label = _combo_name(combo)
            cfg["experiment"]["name"] = run_label
            _apply_axis_to_cfg(cfg, axis_defs, combo)
            phys = cfg.get("physics", {})
            c1v = float(phys.get("c1", float("nan")))
            c2v = float(phys.get("c2", float("nan")))
            if not (c1v > c2v):
                entries.append(
                    _kinetic_skipped_entry(
                        run_label=run_label,
                        combo=combo,
                        Da_v=float(phys.get("Da", float("nan"))),
                        ka_v=float(phys.get("kappa", float("nan"))),
                        c1v=c1v,
                        c2v=c2v,
                        Tv=float(cfg.get("time", {}).get("T", float("nan"))),
                    )
                )
                continue
            valid_jobs.append((combo, cfg, run_label))

        batch_entries, exec_s, jax_timing = _run_jax_kinetic_batch(
            valid_jobs=valid_jobs,
            sweep_dir=sweep_dir,
        )
        entries.extend(batch_entries)
        total_wall = exec_s
        print(
            f"JAX batch: B={len(valid_jobs)} compile={jax_timing.get('compile_s', 0.0):.2f}s "
            f"exec={jax_timing.get('exec_s', 0.0):.2f}s",
            flush=True,
        )
    elif workers > 1:
        payloads: list[dict[str, Any]] = []
        for combo in combos:
            cfg = copy.deepcopy(base_cfg)
            run_label = _combo_name(combo)
            cfg["experiment"]["name"] = run_label
            _apply_axis_to_cfg(cfg, axis_defs, combo)
            payloads.append(
                {
                    "cfg": cfg,
                    "combo": combo,
                    "run_label": run_label,
                    "sweep_dir": str(sweep_dir.resolve()),
                    "chunk_size": chunk_size,
                }
            )
        from continuous_patterns.experiments.process_pool import ManagedProcessPool
        from continuous_patterns.experiments.sweep_worker import ensure_cpu_jax_backend, run_combo

        mp_ctx = get_context("spawn")
        with ManagedProcessPool(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=ensure_cpu_jax_backend,
        ) as pool:
            futures = [pool.submit(run_combo, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Sweep {name} (workers={workers})",
                unit="run",
            ):
                row = fut.result()
                entries.append(row)
                if row.get("status") == "success" and row.get("wall_time_s") is not None:
                    total_wall += float(row["wall_time_s"])
    else:
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
                entries.append(
                    _kinetic_skipped_entry(
                        run_label=run_label,
                        combo=combo,
                        Da_v=Da_v,
                        ka_v=ka_v,
                        c1v=c1v,
                        c2v=c2v,
                        Tv=Tv,
                    )
                )
                continue

            status = "success"
            rel_path: str | None = None
            error: str | None = None
            kinetic_extra: dict[str, Any] = {}
            front_origin: str | None = None
            band_count: int | None = None
            residual_pct: float | None = None
            wall_time_s: float | None = None
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
                diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
                slab_ax = diag.get("slab_axial", {})
                dmb = diag.get("dirichlet_mass_balance", {})
                band_count = (
                    int(slab_ax.get("band_count_axial", 0)) if isinstance(slab_ax, dict) else None
                )
                residual_pct = (
                    float(dmb.get("residual_pct", 0.0)) if isinstance(dmb, dict) else None
                )
                model_nm = str(cfg.get("experiment", {}).get("model", ""))
                if model_nm == "slab_kinetic":
                    kinetic_extra = _kinetic_manifest_extras(cfg, diag)
                else:
                    rim_left = int(cfg.get("geometry", {}).get("rim_left_width_px", 0))
                    rim_right = int(cfg.get("geometry", {}).get("rim_width_px", 0))
                    front_origin = _front_origin_from_h5_or_final(
                        sweep_dir / rel_path,
                        rim_left_width_px=rim_left,
                        rim_width_px=rim_right,
                    )
            except ValueError as exc:
                msg = str(exc)
                if exp_m == "slab_kinetic" and "c1 must be > c2" in msg:
                    entries.append(
                        _kinetic_skipped_entry(
                            run_label=run_label,
                            combo=combo,
                            Da_v=Da_v,
                            ka_v=ka_v,
                            c1v=c1v,
                            c2v=c2v,
                            Tv=Tv,
                        )
                    )
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

            row = {
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

    manifest: dict[str, Any] = {
        "sweep_name": name,
        "timestamp": stamp,
        "source_sweep_yaml": str(sweep_path),
        "results_root": str(sweep_dir.resolve()),
        "nested_results": nested,
        "total_wall_time_s": float(total_wall),
        "osc_match_rate_pct": float(
            100.0
            * len([e for e in entries if e.get("status") == "success" and e.get("osc_match")])
            / max(len([e for e in entries if e.get("status") == "success"]), 1)
        ),
        "backend": backend if use_jax_batch else "numpy",
        "workers": workers,
        "entries": entries,
    }
    if jax_timing:
        manifest["jax_timing"] = jax_timing
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
