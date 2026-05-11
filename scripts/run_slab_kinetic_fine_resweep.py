#!/usr/bin/env python3
"""Run 10 explicit (c1, c2) slab_kinetic pairs with dt_safety=0.1; compare to main sweep."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]

from continuous_patterns.core.io import load_run_config  # noqa: E402
from continuous_patterns.experiments.run import run_one  # noqa: E402


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


def analytic_strip_predict(Da: float, kappa: float, c1: float, c2: float) -> bool:
    lo = 1.0 / (1.0 + float(kappa) * float(Da))
    hi = 1.0 / (1.0 + float(Da))
    fc1 = float(c1)
    fc2 = float(c2)
    return (fc2 > lo) and (fc1 < hi) and (fc1 > fc2)


def _slug_value(v: Any) -> str:
    if isinstance(v, float):
        if math.isfinite(v):
            return f"{v:.6g}".replace("-", "m").replace(".", "p")
        return str(v).lower()
    return str(v)


def _run_label(c1: float, c2: float) -> str:
    return f"c1_{_slug_value(c1)}__c2_{_slug_value(c2)}"


def _lookup_main_entry(
    main_manifest: dict[str, Any], c1: float, c2: float
) -> dict[str, Any] | None:
    for e in main_manifest.get("entries", []):
        if e.get("status") != "success":
            continue
        if abs(float(e.get("c1", -1)) - c1) < 1e-9 and abs(float(e.get("c2", -1)) - c2) < 1e-9:
            return e
    return None


def _rel_tol_ok(a: float | None, b: float | None, tol: float = 0.2) -> bool:
    if a is None or b is None:
        return False
    fa, fb = float(a), float(b)
    if np.isnan(fa) or np.isnan(fb):
        return False
    den = max(abs(fa), abs(fb), 1e-30)
    return abs(fa - fb) / den <= tol


def _rel_tol_ok_counts(a: int | None, b: int | None, tol: float = 0.2) -> bool:
    if a is None or b is None:
        return False
    fa, fb = float(a), float(b)
    den = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / den <= tol


def _pairs_list() -> list[tuple[float, float]]:
    return [
        (0.30, 0.27),
        (0.30, 0.29),
        (0.33, 0.29),
        (0.33, 0.31),
        (0.36, 0.34),
        (0.39, 0.37),
        (0.42, 0.40),
        (0.45, 0.43),
        (0.47, 0.46),
        (0.49, 0.46),
    ]


def finalize_existing_sweep_directory(sweep_dir: Path, main_dir: Path) -> None:
    """Rebuild manifest + tables from run folders (after interrupted sweep)."""
    from continuous_patterns.core.diagnostics_kinetic import compute_mean_dwell_period

    sweep_dir = sweep_dir.resolve()
    main_dir = main_dir.resolve()
    main_manifest = json.loads((main_dir / "manifest_axes.json").read_text(encoding="utf-8"))

    yaml_meta = yaml.safe_load(
        (_REPO / "experiments/sweeps/slab_kinetic_oscillation_window_fine.yaml").read_text(
            encoding="utf-8"
        )
    )
    sweep_meta = yaml_meta.get("sweep", {})

    pairs = _pairs_list()
    entries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    total_wall = 0.0

    for c1, c2 in pairs:
        lbl = _run_label(c1, c2)
        base = sweep_dir / lbl
        if not base.is_dir():
            raise FileNotFoundError(f"missing run folder for pair ({c1}, {c2}): {base}")

        run_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"no timestamp subdir under {base}")

        rdir = run_dirs[-1]
        sj_path = rdir / "summary.json"
        if not sj_path.is_file():
            raise FileNotFoundError(f"missing summary under {rdir}")

        summary = json.loads(sj_path.read_text(encoding="utf-8"))
        rel_path = rdir.relative_to(sweep_dir).as_posix()

        n_tr = int(summary.get("n_transitions", 0))
        md = summary.get("mean_dwell_period")
        pk = summary.get("peak_period")
        wall_s = float(summary.get("wall_time_s", 0.0))
        total_wall += wall_s

        osc_p = analytic_strip_predict(1.0, 3.0, c1, c2)
        osc_o = n_tr >= 2
        row = {
            "run_label": lbl,
            "parameters": {"c1": c1, "c2": c2},
            "status": "success",
            "relative_path": rel_path,
            "band_count_axial": 0,
            "residual_pct": 0.0,
            "front_origin": None,
            "wall_time_s": wall_s,
            "error": None,
            "Da": 1.0,
            "kappa": 3.0,
            "c1": c1,
            "c2": c2,
            "T": float(summary.get("T", float("nan"))),
            "n_transitions": n_tr,
            "dwell_L_mean": summary.get("dwell_L_mean"),
            "dwell_H_mean": summary.get("dwell_H_mean"),
            "peak_period": pk,
            "mean_dwell_period": md,
            "mean_dwell_freq": summary.get("mean_dwell_freq"),
            "thickness_final": summary.get("thickness_final"),
            "osc_predicted": osc_p,
            "osc_observed": osc_o,
            "osc_match": bool(osc_p == osc_o),
        }
        entries.append(row)

        orig = _lookup_main_entry(main_manifest, c1, c2)
        n_o = orig.get("n_transitions") if orig else None
        md_o = orig.get("mean_dwell_period") if orig else None
        if md_o is None and orig is not None:
            summary_path = main_dir / str(orig.get("relative_path", "")) / "summary.json"
            if summary_path.is_file():
                sj = json.loads(summary_path.read_text(encoding="utf-8"))
                dl = np.asarray(sj.get("dwell_L_samples") or [], dtype=np.float64)
                dh = np.asarray(sj.get("dwell_H_samples") or [], dtype=np.float64)
                mp = compute_mean_dwell_period(dl, dh, int(sj.get("n_transitions", 0)))
                md_o = mp["mean_dwell_period"]

        pk_o = orig.get("peak_period") if orig else None

        cmp_row = {
            "c1": c1,
            "c2": c2,
            "n_transitions_original": n_o,
            "n_transitions_fine": n_tr,
            "peak_period_fft_original": pk_o,
            "peak_period_fft_fine": pk,
            "mean_dwell_period_original": md_o,
            "mean_dwell_period_fine": md,
            "converged_n_transitions": _rel_tol_ok_counts(
                int(n_o) if n_o is not None else None,
                int(n_tr),
            ),
            "converged_mean_dwell_period": _rel_tol_ok(
                float(md_o) if md_o is not None else None,
                float(md) if md is not None else None,
            ),
        }
        cmp_row["converged_both"] = bool(
            cmp_row["converged_n_transitions"] and cmp_row["converged_mean_dwell_period"]
        )
        comparisons.append(cmp_row)

    stamp = sweep_dir.name
    manifest: dict[str, Any] = {
        "sweep_name": str(sweep_meta.get("name", "slab_kinetic_oscillation_window_fine")),
        "timestamp": stamp,
        "source_sweep_yaml": str(
            _REPO / "experiments/sweeps/slab_kinetic_oscillation_window_fine.yaml"
        ),
        "results_root": str(sweep_dir.resolve()),
        "nested_results": True,
        "main_sweep_reference": str(main_dir),
        "total_wall_time_s": float(total_wall),
        "comparison_with_main": comparisons,
        "entries": entries,
    }

    (sweep_dir / "manifest_axes.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    lines = [
        "| c1 | c2 | n_trans (orig) | n_trans (fine) | fft peak (orig) | fft peak (fine) | "
        "mean_dwell (orig) | mean_dwell (fine) | conv n | conv τ | conv both |",
        "|----|----|----------------|----------------|----------------|----------------|"
        "-------------------|-------------------|--------|--------|-----------|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['c1']} | {c['c2']} | {c['n_transitions_original']} | "
            f"{c['n_transitions_fine']} | "
            f"{c['peak_period_fft_original']} | {c['peak_period_fft_fine']} | "
            f"{c['mean_dwell_period_original']} | {c['mean_dwell_period_fine']} | "
            f"{c['converged_n_transitions']} | {c['converged_mean_dwell_period']} | "
            f"{c['converged_both']} |"
        )
    (sweep_dir / "table_comparison_main_vs_fine.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    mini = "\n".join(
        [
            "| run_label | c1 | c2 | status | n_transitions | "
            "mean_dwell_period | peak_period | wall_time_s |",
            "|-------------|----|----|--------|---------------|"
            "|-------------------|-------------|-------------|",
        ]
        + [
            f"| {e['run_label']} | {e['c1']} | {e['c2']} | {e['status']} | {e['n_transitions']} | "
            f"{e.get('mean_dwell_period')} | {e.get('peak_period')} | {e['wall_time_s']:.4g} |"
            for e in entries
        ]
    )
    (sweep_dir / "table_all_runs.md").write_text(mini + "\n", encoding="utf-8")

    n_conv_n = sum(1 for c in comparisons if c["converged_n_transitions"])
    n_conv_t = sum(1 for c in comparisons if c["converged_mean_dwell_period"])
    n_both = sum(1 for c in comparisons if c["converged_both"])
    stats = (
        f"convergence_tol=20%  n_transitions: {n_conv_n}/10  "
        f"mean_dwell_period: {n_conv_t}/10  both: {n_both}/10\n"
    )
    (sweep_dir / "convergence_stats.txt").write_text(stats, encoding="utf-8")
    print(stats)
    print(f"Finalized manifest under {sweep_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--main-sweep-dir",
        type=Path,
        required=True,
        help="Main oscillation-window sweep dir containing manifest_axes.json",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_REPO / "results/slab_kinetic/sweep_oscillation_window_fine",
        help="Directory under which a timestamped sweep folder is created",
    )
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument(
        "--finalize-existing",
        type=Path,
        metavar="SWEEP_DIR",
        default=None,
        help="Rebuild manifest + comparison tables from existing run folders (no simulations)",
    )
    args = ap.parse_args(argv)

    if args.finalize_existing is not None:
        if args.main_sweep_dir is None:
            print("--main-sweep-dir required with --finalize-existing", file=sys.stderr)
            return 2
        finalize_existing_sweep_directory(
            args.finalize_existing.resolve(),
            args.main_sweep_dir.resolve(),
        )
        return 0

    main_dir = args.main_sweep_dir.resolve()
    man_main_path = main_dir / "manifest_axes.json"
    if not man_main_path.is_file():
        print(f"Missing {man_main_path}", file=sys.stderr)
        return 1

    main_manifest = json.loads(man_main_path.read_text(encoding="utf-8"))

    yaml_meta = yaml.safe_load(
        (_REPO / "experiments/sweeps/slab_kinetic_oscillation_window_fine.yaml").read_text(
            encoding="utf-8"
        )
    )
    sweep_meta = yaml_meta.get("sweep", {})
    base_cfg = load_run_config(_REPO / sweep_meta["base_config"])
    overrides = sweep_meta.get("overrides", {})
    if isinstance(overrides, dict):
        for dotted_key, val in overrides.items():
            _set_dotted(base_cfg, str(dotted_key), val)

    pairs = _pairs_list()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = args.out_root.resolve() / stamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    total_wall = 0.0

    for c1, c2 in pairs:
        cfg = copy.deepcopy(base_cfg)
        lbl = _run_label(c1, c2)
        cfg["experiment"]["name"] = lbl
        cfg.setdefault("physics", {})
        cfg["physics"]["c1"] = float(c1)
        cfg["physics"]["c2"] = float(c2)

        orig = _lookup_main_entry(main_manifest, c1, c2)
        if orig is None:
            print(f"WARNING: no main manifest entry for ({c1}, {c2})", file=sys.stderr)

        t0 = time.perf_counter()
        result = run_one(
            cfg,
            results_root=sweep_dir,
            chunk_size=args.chunk_size,
            write_artifacts=True,
            show_progress=False,
            log_level="WARNING",
        )
        wall_s = float(time.perf_counter() - t0)
        total_wall += wall_s

        assert result.paths is not None
        rel_path = result.paths.root.relative_to(sweep_dir).as_posix()
        diag = result.diagnostics
        n_tr = int(diag.get("n_transitions", 0))
        md = diag.get("mean_dwell_period")
        pk = diag.get("peak_period")

        osc_p = analytic_strip_predict(1.0, 3.0, c1, c2)
        osc_o = n_tr >= 2
        row = {
            "run_label": lbl,
            "parameters": {"c1": c1, "c2": c2},
            "status": "success",
            "relative_path": rel_path,
            "band_count_axial": 0,
            "residual_pct": 0.0,
            "front_origin": None,
            "wall_time_s": wall_s,
            "error": None,
            "Da": 1.0,
            "kappa": 3.0,
            "c1": c1,
            "c2": c2,
            "T": float(cfg.get("time", {}).get("T", float("nan"))),
            "n_transitions": n_tr,
            "dwell_L_mean": diag.get("dwell_L_mean"),
            "dwell_H_mean": diag.get("dwell_H_mean"),
            "peak_period": pk,
            "mean_dwell_period": md,
            "mean_dwell_freq": diag.get("mean_dwell_freq"),
            "thickness_final": diag.get("thickness_final"),
            "osc_predicted": osc_p,
            "osc_observed": osc_o,
            "osc_match": bool(osc_p == osc_o),
        }
        entries.append(row)

        n_o = orig.get("n_transitions") if orig else None
        md_o = orig.get("mean_dwell_period") if orig else None
        if md_o is None and orig is not None:
            summary_path = main_dir / str(orig.get("relative_path", "")) / "summary.json"
            if summary_path.is_file():
                sj = json.loads(summary_path.read_text(encoding="utf-8"))
                dl = np.asarray(sj.get("dwell_L_samples") or [], dtype=np.float64)
                dh = np.asarray(sj.get("dwell_H_samples") or [], dtype=np.float64)
                from continuous_patterns.core.diagnostics_kinetic import compute_mean_dwell_period

                mp = compute_mean_dwell_period(dl, dh, int(sj.get("n_transitions", 0)))
                md_o = mp["mean_dwell_period"]

        pk_o = orig.get("peak_period") if orig else None

        cmp_row = {
            "c1": c1,
            "c2": c2,
            "n_transitions_original": n_o,
            "n_transitions_fine": n_tr,
            "peak_period_fft_original": pk_o,
            "peak_period_fft_fine": pk,
            "mean_dwell_period_original": md_o,
            "mean_dwell_period_fine": md,
            "converged_n_transitions": _rel_tol_ok_counts(
                int(n_o) if n_o is not None else None,
                int(n_tr),
            ),
            "converged_mean_dwell_period": _rel_tol_ok(
                float(md_o) if md_o is not None else None,
                float(md) if md is not None else None,
            ),
        }
        cmp_row["converged_both"] = bool(
            cmp_row["converged_n_transitions"] and cmp_row["converged_mean_dwell_period"]
        )
        comparisons.append(cmp_row)

    manifest: dict[str, Any] = {
        "sweep_name": str(sweep_meta.get("name", "slab_kinetic_oscillation_window_fine")),
        "timestamp": stamp,
        "source_sweep_yaml": str(
            _REPO / "experiments/sweeps/slab_kinetic_oscillation_window_fine.yaml"
        ),
        "results_root": str(sweep_dir.resolve()),
        "nested_results": True,
        "main_sweep_reference": str(main_dir),
        "total_wall_time_s": float(total_wall),
        "comparison_with_main": comparisons,
        "entries": entries,
    }

    (sweep_dir / "manifest_axes.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # Markdown tables
    lines = [
        "| c1 | c2 | n_trans (orig) | n_trans (fine) | fft peak (orig) | fft peak (fine) | "
        "mean_dwell (orig) | mean_dwell (fine) | conv n | conv τ | conv both |",
        "|----|----|----------------|----------------|----------------|----------------|"
        "-------------------|-------------------|--------|--------|-----------|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['c1']} | {c['c2']} | {c['n_transitions_original']} | "
            f"{c['n_transitions_fine']} | "
            f"{c['peak_period_fft_original']} | {c['peak_period_fft_fine']} | "
            f"{c['mean_dwell_period_original']} | {c['mean_dwell_period_fine']} | "
            f"{c['converged_n_transitions']} | {c['converged_mean_dwell_period']} | "
            f"{c['converged_both']} |"
        )
    (sweep_dir / "table_comparison_main_vs_fine.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    mini = "\n".join(
        [
            "| run_label | c1 | c2 | status | n_transitions | "
            "mean_dwell_period | peak_period | wall_time_s |",
            "|-------------|----|----|--------|---------------|"
            "|-------------------|-------------|-------------|",
        ]
        + [
            f"| {e['run_label']} | {e['c1']} | {e['c2']} | {e['status']} | {e['n_transitions']} | "
            f"{e.get('mean_dwell_period')} | {e.get('peak_period')} | {e['wall_time_s']:.4g} |"
            for e in entries
        ]
    )
    (sweep_dir / "table_all_runs.md").write_text(mini + "\n", encoding="utf-8")

    n_conv_n = sum(1 for c in comparisons if c["converged_n_transitions"])
    n_conv_t = sum(1 for c in comparisons if c["converged_mean_dwell_period"])
    n_both = sum(1 for c in comparisons if c["converged_both"])
    stats = (
        f"convergence_tol=20%  n_transitions: {n_conv_n}/10  "
        f"mean_dwell_period: {n_conv_t}/10  both: {n_both}/10\n"
    )
    (sweep_dir / "convergence_stats.txt").write_text(stats, encoding="utf-8")

    print(stats)
    print(f"Wrote manifest + tables under {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
