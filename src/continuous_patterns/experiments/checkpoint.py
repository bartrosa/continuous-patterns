"""Checkpoint helpers for resuming long experiment batches.

A **complete** single run has ``config.yaml``, ``summary.json``, and
``final_state.npz`` under its timestamp directory. Comparison sweeps write
``comparison_progress.json`` after each row finishes so interrupted matrix
jobs can skip finished work.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from continuous_patterns.core.io import ResultPaths
from continuous_patterns.core.types import SimResult, SimState

COMPARISON_PROGRESS_FILE = "comparison_progress.json"


def is_run_complete(run_dir: Path) -> bool:
    """Return True when standard run artifacts indicate a finished simulation."""
    root = Path(run_dir)
    return (
        (root / "config.yaml").is_file()
        and (root / "summary.json").is_file()
        and (root / "final_state.npz").is_file()
    )


def result_paths_from_run_dir(run_dir: Path) -> ResultPaths:
    """Rebuild :class:`ResultPaths` for an existing timestamp directory."""
    root = Path(run_dir)
    return ResultPaths(
        root=root,
        summary_json=root / "summary.json",
        config_yaml=root / "config.yaml",
        final_state_npz=root / "final_state.npz",
        log_file=root / "run.log",
        snapshots_h5=root / "snapshots.h5",
        jablczynski_plot=root / "jablczynski.png",
    )


def find_newest_complete_run(search_root: Path, experiment_name: str) -> Path | None:
    """Return the newest complete run under ``search_root / experiment_name / *``."""
    base = Path(search_root) / experiment_name
    if not base.is_dir():
        return None
    complete = [p for p in base.iterdir() if p.is_dir() and is_run_complete(p)]
    if not complete:
        return None
    return max(complete, key=lambda p: p.name)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_comparison_progress(out_dir: Path) -> dict[str, Any]:
    """Load ``comparison_progress.json`` or return an empty document."""
    path = Path(out_dir) / COMPARISON_PROGRESS_FILE
    if not path.is_file():
        return {"completed": {}, "failed": {}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return {"completed": {}, "failed": {}}
    doc.setdefault("completed", {})
    doc.setdefault("failed", {})
    return doc


def save_comparison_progress(out_dir: Path, progress: dict[str, Any]) -> None:
    """Persist comparison progress (atomic write)."""
    progress = dict(progress)
    progress["updated_at"] = datetime.now(tz=UTC).isoformat()
    atomic_write_json(Path(out_dir) / COMPARISON_PROGRESS_FILE, progress)


def mark_comparison_run_complete(
    out_dir: Path,
    *,
    run_label: str,
    run_dir: Path,
    model: str,
    wall_time_s: float | None = None,
) -> None:
    """Record a successfully finished comparison row."""
    progress = load_comparison_progress(out_dir)
    completed = dict(progress.get("completed", {}))
    completed[str(run_label)] = {
        "run_dir": str(Path(run_dir).resolve()),
        "model": str(model),
        "wall_time_s": wall_time_s,
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    failed = dict(progress.get("failed", {}))
    failed.pop(str(run_label), None)
    save_comparison_progress(
        out_dir,
        {
            **progress,
            "completed": completed,
            "failed": failed,
        },
    )


def mark_comparison_run_failed(
    out_dir: Path,
    *,
    run_label: str,
    error: str,
) -> None:
    """Record a failed comparison row (does not block resume retries)."""
    progress = load_comparison_progress(out_dir)
    failed = dict(progress.get("failed", {}))
    failed[str(run_label)] = {
        "error": str(error),
        "failed_at": datetime.now(tz=UTC).isoformat(),
    }
    save_comparison_progress(out_dir, {**progress, "failed": failed})


def load_sim_result_from_run_dir(run_dir: Path, model: str) -> SimResult:
    """Rebuild a minimal :class:`SimResult` from on-disk run artifacts."""
    run_dir = Path(run_dir)
    if not is_run_complete(run_dir):
        raise FileNotFoundError(f"incomplete run directory: {run_dir}")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json must be a mapping: {run_dir}")
    if not isinstance(cfg, dict):
        raise ValueError(f"config.yaml must be a mapping: {run_dir}")

    import jax.numpy as jnp

    paths = result_paths_from_run_dir(run_dir)
    model = str(model)

    if model == "slab_kinetic":
        with np.load(run_dir / "history.npz") as hist:
            t_history = np.asarray(hist["t_history"], dtype=np.float64)
            c_at_x0_history = np.asarray(hist["c_at_x0_history"], dtype=np.float64)
            state_history = np.asarray(hist["state_history"], dtype=np.float64)
            thickness_history = np.asarray(hist["thickness_history"], dtype=np.float64)
        with np.load(run_dir / "final_state.npz") as fin:
            c_final = np.asarray(fin["c_final"], dtype=np.float64).ravel()
            t_final = float(fin["t_final"])
            thickness_final = float(fin["thickness_final"])
            state_final_str = str(fin["state_final"])

        transitions: list[tuple[float, str]] = []
        raw_tr = summary.get("kinetic_transitions")
        if isinstance(raw_tr, list):
            for item in raw_tr:
                if isinstance(item, dict) and "t" in item and "kind" in item:
                    transitions.append((float(item["t"]), str(item["kind"])))

        n = int(c_final.size)
        zeros1 = jnp.zeros((n, 1))
        state = SimState(
            phi_m=zeros1,
            phi_c=zeros1,
            phi_q=zeros1,
            phi_imp=zeros1,
            c=jnp.asarray(c_final[:, None], dtype=jnp.float64),
            t=t_final,
        )
        meta = {
            "slab_kinetic": {
                "t_history": t_history,
                "c_at_x0_history": c_at_x0_history,
                "state_history": state_history,
                "thickness_history": thickness_history,
                "thickness_final": thickness_final,
                "t_final": t_final,
                "state_final_str": state_final_str,
                "transitions": transitions,
            }
        }
        return SimResult(
            state_final=state,
            meta=meta,
            diagnostics=dict(summary),
            config_resolved=cfg,
            paths=paths,
        )

    if model == "slab_twophase":
        with np.load(run_dir / "final_state.npz") as fin:
            phi_m = np.asarray(fin["phi_m"], dtype=np.float64).ravel()
            phi_c = np.asarray(fin["phi_c"], dtype=np.float64).ravel()
            c = np.asarray(fin["c"], dtype=np.float64).ravel()
            t_final = float(fin["t_final"])
        gcfg = cfg.get("geometry", {})
        domain_l = float(gcfg.get("L", 100.0))
        n = int(phi_m.size)
        seed_width_px = int(gcfg.get("seed_width_px", 2))
        dx = domain_l / max(n, 1)

        snap_t = np.asarray([], dtype=np.float64)
        snap_c0 = np.asarray([], dtype=np.float64)
        snap_cf = np.asarray([], dtype=np.float64)
        snap_xf = np.asarray([], dtype=np.float64)
        snap_solid = np.zeros((0, n), dtype=np.float64)
        snap_path = run_dir / "snapshots.npz"
        if snap_path.is_file():
            with np.load(snap_path) as snap:
                if "snapshots_t" in snap.files:
                    snap_t = np.asarray(snap["snapshots_t"], dtype=np.float64)
                if "snapshots_c0" in snap.files:
                    snap_c0 = np.asarray(snap["snapshots_c0"], dtype=np.float64)
                if "snapshots_c_front" in snap.files:
                    snap_cf = np.asarray(snap["snapshots_c_front"], dtype=np.float64)
                if "snapshots_x_front" in snap.files:
                    snap_xf = np.asarray(snap["snapshots_x_front"], dtype=np.float64)
                if "snapshots_solid" in snap.files:
                    snap_solid = np.asarray(snap["snapshots_solid"], dtype=np.float64)
                if "domain_L" in snap.files:
                    domain_l = float(snap["domain_L"])

        zeros1 = jnp.zeros((n, 1))
        state = SimState(
            phi_m=jnp.asarray(phi_m[:, None], dtype=jnp.float64),
            phi_c=jnp.asarray(phi_c[:, None], dtype=jnp.float64),
            phi_q=zeros1,
            phi_imp=zeros1,
            c=jnp.asarray(c[:, None], dtype=jnp.float64),
            t=t_final,
        )
        phi_tot = phi_m + phi_c
        meta_block: dict[str, Any] = {
            "geometry": {"L": domain_l, "n": n, "seed_width_px": seed_width_px, "dx": dx},
            "demixing": str((cfg.get("physics") or {}).get("demixing", "ac")),
            "phi_tot_final": phi_tot,
            "snapshots_t": snap_t,
            "snapshots_solid": snap_solid,
            "snapshots_c0": snap_c0,
            "snapshots_c_front": snap_cf,
            "snapshots_x_front": snap_xf,
            "domain_L": domain_l,
            "t_final": t_final,
            "seed_exclude_x": float(3 * seed_width_px * dx),
        }
        return SimResult(
            state_final=state,
            meta={"slab_twophase": meta_block},
            diagnostics=dict(summary),
            config_resolved=cfg,
            paths=paths,
        )

    raise ValueError(f"load_sim_result_from_run_dir: unsupported model {model!r}")
