"""Tests for experiment checkpoint / resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from continuous_patterns.experiments.checkpoint import (
    COMPARISON_PROGRESS_FILE,
    find_newest_complete_run,
    is_run_complete,
    load_sim_result_from_run_dir,
    mark_comparison_run_complete,
)


def _write_minimal_kinetic_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "test", "model": "slab_kinetic"},
                "geometry": {"L": 1.0, "n": 8},
                "physics": {"c1": 0.7, "c2": 0.6},
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "n_transitions": 2,
        "mean_dwell_period": 0.3,
        "thickness_final": 1.0,
        "wall_time_s": 1.0,
        "kinetic_transitions": [{"t": 0.5, "kind": "L→H"}],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    t = np.linspace(0, 1, 5)
    np.savez_compressed(
        run_dir / "history.npz",
        t_history=t,
        c_at_x0_history=np.full(5, 0.5),
        state_history=np.zeros(5),
        thickness_history=np.linspace(0, 1, 5),
    )
    np.savez_compressed(
        run_dir / "final_state.npz",
        c_final=np.full(8, 0.5),
        state_final=np.array("L"),
        thickness_final=1.0,
        t_final=1.0,
    )


def _write_minimal_twophase_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 16
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "tp", "model": "slab_twophase"},
                "geometry": {"L": 100.0, "n": n, "seed_width_px": 2},
                "physics": {"demixing": "ac", "bc_right": "noflux"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"mass_residual_pct": 0.0, "min_phi_w": 0.05, "wall_time_s": 2.0}),
        encoding="utf-8",
    )
    phi_m = np.linspace(0, 0.2, n)
    phi_c = np.linspace(0, 0.1, n)
    np.savez_compressed(
        run_dir / "final_state.npz",
        phi_m=phi_m,
        phi_c=phi_c,
        c=np.full(n, 0.3),
        t_final=100.0,
    )
    np.savez_compressed(
        run_dir / "snapshots.npz",
        snapshots_t=np.array([0.0, 50.0]),
        snapshots_solid=np.stack([phi_m + phi_c, phi_m + phi_c]),
        snapshots_c0=np.array([0.3, 0.3]),
        snapshots_x_front=np.array([0.0, 1.0]),
        domain_L=100.0,
    )


def test_is_run_complete_requires_all_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert not is_run_complete(run_dir)
    _write_minimal_kinetic_run(run_dir)
    assert is_run_complete(run_dir)


def test_find_newest_complete_run_picks_latest_stamp(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    old = root / "E1" / "20260101T000000Z"
    new = root / "E1" / "20260102T000000Z"
    _write_minimal_kinetic_run(old)
    _write_minimal_kinetic_run(new)
    found = find_newest_complete_run(root, "E1")
    assert found == new


def test_load_sim_result_from_kinetic_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_kinetic_run(run_dir)
    res = load_sim_result_from_run_dir(run_dir, "slab_kinetic")
    assert res.paths is not None
    assert res.diagnostics["n_transitions"] == 2
    assert "t_history" in res.meta["slab_kinetic"]


def test_load_sim_result_from_twophase_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_twophase_run(run_dir)
    res = load_sim_result_from_run_dir(run_dir, "slab_twophase")
    assert float(res.diagnostics["mass_residual_pct"]) == 0.0
    assert res.meta["slab_twophase"]["snapshots_t"].size == 2


def test_comparison_progress_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "E4" / "20260101T000000Z"
    _write_minimal_twophase_run(run_dir)
    mark_comparison_run_complete(
        tmp_path,
        run_label="E4_TP_gamma4_open",
        run_dir=run_dir,
        model="slab_twophase",
        wall_time_s=12.0,
    )
    progress_path = tmp_path / COMPARISON_PROGRESS_FILE
    assert progress_path.is_file()
    doc = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "E4_TP_gamma4_open" in doc["completed"]


def test_banding_cli_has_no_resume_flags() -> None:
    """v4: resume removed — every matrix run is fresh; finalize-outputs is export-only."""
    import importlib.util
    import inspect

    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts/run_banding_comparison.py"
    spec = importlib.util.spec_from_file_location("run_banding_comparison", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sig = inspect.signature(mod.run_comparison)
    assert "resume_dir" not in sig.parameters
    assert "finalize_only" not in sig.parameters
    assert hasattr(mod, "finalize_comparison_outputs")
