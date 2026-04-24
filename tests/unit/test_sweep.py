"""Unit tests for :mod:`continuous_patterns.experiments.sweep`."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from continuous_patterns.experiments.sweep import (
    _set_dotted,
    main,
    run_sweep,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GAMMA_SWEEP_TEMPLATE = (
    _REPO_ROOT
    / "src"
    / "continuous_patterns"
    / "experiments"
    / "templates"
    / "sweeps"
    / "gamma_scan.yaml"
)


def _write_minimal_agate_ch_config(path: Path) -> None:
    cfg = {
        "experiment": {"name": "sweep_base", "model": "agate_ch", "seed": 0},
        "geometry": {"type": "circular_cavity", "L": 6.0, "R": 2.0, "n": 24},
        "physics": {
            "W": 1.0,
            "gamma": 2.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.05,
            "M_c": 0.05,
            "D_c": 0.05,
            "k_rxn": 0.2,
            "c_sat": 0.1,
            "c_0": 0.4,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "lambda_bar": 10.0,
            "use_ratchet": False,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 0.2, "snapshot_every": 50},
        "output": {"save_final_state": True, "flux_sample_dt": 0.1},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_set_dotted_creates_nested() -> None:
    cfg: dict = {}
    _set_dotted(cfg, "a.b.c", 42)
    assert cfg == {"a": {"b": {"c": 42}}}


def test_sweep_empty_grid_runs_base_config(tmp_path: Path) -> None:
    """Empty ``grid`` → a single run using base + overrides."""
    base = tmp_path / "base.yaml"
    _write_minimal_agate_ch_config(base)

    sweep_cfg = {
        "sweep": {"name": "empty", "base_config": str(base.resolve())},
        "overrides": {"geometry.n": 16, "time.T": 1.0},
    }

    result = run_sweep(sweep_cfg, results_root=tmp_path, chunk_size=50)
    assert len(result.entries) == 1
    assert result.entries[0]["status"] == "success"
    assert (result.sweep_root / "manifest.json").is_file()
    assert (result.sweep_root / "report.md").is_file()


def test_sweep_cartesian_expansion(tmp_path: Path) -> None:
    """Two-axis ``grid`` → full Cartesian product."""
    base = tmp_path / "base.yaml"
    _write_minimal_agate_ch_config(base)

    sweep_cfg = {
        "sweep": {"name": "cartesian", "base_config": str(base.resolve())},
        "grid": {
            "physics.gamma": [1.0, 2.0],
            "physics.W": [0.5, 1.0, 2.0],
        },
        "overrides": {"geometry.n": 16, "time.T": 0.5},
    }

    result = run_sweep(sweep_cfg, results_root=tmp_path, chunk_size=50)
    assert len(result.entries) == 6
    assert all(e["status"] == "success" for e in result.entries)


def test_sweep_manifest_captures_parameters_and_status(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    _write_minimal_agate_ch_config(base)
    sweep_cfg = {
        "sweep": {"name": "manifested", "base_config": str(base.resolve())},
        "grid": {"physics.gamma": [1.5, 2.5]},
        "overrides": {"geometry.n": 16, "time.T": 0.3},
    }
    result = run_sweep(sweep_cfg, results_root=tmp_path, chunk_size=50)
    data = json.loads((result.sweep_root / "manifest.json").read_text(encoding="utf-8"))
    assert data["sweep_name"] == "manifested"
    assert len(data["runs"]) == 2
    assert data["runs"][0]["parameters"] == {"physics.gamma": 1.5}
    assert data["runs"][0]["status"] == "success"
    assert data["runs"][0]["relative_path"] is not None


def test_sweep_cli_smoke(tmp_path: Path) -> None:
    """CLI ``main()`` runs a downsized copy of the shipped gamma sweep."""
    sweep_cfg = yaml.safe_load(_GAMMA_SWEEP_TEMPLATE.read_text(encoding="utf-8"))
    sweep_cfg["grid"]["physics.gamma"] = [1.0, 2.0]
    sweep_cfg["overrides"]["geometry.n"] = 16
    sweep_cfg["overrides"]["time.T"] = 0.5
    _tpl = _REPO_ROOT / "src" / "continuous_patterns" / "experiments" / "templates"
    sweep_cfg["sweep"]["base_config"] = str(_tpl / "agate_ch_baseline.yaml")

    sweep_path = tmp_path / "tiny_sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False), encoding="utf-8")

    rc = main(["--sweep", str(sweep_path), "--out-dir", str(tmp_path), "--chunk-size", "50"])
    assert rc == 0
