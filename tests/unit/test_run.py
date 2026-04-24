"""Unit tests for :mod:`continuous_patterns.experiments.run`."""

from __future__ import annotations

from pathlib import Path

import pytest

from continuous_patterns.core.io import load_run_config, save_run_config
from continuous_patterns.experiments.run import main, run_one

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_DIR = _REPO_ROOT / "src" / "continuous_patterns" / "experiments" / "templates"


def _minimal_agate_ch_cfg() -> dict:
    return {
        "experiment": {"name": "unit_agate", "model": "agate_ch", "seed": 0},
        "geometry": {"type": "circular_cavity", "L": 6.0, "R": 2.0, "n": 24},
        "physics": {
            "W": 1.0,
            "gamma": 1.0,
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
        "output": {
            "save_final_state": True,
            "flux_sample_dt": 0.1,
            "record_spectral_mass_diagnostic": False,
        },
    }


def test_run_one_dispatches_agate_ch(tmp_path: Path) -> None:
    """``run_one`` with agate_ch writes expected artifacts under ``results_root``."""
    cfg = _minimal_agate_ch_cfg()
    result = run_one(cfg, results_root=tmp_path, chunk_size=50)

    assert result.paths is not None
    assert result.paths.root.is_dir()
    assert result.paths.config_yaml.is_file()
    assert result.paths.summary_json.is_file()
    assert result.paths.final_state_npz.is_file()
    assert (result.paths.root / "figures_final.png").is_file()


def test_run_one_no_write(tmp_path: Path) -> None:
    """``write_artifacts=False`` skips directory allocation and files."""
    _ = tmp_path
    cfg = _minimal_agate_ch_cfg()
    result = run_one(cfg, write_artifacts=False, chunk_size=50)
    assert result.paths is None


def test_run_one_unknown_model() -> None:
    """Unknown ``experiment.model`` raises a clear ``ValueError``."""
    cfg = {
        "experiment": {"name": "x", "model": "nonexistent"},
        "geometry": {"type": "circular_cavity", "L": 4.0, "R": 1.0, "n": 16},
        "physics": {
            "W": 1.0,
            "gamma": 1.0,
            "kappa_x": 0.5,
            "kappa_y": 0.5,
            "M_m": 0.1,
            "M_c": 0.1,
            "D_c": 0.1,
            "k_rxn": 0.1,
            "c_sat": 0.0,
            "c_0": 0.5,
            "c_ostwald": 0.5,
            "w_ostwald": 0.1,
            "lambda_bar": 10.0,
        },
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 0.02, "snapshot_every": 10},
        "output": {},
    }
    with pytest.raises(ValueError, match="Unknown model"):
        run_one(cfg, write_artifacts=False, chunk_size=10)


def test_cli_main_reads_template(tmp_path: Path) -> None:
    """``main()`` runs a downsized copy of the agate_ch template."""
    template = _TEMPLATES_DIR / "agate_ch_baseline.yaml"
    cfg = load_run_config(template)
    cfg["geometry"]["n"] = 32
    cfg["time"]["T"] = 1.0

    cfg_path = tmp_path / "tiny.yaml"
    save_run_config(cfg_path, cfg)

    rc = main(
        [
            "--config",
            str(cfg_path),
            "--out-dir",
            str(tmp_path),
            "--chunk-size",
            "50",
        ]
    )
    assert rc == 0


def test_templates_load_successfully() -> None:
    """Shipped templates under ``experiments/templates/`` pass ``load_run_config``."""
    assert _TEMPLATES_DIR.is_dir()
    templates = sorted(_TEMPLATES_DIR.glob("*.yaml"))
    assert len(templates) >= 2
    for tmpl in templates:
        cfg = load_run_config(tmpl)
        assert cfg["experiment"]["model"] in ("agate_ch", "agate_stage2")
