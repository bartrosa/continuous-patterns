"""Unit tests for :mod:`continuous_patterns.core.io`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from continuous_patterns.core.io import (
    allocate_run_dir,
    load_run_config,
    save_final_state_npz,
    save_run_config,
    save_summary,
)
from continuous_patterns.core.plotting import plot_fields_final


def test_load_run_config_rejects_flat_yaml(tmp_path: Path) -> None:
    flat = tmp_path / "flat.yaml"
    flat.write_text(
        yaml.safe_dump({"grid": 64, "L": 10.0, "model": "agate_ch"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experiment"):
        load_run_config(flat)


def test_load_run_config_validates_nested_schema(tmp_path: Path) -> None:
    p = tmp_path / "nested.yaml"
    p.write_text(
        """
experiment:
  name: demo
  model: agate_ch
grid:
  n: 32
  L: 4.0
physics:
  W: 1.0
""",
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["experiment"]["model"] == "agate_ch"
    assert cfg["grid"]["n"] == 32
    assert cfg["physics"]["W"] == 1.0


def test_allocate_run_dir_creates_timestamped_layout(tmp_path: Path) -> None:
    rp = allocate_run_dir(experiment_name="agate_ch", results_root=tmp_path)
    assert rp.root.is_dir()
    assert rp.summary_json == rp.root / "summary.json"
    assert rp.config_yaml == rp.root / "config.yaml"
    assert rp.final_state_npz == rp.root / "final_state.npz"
    assert rp.root.parent.name == "agate_ch"


def test_save_summary_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    payload = {"ok": True, "loss": 0.25, "nested": {"a": 1}}
    save_summary(out, payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


def test_roundtrip_save_load_config_identical(tmp_path: Path) -> None:
    cfg = {
        "experiment": {"name": "t1", "model": "agate_stage2"},
        "grid": {"n": 16, "L": 2.0},
        "physics": {"gamma": 2.0},
        "integration": {"dt": 0.001},
    }
    path = tmp_path / "config.yaml"
    save_run_config(path, cfg)
    again = load_run_config(path)
    assert again == cfg


def test_save_final_state_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "final.npz"
    phi_m = np.ones((4, 4), dtype=np.float32)
    phi_c = np.zeros((4, 4), dtype=np.float32)
    c = 0.5 * np.ones((4, 4), dtype=np.float32)
    chi = np.ones((4, 4), dtype=np.float32)
    save_final_state_npz(path, phi_m=phi_m, phi_c=phi_c, c=c, chi=chi)
    z = np.load(path)
    assert np.array_equal(z["phi_m"], phi_m)
    assert np.array_equal(z["chi"], chi)


def test_plot_fields_final_writes_png(tmp_path: Path) -> None:
    n = 12
    L = 1.0
    rng = np.random.default_rng(0)
    pm = rng.random((n, n))
    pc = rng.random((n, n))
    cc = rng.random((n, n))
    ch = np.ones((n, n))
    out = plot_fields_final(pm, pc, cc, L=L, R=0.2, path=tmp_path, chi=ch)
    assert out.name == "figures_final.png"
    assert out.is_file()
    assert out.stat().st_size > 0
