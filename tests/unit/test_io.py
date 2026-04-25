"""Unit tests for :mod:`continuous_patterns.core.io`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from continuous_patterns.core.io import (
    allocate_run_dir,
    load_run_config,
    save_final_state_npz,
    save_run_config,
    save_summary,
)
from continuous_patterns.core.plotting import (
    parse_run_stamp_utc,
    plot_fields_final,
    write_evolution_gif,
)


def test_load_run_config_rejects_flat_yaml(tmp_path: Path) -> None:
    flat = tmp_path / "flat.yaml"
    flat.write_text(
        yaml.safe_dump({"grid": 64, "L": 10.0, "model": "agate_ch"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experiment"):
        load_run_config(flat)


def test_load_run_config_fills_output_from_defaults(tmp_path: Path) -> None:
    """Merged library defaults supply ``output`` when the experiment YAML omits it."""
    p = tmp_path / "minimal.yaml"
    p.write_text(
        """
experiment:
  name: demo
  model: agate_ch
  seed: 1
geometry:
  type: circular_cavity
  L: 4.0
  R: 1.5
  n: 32
physics:
  W: 1.0
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.05
  T: 1.0
  snapshot_every: 10
""",
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["time"]["dt"] == 0.05
    assert cfg["output"]["flux_sample_dt"] == 2.0
    assert cfg["output"]["log_level"] == "INFO"
    assert cfg["output"]["record_spectral_mass_diagnostic"] is True


def test_load_run_config_user_settings_override(tmp_path: Path) -> None:
    """Explicit ``user_settings_path`` overrides library defaults."""
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        """
experiment:
  name: demo
  model: agate_ch
  seed: 1
geometry:
  type: circular_cavity
  L: 4.0
  R: 1.5
  n: 32
physics:
  W: 1.0
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.01
  T: 1.0
  snapshot_every: 10
output:
  save_final_state: true
""",
        encoding="utf-8",
    )
    user = tmp_path / "user.yaml"
    user.write_text("output:\n  log_level: DEBUG\n", encoding="utf-8")
    cfg = load_run_config(exp, user_settings_path=user)
    assert cfg["output"]["log_level"] == "DEBUG"


def test_load_run_config_validates_nested_schema(tmp_path: Path) -> None:
    p = tmp_path / "nested.yaml"
    p.write_text(
        """
experiment:
  name: demo
  model: agate_ch
  seed: 1
geometry:
  type: circular_cavity
  L: 4.0
  R: 1.5
  n: 32
physics:
  W: 1.0
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.01
  T: 1.0
  snapshot_every: 10
output:
  save_final_state: true
""",
        encoding="utf-8",
    )
    cfg = load_run_config(p)
    assert cfg["experiment"]["model"] == "agate_ch"
    assert cfg["geometry"]["n"] == 32
    assert cfg["physics"]["W"] == 1.0


def test_expensive_output_hard_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML GIF/HDF5 toggles are forced off unless CP_ALLOW_EXPENSIVE_OUTPUT=1."""
    import continuous_patterns.core.io as io_mod

    yaml_body = """
experiment:
  name: expensive_flags
  model: agate_ch
  seed: 1
geometry:
  type: circular_cavity
  L: 4.0
  R: 1.5
  n: 32
physics:
  W: 1.0
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.01
  T: 1.0
  snapshot_every: 10
output:
  save_final_state: true
  record_evolution_gif: true
  save_snapshots_h5: true
"""
    p = tmp_path / "expensive_flags.yaml"
    p.write_text(yaml_body, encoding="utf-8")

    io_mod._warned_gif_hard_disabled = False
    io_mod._warned_h5_hard_disabled = False
    monkeypatch.delenv("CP_ALLOW_EXPENSIVE_OUTPUT", raising=False)
    cfg = load_run_config(p)
    assert cfg["output"]["record_evolution_gif"] is False
    assert cfg["output"]["save_snapshots_h5"] is False

    io_mod._warned_gif_hard_disabled = False
    io_mod._warned_h5_hard_disabled = False
    monkeypatch.setenv("CP_ALLOW_EXPENSIVE_OUTPUT", "1")
    cfg_allowed = load_run_config(p)
    assert cfg_allowed["output"]["record_evolution_gif"] is True
    assert cfg_allowed["output"]["save_snapshots_h5"] is True


def test_allocate_run_dir_creates_timestamped_layout(tmp_path: Path) -> None:
    rp = allocate_run_dir(experiment_name="agate_ch", results_root=tmp_path)
    assert rp.root.is_dir()
    assert rp.summary_json == rp.root / "summary.json"
    assert rp.config_yaml == rp.root / "config.yaml"
    assert rp.final_state_npz == rp.root / "final_state.npz"
    assert rp.log_file == rp.root / "run.log"
    assert rp.snapshots_h5 == rp.root / "snapshots.h5"
    assert rp.jablczynski_plot == rp.root / "jablczynski.png"
    assert rp.root.parent.name == "agate_ch"


def test_save_summary_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    payload = {"ok": True, "loss": 0.25, "nested": {"a": 1}}
    save_summary(out, payload)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


def test_roundtrip_save_load_config_identical(tmp_path: Path) -> None:
    cfg = {
        "experiment": {"name": "t1", "model": "agate_stage2", "seed": 7},
        "geometry": {"type": "circular_cavity", "L": 2.0, "R": 0.0, "n": 16},
        "physics": {"gamma": 2.0, "W": 1.0},
        "time": {"dt": 0.001, "T": 0.01, "snapshot_every": 5},
        "output": {"save_final_state": True},
    }
    path = tmp_path / "config.yaml"
    save_run_config(path, cfg)
    once = load_run_config(path)
    path2 = tmp_path / "config2.yaml"
    save_run_config(path2, once)
    twice = load_run_config(path2)
    assert once == twice


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
    out = plot_fields_final(pm, pc, cc, L=L, R=0.2, path=tmp_path)
    assert out.name == "figures_final.png"
    assert out.is_file()
    baseline_sz = out.stat().st_size
    assert baseline_sz > 0

    out_titled = plot_fields_final(
        pm,
        pc,
        cc,
        L=L,
        R=0.2,
        path=tmp_path / "titled.png",
        title="test_run — 2026-04-24 12:00 UTC",
        include_params_panel=False,
    )
    assert out_titled.is_file()
    assert out_titled.stat().st_size >= baseline_sz * 0.9


def test_plot_fields_final_includes_params_panel(tmp_path: Path) -> None:
    n = 12
    L = 1.0
    rng = np.random.default_rng(0)
    pm = rng.random((n, n))
    pc = rng.random((n, n))
    cc = rng.random((n, n))
    params = {
        "experiment": {"model": "agate_ch", "name": "test"},
        "geometry": {"L": 1.0, "R": 0.2, "n": 12},
        "physics": {"gamma": 3.0, "kappa_x": 0.5, "kappa_y": 0.5, "use_ratchet": True},
        "stress": {"mode": "none", "sigma_0": 0.0, "stress_coupling_B": 0.0},
        "time": {"dt": 0.01, "T": 1.0},
        "_diagnostics": {
            "spectral_mass_drift": {"leak_pct": 1.2e-6},
            "dirichlet_mass_balance": {"residual_pct": 0.05, "ratio": 1.002},
            "surface_flux_balance": {
                "leak_pct": 0.58,
                "n_samples": 42,
                "front_arrival_t": 88.0,
            },
            "jab_canonical": {"n_bands": 15, "q_cv": 0.12},
            "wall_time_s": 123.4,
        },
    }
    out = plot_fields_final(
        pm,
        pc,
        cc,
        L=L,
        R=0.2,
        path=tmp_path,
        title="test_run — 2026-04-24 12:00 UTC",
        params=params,
        include_params_panel=True,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    out_nopanel = plot_fields_final(
        pm,
        pc,
        cc,
        L=L,
        R=0.2,
        path=tmp_path / "nopanel",
        include_params_panel=False,
    )
    assert out_nopanel.is_file()
    assert out.stat().st_size > out_nopanel.stat().st_size


def test_write_evolution_gif_writes_file(tmp_path: Path) -> None:
    n = 8
    L = 1.0
    R = 0.2
    a0 = np.linspace(0.0, 1.0, n * n, dtype=np.float64).reshape(n, n)
    a1 = np.linspace(1.0, 0.0, n * n, dtype=np.float64).reshape(n, n)
    snaps = [(0.0, a0), (1.0, a1)]
    out = tmp_path / "evo.gif"
    path = write_evolution_gif(snaps, out, L=L, R=R, fps=5, field_name="phi_m")
    assert path is not None
    assert path.is_file()
    assert path.stat().st_size > 0


def test_parse_run_stamp_utc() -> None:
    assert parse_run_stamp_utc("20260424T105328Z") == "2026-04-24 10:53 UTC"
    assert parse_run_stamp_utc("not-a-stamp") is None


def _base_card(geometry_block: str) -> str:
    return f"""
experiment:
  name: gtest
  model: agate_ch
  seed: 1
{geometry_block}
physics:
  W: 1.0
stress:
  mode: none
  sigma_0: 0.0
  stress_coupling_B: 0.0
time:
  dt: 0.01
  T: 1.0
  snapshot_every: 10
"""


def test_geometry_spec_elliptic_and_slot_load(tmp_path: Path) -> None:
    for block in (
        """
geometry:
  type: elliptic_cavity
  L: 40.0
  n: 32
  a: 8.0
  b: 6.0
  theta: 0.1
  eps_scale: 2.0
""",
        """
geometry:
  type: rectangular_slot
  L: 100.0
  n: 32
  width: 20.0
  height: 10.0
  theta: 0.0
  eps_scale: 2.0
""",
    ):
        p = tmp_path / "g.yaml"
        p.write_text(_base_card(block), encoding="utf-8")
        cfg = load_run_config(p)
        assert cfg["geometry"]["type"] in ("elliptic_cavity", "rectangular_slot")


def test_geometry_spec_wedge_and_polygon_load(tmp_path: Path) -> None:
    poly = """
geometry:
  type: polygon_cavity
  L: 100.0
  n: 32
  n_sides: 5
  R: 15.0
  theta_offset: 0.0
  eps_scale: 2.0
"""
    p1 = tmp_path / "poly.yaml"
    p1.write_text(_base_card(poly), encoding="utf-8")
    assert load_run_config(p1)["geometry"]["n_sides"] == 5

    wedge = """
geometry:
  type: wedge_cavity
  L: 200.0
  n: 32
  R_inner: 5.0
  R_outer: 40.0
  opening_angle: 1.2
  theta_center: 0.0
  eps_scale: 2.0
"""
    p2 = tmp_path / "wedge.yaml"
    p2.write_text(_base_card(wedge), encoding="utf-8")
    assert load_run_config(p2)["geometry"]["R_outer"] == 40.0


def test_geometry_spec_missing_required_raises(tmp_path: Path) -> None:
    bad_elliptic = """
geometry:
  type: elliptic_cavity
  L: 40.0
  n: 32
  a: 8.0
  eps_scale: 2.0
"""
    p = tmp_path / "bad_e.yaml"
    p.write_text(_base_card(bad_elliptic), encoding="utf-8")
    with pytest.raises(ValidationError, match="b"):
        load_run_config(p)


def test_geometry_spec_polygon_exclusive_and_wedge_order(tmp_path: Path) -> None:
    both = """
geometry:
  type: polygon_cavity
  L: 100.0
  n: 32
  n_sides: 6
  R: 10.0
  vertices: [[20, 20], [30, 20], [30, 30]]
  eps_scale: 2.0
"""
    p = tmp_path / "both.yaml"
    p.write_text(_base_card(both), encoding="utf-8")
    with pytest.raises(ValidationError, match="exclusively"):
        load_run_config(p)

    neither = """
geometry:
  type: polygon_cavity
  L: 100.0
  n: 32
  eps_scale: 2.0
"""
    p2 = tmp_path / "neither.yaml"
    p2.write_text(_base_card(neither), encoding="utf-8")
    with pytest.raises(ValidationError, match="exclusively"):
        load_run_config(p2)

    bad_w = """
geometry:
  type: wedge_cavity
  L: 200.0
  n: 32
  R_inner: 40.0
  R_outer: 20.0
  opening_angle: 1.0
  eps_scale: 2.0
"""
    p3 = tmp_path / "bad_w.yaml"
    p3.write_text(_base_card(bad_w), encoding="utf-8")
    with pytest.raises(ValidationError, match="R_inner"):
        load_run_config(p3)
