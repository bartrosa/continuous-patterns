"""HDF5 snapshot round-trip (``save_snapshots_h5`` / ``load_snapshots_h5``)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from continuous_patterns.core.io import load_snapshots_h5, save_snapshots_h5


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.h5"
    snaps = [
        {
            "step": 500,
            "t": 5.0,
            "phi_m": np.random.default_rng(0).random((32, 32)).astype(np.float32),
            "phi_c": np.random.default_rng(1).random((32, 32)).astype(np.float32),
            "c": np.random.default_rng(2).random((32, 32)).astype(np.float32),
        },
        {
            "step": 1000,
            "t": 10.0,
            "phi_m": np.random.default_rng(3).random((32, 32)).astype(np.float32),
            "phi_c": np.random.default_rng(4).random((32, 32)).astype(np.float32),
            "c": np.random.default_rng(5).random((32, 32)).astype(np.float32),
        },
    ]
    save_snapshots_h5(path, snaps, dt=0.01)
    loaded = load_snapshots_h5(path)

    assert len(loaded) == 2
    assert loaded[0]["step"] == 500
    assert loaded[1]["t"] == pytest.approx(10.0)
    np.testing.assert_allclose(loaded[0]["phi_m"], snaps[0]["phi_m"], rtol=1e-6)
    np.testing.assert_allclose(loaded[1]["c"], snaps[1]["c"], rtol=1e-6)


def test_save_empty_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    save_snapshots_h5(path, [], dt=0.01)
    loaded = load_snapshots_h5(path)
    assert loaded == []


def test_config_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "snap.h5"
    cfg = {"experiment": {"name": "test"}, "physics": {"gamma": 4.0}}
    snaps = [
        {
            "step": 100,
            "t": 1.0,
            "phi_m": np.zeros((8, 8), dtype=np.float32),
            "phi_c": np.zeros((8, 8), dtype=np.float32),
            "c": np.zeros((8, 8), dtype=np.float32),
        }
    ]
    save_snapshots_h5(path, snaps, dt=0.01, cfg_summary=cfg)

    with h5py.File(path, "r") as h5:
        stored = json.loads(h5["meta"].attrs["config_json"])
        assert stored["experiment"]["name"] == "test"
