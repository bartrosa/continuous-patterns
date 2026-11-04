"""Unit tests for S1 Ostwald-window sweep helpers."""

from __future__ import annotations

from continuous_patterns.experiments.s1_ostwald_window import check_s1_anchor


def test_anchor_pass() -> None:
    ok, msg = check_s1_anchor(
        {
            "run_label": "S1_cost0p25_g4",
            "status": "success",
            "row_validity": "OK",
            "n_bands_zone": 5,
            "band_zone_extent": 36.0,
        }
    )
    assert ok
    assert msg.startswith("PASS")


def test_anchor_fail_n_bands() -> None:
    ok, msg = check_s1_anchor(
        {
            "run_label": "S1_cost0p25_g4",
            "status": "success",
            "row_validity": "OK",
            "n_bands_zone": 2,
            "band_zone_extent": 36.0,
        }
    )
    assert not ok
    assert "FAIL anchor" in msg


def test_anchor_skip_other_label() -> None:
    ok, msg = check_s1_anchor({"run_label": "S1_cost0p22_g4", "status": "success"})
    assert ok
    assert "SKIP" in msg
