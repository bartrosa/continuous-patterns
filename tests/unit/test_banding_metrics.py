"""Unit tests for model-agnostic banding metrics."""

from __future__ import annotations

import numpy as np
import pytest

from continuous_patterns.core.banding_metrics import (
    BandingRecord,
    alternating_zone_metrics,
    compute_front_origin,
    compute_liesegang_laws,
    deposit_extent,
    profile_to_label_segments,
    record_from_label_history,
    record_from_profile,
    s1_twophase_validity,
)
from continuous_patterns.core.diagnostics_slab import compute_slab_axial_diagnostics


def test_jablczynski_recovers_p() -> None:
    n = 20
    p_true = 0.15
    x = 0.1 * (1.0 + p_true) ** np.arange(n, dtype=np.float64)
    rec = BandingRecord(
        source="profile",
        model="test",
        run_label="synthetic_geom",
        band_positions=x,
        band_widths=np.full_like(x, 0.01),
        band_labels=["m"] * n,
        band_times=None,
        front_pos_t=None,
        c_front_t=None,
        domain_length=float(x[-1]),
    )
    laws = compute_liesegang_laws(rec)
    assert abs(float(laws["p_fit"]) - p_true) < 1e-3
    assert float(laws["Q_cv"]) < 1e-6


def test_profile_peaks_positions() -> None:
    n = 512
    L = 100.0
    x = (np.arange(n, dtype=np.float64) + 0.5) * (L / n)
    sigma = 1.0
    centers = [10.0, 25.0, 40.0, 60.0, 80.0]
    profile = np.zeros_like(x)
    for c in centers:
        profile += np.exp(-0.5 * ((x - c) / sigma) ** 2)

    rec = record_from_profile(
        model="test",
        run_label="gaussians",
        profile_x=x,
        profile=profile,
        domain_length=L,
    )
    peak_pos = rec.extra["peak_positions"]
    dx = L / n
    for obs, exp in zip(peak_pos, centers, strict=True):
        assert abs(float(obs) - exp) <= 0.5 * dx
    assert rec.band_positions.size >= len(centers)


def test_label_record_segments() -> None:
    transitions = [(1.0, "L→H"), (2.0, "H→L"), (3.0, "L→H"), (4.0, "H→L")]
    t_hist = np.linspace(0.0, 5.0, 51)
    knots_t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    knots_th = [0.0, 0.2, 0.5, 0.9, 1.2, 1.4]
    th_hist = np.interp(t_hist, knots_t, knots_th)
    c0 = np.linspace(0.5, 0.1, t_hist.size)
    st = np.zeros(t_hist.size, dtype=np.float64)

    rec = record_from_label_history(
        model="slab_kinetic",
        run_label="synthetic_labels",
        t_history=t_hist,
        thickness_history=th_hist,
        state_history=st,
        transitions=transitions,
        c_at_x0_history=c0,
        domain_length=1.4,
    )
    assert rec.band_positions.size == 5
    assert rec.band_widths.size == 5
    assert abs(float(np.sum(rec.band_widths)) - 1.4) < 1e-12
    widths_expected = np.diff(np.asarray(knots_th, dtype=np.float64))
    np.testing.assert_allclose(rec.band_widths, widths_expected, rtol=0.0, atol=1e-12)


def test_time_law_alpha() -> None:
    t = np.linspace(0.1, 10.0, 30)
    x = np.sqrt(t)
    rec = record_from_label_history(
        model="slab_kinetic",
        run_label="sqrt_time",
        t_history=t,
        thickness_history=x,
        state_history=np.zeros_like(t),
        transitions=[],
        c_at_x0_history=np.zeros_like(t),
        domain_length=float(x[-1]),
    )
    rec.band_positions = x
    rec.band_widths = np.full_like(x, 0.01)
    rec.band_times = t
    laws = compute_liesegang_laws(rec)
    alpha = float(laws["time_law"]["alpha"])
    assert abs(alpha - 0.5) < 0.02


def test_shared_peak_detection_matches_legacy() -> None:
    n = 256
    L = 100.0
    x = (np.arange(n, dtype=np.float64) + 0.5) * (L / n)
    centers = [10.0, 25.0, 40.0, 60.0, 80.0]
    width = 1.4
    profile = np.zeros_like(x)
    for c in centers:
        profile += np.exp(-0.5 * ((x - c) / width) ** 2)
    phi_m = np.broadcast_to(profile[:, None], (n, n))
    phi_c = np.zeros((n, n), dtype=np.float64)

    legacy = compute_slab_axial_diagnostics(phi_m, phi_c, L=L)
    rec = record_from_profile(
        model="slab_reactive",
        run_label="legacy_check",
        profile_x=x,
        profile=profile,
        domain_length=L,
    )
    assert int(legacy["band_count_axial"]) == int(rec.extra["n_peaks"])
    np.testing.assert_allclose(
        legacy["peak_positions_x"],
        rec.extra["peak_positions"],
        rtol=0.0,
        atol=1e-12,
    )


def test_front_origin_left() -> None:
    n_x = 40
    n_t = 30
    x = np.linspace(0.0, 1.0, n_x)
    times = np.linspace(0.0, 10.0, n_t)
    solid = np.zeros((n_t, n_x), dtype=np.float64)
    for i, t in enumerate(times):
        front_ix = int(round((t / times[-1]) * (n_x - 1)))
        solid[i, : front_ix + 1] = 1.0
    out = compute_front_origin(solid_xt=solid, frame_times=times, x=x, threshold=0.3)
    assert out["origin"] == "LEFT"
    assert float(out["spearman_rho"]) > 0.8


def test_front_origin_right() -> None:
    n_x = 40
    n_t = 30
    x = np.linspace(0.0, 1.0, n_x)
    times = np.linspace(0.0, 10.0, n_t)
    solid = np.zeros((n_t, n_x), dtype=np.float64)
    for i, t in enumerate(times):
        front_ix = int(round((1.0 - t / times[-1]) * (n_x - 1)))
        solid[i, front_ix:] = 1.0
    out = compute_front_origin(solid_xt=solid, frame_times=times, x=x, threshold=0.3)
    assert out["origin"] == "RIGHT"
    assert float(out["spearman_rho"]) < -0.8


def test_front_origin_both() -> None:
    n_x = 41
    n_t = 40
    x = np.linspace(0.0, 100.0, n_x)
    times = np.linspace(0.0, 20.0, n_t)
    solid = np.zeros((n_t, n_x), dtype=np.float64)
    mid = n_x // 2
    for i, t in enumerate(times):
        reach = int(round((t / times[-1]) * mid))
        solid[i, :reach] = 1.0
        solid[i, n_x - reach :] = 1.0
    out = compute_front_origin(solid_xt=solid, frame_times=times, x=x, threshold=0.3)
    assert out["origin"] == "BOTH"
    assert np.isfinite(float(out["meeting_x"]))
    assert abs(float(out["meeting_x"]) - float(x[mid])) < 15.0


def test_laws_gated_below_six_bands() -> None:
    rec = BandingRecord(
        source="label_history",
        model="test",
        run_label="five_bands",
        band_positions=np.linspace(1.0, 6.0, 5),
        band_widths=np.full(5, 0.2),
        band_labels=["m", "c", "m", "c", "m"],
        band_times=None,
        front_pos_t=None,
        c_front_t=None,
        domain_length=10.0,
    )
    laws = compute_liesegang_laws(rec)
    assert laws["laws_reason"] == "insufficient_bands"
    assert not np.isfinite(float(laws["p_fit"]))


def test_laws_available_at_six_bands() -> None:
    n = 6
    p_true = 0.12
    x = 0.5 * (1.0 + p_true) ** np.arange(n, dtype=np.float64)
    rec = BandingRecord(
        source="profile",
        model="test",
        run_label="six_bands",
        band_positions=x,
        band_widths=np.full(n, 0.01),
        band_labels=["m", "c"] * 3,
        band_times=None,
        front_pos_t=None,
        c_front_t=None,
        domain_length=float(x[-1]),
    )
    laws = compute_liesegang_laws(rec)
    assert laws["laws_reason"] is None
    assert np.isfinite(float(laws["p_fit"]))


def test_band_strip_full_coverage() -> None:
    t_hist = np.linspace(0.0, 5.0, 51)
    th_hist = np.linspace(0.0, 1.0, t_hist.size)
    rec = record_from_label_history(
        model="slab_kinetic",
        run_label="coverage",
        t_history=t_hist,
        thickness_history=th_hist,
        state_history=np.zeros(t_hist.size),
        transitions=[(1.0, "L→H"), (2.0, "H→L"), (3.0, "L→H")],
        c_at_x0_history=np.zeros(t_hist.size),
        domain_length=1.0,
    )
    extent = deposit_extent(rec.band_positions, rec.band_widths)
    assert abs(extent - float(th_hist[-1])) < 1e-12


def test_alternating_zone_metrics_terminal_block_excluded() -> None:
    pos = np.array([2.0, 5.0, 8.0, 11.0, 20.0])
    widths = np.array([1.0, 1.0, 1.0, 1.0, 8.0])
    labels = ["m", "c", "m", "c", "c"]
    zone = alternating_zone_metrics(pos, widths, labels)
    assert zone["n_bands_zone"] == 4
    assert zone["band_zone_extent"] == pytest.approx(8.5)
    assert zone["mean_spacing_zone"] == pytest.approx(3.0)
    assert zone["median_spacing_zone"] == pytest.approx(3.0)
    assert zone["mean_spacing_zone_excl_last"] == pytest.approx(3.0)


def test_alternating_zone_excl_last_omits_terminal_spacing() -> None:
    pos = np.array([0.0, 10.0, 20.0, 80.0, 100.0])
    widths = np.array([2.0, 2.0, 2.0, 2.0, 20.0])
    labels = ["m", "c", "m", "c", "c"]
    zone = alternating_zone_metrics(pos, widths, labels)
    assert zone["n_bands_zone"] == 4
    assert zone["mean_spacing_zone"] == pytest.approx(80.0 / 3.0)
    assert zone["median_spacing_zone"] == pytest.approx(10.0)
    assert zone["mean_spacing_zone_excl_last"] == pytest.approx(10.0)


def test_s1_validity_thresholds() -> None:
    assert s1_twophase_validity({"mass_residual_pct": 1e-7, "min_phi_w": 0.01}) == "OK"
    assert s1_twophase_validity({"mass_residual_pct": 5e-4, "min_phi_w": 0.0}) == "OK"
    assert s1_twophase_validity({"mass_residual_pct": 5e-3, "min_phi_w": 0.0}) == "WARN"
    assert s1_twophase_validity({"mass_residual_pct": 0.15, "min_phi_w": 0.0}) == (
        "NUMERICALLY_INVALID"
    )
    assert s1_twophase_validity({"mass_residual_pct": 0.0, "min_phi_w": -1e-5}) == (
        "NUMERICALLY_INVALID"
    )


def test_clip_rate() -> None:
    from continuous_patterns.core.banding_metrics import compute_clip_rate

    rate = compute_clip_rate(745_000, 23_450_605, 256)
    assert rate is not None
    assert 0.0 < rate < 1.0


def test_profile_to_label_segments_no_gaps() -> None:
    x = np.linspace(0.0, 10.0, 101)
    profile = np.sin(2.0 * np.pi * x / 10.0)
    pos, widths, labels = profile_to_label_segments(x, profile)
    assert pos.size == widths.size == len(labels)
    assert abs(float(np.sum(widths)) - (float(x[-1]) - float(x[0]))) < 0.5
