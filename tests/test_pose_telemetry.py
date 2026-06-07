from __future__ import annotations

import pytest

from detectivepotty.pose.telemetry import (
    OUTCOME_BAD_SHAPE,
    OUTCOME_SKIP_TINY_CROP,
    OUTCOME_SUCCESS,
    OUTCOMES,
    PoseTelemetry,
)


def test_fresh_snapshot_is_empty() -> None:
    snap = PoseTelemetry().snapshot()
    assert snap.total_calls == 0
    assert all(snap.outcomes[name] == 0 for name in OUTCOMES)
    assert snap.latency_ms_count == 0
    assert snap.latency_ms_mean is None
    assert snap.latency_ms_p95 is None
    assert snap.cold_start_ms is None
    assert snap.build_failed is False
    assert snap.mean_kpt_conf is None
    assert snap.success_rate is None
    assert snap.started_at > 0
    assert snap.last_success_at is None


def test_record_success_updates_counts_latency_and_health() -> None:
    t = PoseTelemetry(conf_threshold=0.5)
    t.record(OUTCOME_SUCCESS, latency_ms=40.0, mean_conf=0.8, frac_ge=0.75, n_keypoints=30)
    snap = t.snapshot()
    assert snap.total_calls == 1
    assert snap.outcomes[OUTCOME_SUCCESS] == 1
    assert snap.latency_ms_count == 1
    assert snap.latency_ms_mean == pytest.approx(40.0)
    assert snap.mean_kpt_conf == pytest.approx(0.8)
    assert snap.mean_frac_conf_ge == pytest.approx(0.75)
    assert snap.mean_keypoints == pytest.approx(30.0)
    assert snap.success_rate == pytest.approx(1.0)
    assert snap.last_success_at is not None


def test_non_inference_outcome_records_no_latency() -> None:
    t = PoseTelemetry()
    t.record(OUTCOME_SKIP_TINY_CROP)
    snap = t.snapshot()
    assert snap.total_calls == 1
    assert snap.outcomes[OUTCOME_SKIP_TINY_CROP] == 1
    assert snap.latency_ms_count == 0
    assert snap.latency_ms_mean is None
    assert snap.last_success_at is None


def test_bad_shape_records_latency_but_not_success() -> None:
    t = PoseTelemetry()
    t.record(OUTCOME_BAD_SHAPE, latency_ms=12.0)
    snap = t.snapshot()
    assert snap.outcomes[OUTCOME_BAD_SHAPE] == 1
    assert snap.latency_ms_count == 1
    assert snap.mean_kpt_conf is None  # health only accrues on success


def test_outcomes_partition_total_calls() -> None:
    t = PoseTelemetry()
    t.record(OUTCOME_SUCCESS, latency_ms=10.0, mean_conf=0.9, frac_ge=1.0, n_keypoints=39)
    t.record(OUTCOME_SUCCESS, latency_ms=20.0, mean_conf=0.7, frac_ge=0.5, n_keypoints=20)
    t.record(OUTCOME_SKIP_TINY_CROP)
    t.record(OUTCOME_BAD_SHAPE, latency_ms=5.0)
    snap = t.snapshot()
    assert snap.total_calls == 4
    assert sum(snap.outcomes.values()) == snap.total_calls
    assert snap.success_rate == pytest.approx(0.5)
    assert snap.mean_kpt_conf == pytest.approx(0.8)  # (0.9 + 0.7) / 2 successes


def test_latency_percentiles() -> None:
    t = PoseTelemetry()
    for v in range(1, 101):
        t.record(OUTCOME_SUCCESS, latency_ms=float(v), mean_conf=0.5, frac_ge=1.0, n_keypoints=1)
    snap = t.snapshot()
    assert snap.latency_ms_count == 100
    assert snap.latency_ms_min == pytest.approx(1.0)
    assert snap.latency_ms_max == pytest.approx(100.0)
    assert snap.latency_ms_mean == pytest.approx(50.5)
    assert snap.latency_ms_p50 == pytest.approx(50.5)
    assert snap.latency_ms_p95 == pytest.approx(95.05)
    assert snap.latency_ms_p99 == pytest.approx(99.01)


def test_latency_window_bounds_percentiles_but_not_lifetime_stats() -> None:
    t = PoseTelemetry(latency_window=3)
    for v in (1.0, 2.0, 3.0, 100.0, 200.0):
        t.record(OUTCOME_SUCCESS, latency_ms=v, mean_conf=0.5, frac_ge=1.0, n_keypoints=1)
    snap = t.snapshot()
    # Lifetime count/min/max see all 5 samples...
    assert snap.latency_ms_count == 5
    assert snap.latency_ms_min == pytest.approx(1.0)
    assert snap.latency_ms_max == pytest.approx(200.0)
    # ...but percentiles only see the last 3 (3, 100, 200).
    assert snap.latency_ms_p50 == pytest.approx(100.0)


def test_record_cold_start() -> None:
    ok = PoseTelemetry()
    ok.record_cold_start(1234.0, ok=True)
    snap = ok.snapshot()
    assert snap.cold_start_ms == pytest.approx(1234.0)
    assert snap.build_failed is False

    bad = PoseTelemetry()
    bad.record_cold_start(50.0, ok=False)
    assert bad.snapshot().build_failed is True


def test_reset_clears_everything() -> None:
    t = PoseTelemetry()
    t.record(OUTCOME_SUCCESS, latency_ms=10.0, mean_conf=0.9, frac_ge=1.0, n_keypoints=39)
    t.record_cold_start(100.0, ok=True)
    before = t.snapshot().started_at
    t.reset()
    snap = t.snapshot()
    assert snap.total_calls == 0
    assert snap.latency_ms_count == 0
    assert snap.cold_start_ms is None
    assert snap.started_at >= before


def test_snapshot_to_dict_is_json_friendly() -> None:
    import json

    t = PoseTelemetry()
    t.record(OUTCOME_SUCCESS, latency_ms=10.0, mean_conf=0.9, frac_ge=1.0, n_keypoints=39)
    d = t.snapshot().to_dict()
    json.dumps(d)  # must not raise
    assert d["outcomes"][OUTCOME_SUCCESS] == 1
    assert d["latency_ms"]["count"] == 1
    assert d["cold_start"]["failed"] is False
