from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pose.base import build_synthetic_pose
from detectivepotty.pose.gate import PoseGate
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _frame(frame_idx: int) -> Frame:
    return Frame(
        bgr=np.zeros((120, 160, 3), dtype=np.uint8),
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS,
        source_id="camera://cam-1",
    )


def _detection(frame_idx: int, bbox: BBox) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=0.9,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS,
    )


def _squat_fn(frame: Frame, detection: Detection):
    return build_synthetic_pose(
        detection.bbox,
        frame_idx=detection.frame_idx,
        mono_ts=detection.mono_ts,
        posture="squat",
    )


def _bbox() -> BBox:
    return BBox(40, 20, 110, 100)


def test_gate_reports_squat_and_stationary_for_clean_window() -> None:
    gate = PoseGate(_squat_fn, min_required_frames=2, min_pose_coverage=0.5)
    detections = [_detection(idx, _bbox()) for idx in range(4)]
    for det in detections:
        gate.observe(_frame(det.frame_idx), [det])

    result = gate.posture(detections)

    assert result is not None
    assert result.pose_squat is True
    assert result.pose_stationary is True
    assert result.valid_frames == 4
    assert result.attempted_frames == 4


def test_gate_distrusts_sparse_window_even_if_individual_poses_are_good() -> None:
    # Pose succeeds on only 2 of 6 attempted detections -> coverage 0.33 < 0.5.
    succeed_on = {1, 4}

    def sparse_fn(frame: Frame, detection: Detection):
        if detection.frame_idx in succeed_on:
            return _squat_fn(frame, detection)
        return None

    gate = PoseGate(sparse_fn, min_required_frames=2, min_pose_coverage=0.5)
    detections = [_detection(idx, _bbox()) for idx in range(6)]
    for det in detections:
        gate.observe(_frame(det.frame_idx), [det])

    assert gate.posture(detections) is None


def test_gate_returns_none_below_min_required_frames() -> None:
    gate = PoseGate(_squat_fn, min_required_frames=3, min_pose_coverage=0.5)
    detections = [_detection(idx, _bbox()) for idx in range(2)]
    for det in detections:
        gate.observe(_frame(det.frame_idx), [det])

    assert gate.posture(detections) is None


def test_gate_estimator_failure_degrades_to_no_pose() -> None:
    def boom_fn(frame: Frame, detection: Detection):
        raise RuntimeError("backend exploded")

    gate = PoseGate(boom_fn, min_required_frames=2, min_pose_coverage=0.5)
    detections = [_detection(idx, _bbox()) for idx in range(4)]
    for det in detections:
        gate.observe(_frame(det.frame_idx), [det])

    # Attempts were recorded (as None), so coverage is 0 -> distrusted, no crash.
    assert gate.posture(detections) is None


def test_gate_fingerprint_guard_skips_stale_entries() -> None:
    gate = PoseGate(_squat_fn, min_required_frames=2, min_pose_coverage=0.5)
    detection = _detection(0, _bbox())
    gate.observe(_frame(0), [detection])
    # Mutate the detection after caching: the fingerprint (frame_idx, mono_ts) no
    # longer matches, so the stale pose must be ignored.
    detection.frame_idx = 99
    assert gate.posture([detection]) is None


def test_gate_prune_drops_unreferenced_ids() -> None:
    gate = PoseGate(_squat_fn, min_required_frames=2, min_pose_coverage=0.5)
    keep = _detection(0, _bbox())
    drop = _detection(1, _bbox())
    gate.observe(_frame(0), [keep, drop])

    gate.prune({id(keep)})

    # Only the kept detection still resolves to a cached pose.
    assert id(keep) in gate._by_id
    assert id(drop) not in gate._by_id
