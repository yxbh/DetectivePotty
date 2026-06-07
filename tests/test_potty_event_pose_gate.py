from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from detectivepotty.config import CameraConfig
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pose.base import build_synthetic_pose
from detectivepotty.pose.gate import PoseGate
from detectivepotty.potty_event import PottyEventDetector
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _camera_config(**overrides: object) -> CameraConfig:
    values = {
        "id": "cam-1",
        "name": "Backyard",
        "detection_conf_threshold": 0.25,
        "event_duration_s": 2.0,
        "stationary_threshold_s": 1.0,
        "squat_threshold": 0.3,
        "sample_rate_fps": 1.0,
    }
    values.update(overrides)
    return CameraConfig(**values)


def _frame(idx: int, mono_ts: float | None = None) -> Frame:
    ts = float(idx) if mono_ts is None else mono_ts
    return Frame(
        bgr=np.zeros((120, 160, 3), dtype=np.uint8),
        frame_idx=idx,
        mono_ts=ts,
        wall_ts=BASE_TS + timedelta(seconds=ts),
        source_id="camera://cam-1",
    )


def _detection(idx: int, bbox: BBox, mono_ts: float | None = None) -> Detection:
    ts = float(idx) if mono_ts is None else mono_ts
    return Detection(
        bbox=bbox,
        confidence=0.9,
        class_name="dog",
        frame_idx=idx,
        mono_ts=ts,
        wall_ts=BASE_TS + timedelta(seconds=ts),
    )


def _standing(x: float = 40.0) -> BBox:
    return BBox(x, 20, x + 40, 100)


def _squat_gate(**kwargs: object) -> PoseGate:
    def squat_fn(frame: Frame, detection: Detection):
        return build_synthetic_pose(
            detection.bbox,
            frame_idx=detection.frame_idx,
            mono_ts=detection.mono_ts,
            posture="squat",
        )

    return PoseGate(squat_fn, min_required_frames=2, min_pose_coverage=0.5, **kwargs)


def test_pose_gate_promotes_bbox_non_squat_window_to_event() -> None:
    """A stationary dog whose *bbox* never reads as a squat still emits when pose
    reports a squat — pose adds recall the bbox heuristic missed."""

    detector = PottyEventDetector(_camera_config(), pose_gate=_squat_gate())

    emitted = []
    for idx in range(6):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert len(emitted) == 1
    candidate = emitted[0]
    assert candidate.squat_metric < 0.3  # bbox alone would not have qualified
    assert "pose" in candidate.posture_summary
    assert candidate.posture_summary["pose"]["pose_squat"] is True


def test_same_window_without_gate_emits_nothing() -> None:
    """Identical standing sequence with the gate OFF stays a non-event and carries
    no pose key (gate-off path is byte-for-byte unchanged)."""

    detector = PottyEventDetector(_camera_config())

    emitted = []
    for idx in range(6):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert emitted == []


def test_gate_off_posture_summary_has_no_pose_key() -> None:
    """An ordinary bbox squat event records no pose key when the gate is off."""

    detector = PottyEventDetector(_camera_config())

    def squat(x: float = 40.0) -> BBox:
        return BBox(x - 15, 35, x + 55, 85)

    boxes = [[_standing()], [_standing()], [squat()], [squat()], [squat()], [squat()]]
    emitted = []
    for idx, frame_boxes in enumerate(boxes):
        emitted.extend(
            detector.process(_frame(idx), [_detection(idx, b) for b in frame_boxes])
        )
    emitted.extend(detector.flush())

    assert len(emitted) == 1
    assert "pose" not in emitted[0].posture_summary


def test_sparse_pose_does_not_fabricate_event() -> None:
    """When pose succeeds on too few frames the gate stays silent, so a
    bbox-non-squat window does not emit (no fabricated recall)."""

    succeed_on = {1}

    def sparse_fn(frame: Frame, detection: Detection):
        if detection.frame_idx in succeed_on:
            return build_synthetic_pose(
                detection.bbox,
                frame_idx=detection.frame_idx,
                mono_ts=detection.mono_ts,
                posture="squat",
            )
        return None

    gate = PoseGate(sparse_fn, min_required_frames=2, min_pose_coverage=0.5)
    detector = PottyEventDetector(_camera_config(), pose_gate=gate)

    emitted = []
    for idx in range(6):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert emitted == []


def test_pose_cannot_bypass_coverage_requirement() -> None:
    """Even with pose reporting squat AND stationary, a window that does not cover
    ``stationary_threshold_s`` must not emit — pose is additive, it never removes
    the bbox coverage gate (the recall fix)."""

    config = _camera_config(stationary_threshold_s=2.0, event_duration_s=0.1)
    detector = PottyEventDetector(config, pose_gate=_squat_gate())

    # Two detections only 0.3s apart: a real trailing window far short of the 2.0s
    # threshold (coverage tolerance caps at 1.0s here), so covered_long_enough=False.
    emitted = []
    for idx, mono in enumerate((0.0, 0.3)):
        emitted.extend(
            detector.process(_frame(idx, mono), [_detection(idx, _standing(), mono)])
        )
    emitted.extend(detector.flush())

    assert emitted == []
