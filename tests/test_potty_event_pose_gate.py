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
        # The trigger is dwell-only; default to a real (>0) dwell. Tests that want to
        # isolate the pose gate without a dwell trigger use a short hold instead.
        "dwell_trigger_s": 5.0,
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


def test_gate_on_dwell_event_strips_pose_squat_from_metadata() -> None:
    """With the gate ON, a dwell-triggered event records the pose summary but must NOT
    leak a ``pose_squat`` key: the trigger is dwell-only, so the gate's squat signal
    is no longer a decision input and must not appear as if it were training truth."""

    detector = PottyEventDetector(_camera_config(), pose_gate=_squat_gate())

    emitted = []
    for idx in range(8):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert len(emitted) == 1
    pose_summary = emitted[0].posture_summary["pose"]
    assert "pose_squat" not in pose_summary
    # The gate still ran and contributed its trustworthy stationarity signal.
    assert "pose_stationary" in pose_summary
    assert "pose_coverage" in pose_summary


def test_pose_squat_alone_does_not_trigger() -> None:
    """A short hold where pose reports a squat but the dog never dwells long enough
    must NOT emit: pose squat is no longer a trigger, only sustained dwell is."""

    detector = PottyEventDetector(_camera_config(dwell_trigger_s=5.0), pose_gate=_squat_gate())

    emitted = []
    for idx in range(4):  # only ~3s held: below the 5s dwell trigger
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert emitted == []


def test_same_short_window_without_gate_emits_nothing() -> None:
    """The same sub-dwell standing sequence with the gate OFF is also a non-event
    (gate-off path is unchanged)."""

    detector = PottyEventDetector(_camera_config(dwell_trigger_s=5.0))

    emitted = []
    for idx in range(4):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert emitted == []


def test_gate_off_dwell_event_has_no_pose_key() -> None:
    """A dwell event records no pose key when the gate is off."""

    detector = PottyEventDetector(_camera_config())

    emitted = []
    for idx in range(8):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert len(emitted) == 1
    assert "pose" not in emitted[0].posture_summary


def test_sparse_pose_does_not_fabricate_event() -> None:
    """When pose succeeds on too few frames the gate stays silent; a sub-dwell hold
    therefore does not emit (no fabricated recall)."""

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
    detector = PottyEventDetector(_camera_config(dwell_trigger_s=5.0), pose_gate=gate)

    emitted = []
    for idx in range(4):
        emitted.extend(detector.process(_frame(idx), [_detection(idx, _standing())]))
    emitted.extend(detector.flush())

    assert emitted == []


def test_pose_cannot_bypass_coverage_requirement() -> None:
    """Even with pose reporting squat AND stationary, a window that does not cover
    ``stationary_threshold_s`` must not emit — pose is additive, it never removes the
    bbox coverage gate (the recall fix). Here dwell would otherwise fire immediately
    (tiny trigger), so coverage is the only thing keeping the window a non-event."""

    config = _camera_config(
        stationary_threshold_s=2.0,
        dwell_trigger_s=0.1,
        event_duration_s=0.1,
    )
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
