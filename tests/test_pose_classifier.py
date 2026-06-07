"""Tests for the pose-based pee/poop classifier and its decision function."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.classify.pose import (
    PoseDecisionThresholds,
    PosePottyClassifier,
    classify_pose_features,
)
from detectivepotty.config import PoseConfig
from detectivepotty.events import ClassifierGuess, Detection, Track
from detectivepotty.geometry import BBox
from detectivepotty.pose.base import MockPoseEstimator, PoseEstimator, build_synthetic_pose
from detectivepotty.pose.features import PoseFeatures
from detectivepotty.pose.keypoints import PoseKeypoints
from detectivepotty.sources.base import Frame

_WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_window(n: int = 5, *, bbox: BBox | None = None, spacing: float = 0.2):
    bbox = bbox or BBox(20.0, 20.0, 120.0, 120.0)
    frames: list[Frame] = []
    detections: list[Detection] = []
    for i in range(n):
        bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        frames.append(
            Frame(bgr=bgr, frame_idx=i, mono_ts=i * spacing, wall_ts=_WALL, source_id="cam")
        )
        detections.append(
            Detection(
                bbox=bbox,
                confidence=0.9,
                class_name="dog",
                frame_idx=i,
                mono_ts=i * spacing,
                wall_ts=_WALL,
            )
        )
    return frames, Track(track_id="t1", detections=detections)


def _features(**overrides) -> PoseFeatures:
    base = dict(
        n_frames_total=5,
        n_frames_valid=5,
        coverage=1.0,
        dwell_duration_s=0.0,
        body_scale_px=100.0,
        body_scale_quality="torso",
        fallback_recommended=False,
    )
    base.update(overrides)
    return PoseFeatures(**base)


class _StubFallback(PottyClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, track: Track, frames: Sequence[Frame]) -> ClassifierResult:
        self.calls += 1
        return ClassifierResult(guess=ClassifierGuess.UNKNOWN, confidence=0.123)


class _NoneEstimator(PoseEstimator):
    def estimate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


class _SparseEstimator(PoseEstimator):
    """Returns valid torso keypoints but too few for a confident posture decision."""

    def estimate(self, frame_bgr, bbox, frame_idx=0, mono_ts=None, wall_ts=None, source_id=None):  # noqa: ANN001
        x1, y1 = bbox.x1, bbox.y1
        w, h = bbox.width, bbox.height
        named = {
            "neck_base": (x1 + 0.20 * w, y1 + 0.40 * h, 0.9),
            "back_base": (x1 + 0.30 * w, y1 + 0.30 * h, 0.9),
            "back_end": (x1 + 0.75 * w, y1 + 0.33 * h, 0.9),
        }
        return PoseKeypoints.from_mapping(
            named, frame_idx=frame_idx, mono_ts=mono_ts or 0.0, crop_bbox=bbox
        )


class _CountingMock(MockPoseEstimator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def estimate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        return super().estimate(*args, **kwargs)


class _BBoxRecordingEstimator(PoseEstimator):
    """Records every bbox it is asked to pose so crop-box wiring can be asserted."""

    def __init__(self) -> None:
        self.received: list[BBox] = []

    def estimate(self, frame_bgr, bbox, frame_idx=0, mono_ts=None, wall_ts=None, source_id=None):  # noqa: ANN001
        self.received.append(bbox)
        return build_synthetic_pose(
            bbox, frame_idx=frame_idx, mono_ts=mono_ts or 0.0, posture="stand"
        )


def _varied_window() -> tuple[list[Frame], Track]:
    # Three stationary frames whose detector boxes vary in extent (the dog is
    # under-segmented differently each frame); a union over the window recovers the
    # fuller extent. mono_ts == frame_idx seconds.
    boxes = [BBox(60, 60, 80, 80), BBox(50, 55, 90, 85), BBox(55, 50, 85, 95)]
    frames: list[Frame] = []
    detections: list[Detection] = []
    for i, box in enumerate(boxes):
        frames.append(
            Frame(
                bgr=np.zeros((200, 200, 3), dtype=np.uint8),
                frame_idx=i,
                mono_ts=float(i),
                wall_ts=_WALL,
                source_id="cam",
            )
        )
        detections.append(
            Detection(
                bbox=box,
                confidence=0.9,
                class_name="dog",
                frame_idx=i,
                mono_ts=float(i),
                wall_ts=_WALL,
            )
        )
    return frames, Track(track_id="t1", detections=detections)


def test_estimate_window_passes_raw_boxes_when_union_disabled() -> None:
    estimator = _BBoxRecordingEstimator()
    classifier = PosePottyClassifier(estimator, PoseConfig(), _StubFallback())
    frames, track = _varied_window()

    classifier._estimate_window(track, frames)

    assert estimator.received == [det.bbox for det in track.detections]


def test_estimate_window_unions_boxes_when_enabled() -> None:
    estimator = _BBoxRecordingEstimator()
    classifier = PosePottyClassifier(
        estimator, PoseConfig(box_union_window_s=10.0), _StubFallback()
    )
    frames, track = _varied_window()

    classifier._estimate_window(track, frames)

    # First frame has no earlier in-window neighbor -> raw box; the last frame
    # unions all three into the recovered extent.
    assert estimator.received[0] == BBox(60, 60, 80, 80)
    assert estimator.received[-1] == BBox(50, 50, 90, 95)



# --- classify_pose_features decision function --------------------------------


def test_decision_poop_signature():
    features = _features(
        spine_angle_deg=90.0,
        hip_offset_ratio=0.30,
        centroid_motion_ratio=0.05,
        dwell_duration_s=8.0,
    )
    guess, confidence = classify_pose_features(features)
    assert guess == ClassifierGuess.POOP
    assert 0.0 < confidence <= 0.65


def test_decision_pee_signature():
    features = _features(
        spine_angle_deg=178.0,
        hip_offset_ratio=0.0,
        centroid_motion_ratio=0.9,
        dwell_duration_s=1.0,
    )
    guess, confidence = classify_pose_features(features)
    assert guess == ClassifierGuess.PEE
    assert 0.0 < confidence <= 0.6


def test_decision_leg_lift_overrides_to_pee():
    features = _features(
        spine_angle_deg=90.0,
        hip_offset_ratio=0.30,
        centroid_motion_ratio=0.05,
        dwell_duration_s=8.0,
        hind_paw_asymmetry=0.5,
    )
    guess, _ = classify_pose_features(features)
    assert guess == ClassifierGuess.PEE


def test_decision_thresholds_are_tunable():
    features = _features(
        spine_angle_deg=160.0, hip_offset_ratio=0.20, dwell_duration_s=0.0
    )
    # Default thresholds: only the deep-squat signal fires -> PEE.
    assert classify_pose_features(features)[0] == ClassifierGuess.PEE
    # Looser thresholds make both the spine and squat signals fire -> POOP.
    loose = PoseDecisionThresholds(arched_spine_deg=170.0, poop_fraction=0.5)
    assert classify_pose_features(features, loose)[0] == ClassifierGuess.POOP


def test_decision_insufficient_evidence_returns_none():
    # Only one geometric signal available (spine) -> not enough to classify.
    assert classify_pose_features(_features(spine_angle_deg=90.0)) is None
    # Two non-core signals (motion + tail) but no core squat signal -> None.
    only_non_core = _features(centroid_motion_ratio=0.1, tail_angle_deg=150.0)
    assert classify_pose_features(only_non_core) is None


# --- PosePottyClassifier integration -----------------------------------------


def test_squat_window_classifies_poop():
    estimator = MockPoseEstimator(posture="squat", confidence=0.9)
    classifier = PosePottyClassifier(estimator, PoseConfig(), _StubFallback())
    frames, track = _make_window(5)
    result = classifier.classify(track, frames)
    assert result.guess == ClassifierGuess.POOP
    assert result.needs_label is True


def test_stand_window_classifies_pee():
    estimator = MockPoseEstimator(posture="stand", confidence=0.9)
    classifier = PosePottyClassifier(estimator, PoseConfig(), _StubFallback())
    frames, track = _make_window(5, spacing=0.1)
    result = classifier.classify(track, frames)
    assert result.guess == ClassifierGuess.PEE


def test_no_pose_falls_back_to_heuristic():
    fallback = _StubFallback()
    classifier = PosePottyClassifier(_NoneEstimator(), PoseConfig(), fallback)
    frames, track = _make_window(5)
    result = classifier.classify(track, frames)
    assert fallback.calls == 1
    assert result.confidence == 0.123


def test_low_quality_pose_falls_back_to_heuristic():
    # Only two valid frames < min_required_frames -> fallback_recommended -> heuristic.
    fallback = _StubFallback()
    classifier = PosePottyClassifier(
        MockPoseEstimator(posture="squat"), PoseConfig(), fallback
    )
    frames, track = _make_window(2)
    result = classifier.classify(track, frames)
    assert fallback.calls == 1
    assert result.confidence == 0.123


def test_insufficient_posture_evidence_falls_back_to_heuristic():
    # Good torso quality (passes coverage gates) but too few posture signals.
    fallback = _StubFallback()
    classifier = PosePottyClassifier(_SparseEstimator(), PoseConfig(), fallback)
    frames, track = _make_window(5)
    result = classifier.classify(track, frames)
    assert fallback.calls == 1
    assert result.confidence == 0.123


def test_estimate_window_caps_frame_count():
    estimator = _CountingMock(posture="squat", confidence=0.9)
    classifier = PosePottyClassifier(
        estimator, PoseConfig(), _StubFallback(), max_pose_frames=4
    )
    frames, track = _make_window(20)
    classifier.classify(track, frames)
    assert estimator.calls == 4
