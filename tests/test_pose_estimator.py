from __future__ import annotations

import numpy as np

from detectivepotty.geometry import BBox
from detectivepotty.pose.base import MockPoseEstimator, build_synthetic_pose
from detectivepotty.pose.keypoints import HIPS, NECK, SPINE_MID, TAIL_END, WITHERS


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_mock_estimator_returns_pose_inside_bbox() -> None:
    bbox = BBox(100, 150, 260, 290)
    pose = MockPoseEstimator().estimate(FRAME, bbox, frame_idx=3, mono_ts=2.0)
    assert pose is not None
    assert pose.frame_idx == 3
    assert pose.image_size == (640, 480)
    for role in (NECK, WITHERS, SPINE_MID, HIPS):
        kp = pose.get_role(role, min_conf=0.5)
        assert kp is not None
        assert bbox.x1 <= kp.x <= bbox.x2
        assert bbox.y1 <= kp.y <= bbox.y2


def test_mock_estimator_zero_area_bbox_returns_none() -> None:
    assert MockPoseEstimator().estimate(FRAME, BBox(10, 10, 10, 40)) is None


def test_mock_estimator_can_drop_roles() -> None:
    pose = MockPoseEstimator(missing_roles=(TAIL_END,)).estimate(FRAME, BBox(0, 0, 100, 80))
    assert pose is not None
    assert pose.get_role(TAIL_END, min_conf=0.0) is None
    assert pose.get_role(HIPS, min_conf=0.5) is not None


def test_per_frame_posture_override() -> None:
    estimator = MockPoseEstimator(posture="stand", per_frame_posture={5: "squat"})
    bbox = BBox(0, 0, 160, 120)
    stand = estimator.estimate(FRAME, bbox, frame_idx=0)
    squat = estimator.estimate(FRAME, bbox, frame_idx=5)
    assert stand is not None and squat is not None
    # The squatting pelvis sits lower in the box than the standing one.
    assert squat.get_role(HIPS, 0.5).y > stand.get_role(HIPS, 0.5).y


def test_build_synthetic_pose_facing_flips_head() -> None:
    bbox = BBox(0, 0, 100, 100)
    left = build_synthetic_pose(bbox, facing="left")
    right = build_synthetic_pose(bbox, facing="right")
    # Nose is near the left edge facing left, near the right edge facing right.
    assert left.get_role("head", 0.0).x < 50
    assert right.get_role("head", 0.0).x > 50


def test_mock_estimator_has_no_telemetry() -> None:
    assert MockPoseEstimator().telemetry_snapshot() is None
