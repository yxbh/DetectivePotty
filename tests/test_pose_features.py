from __future__ import annotations

from detectivepotty.geometry import BBox
from detectivepotty.pose.base import build_synthetic_pose
from detectivepotty.pose.features import extract_pose_features
from detectivepotty.pose.keypoints import NECK, SPINE_MID, TAIL_END, WITHERS


def _window(posture: str, n: int = 5, facing: str = "left", bbox: BBox | None = None,
            **kw) -> list:
    box = bbox if bbox is not None else BBox(100, 200, 260, 320)
    return [
        build_synthetic_pose(box, frame_idx=i, mono_ts=i * 0.2, posture=posture, facing=facing, **kw)
        for i in range(n)
    ]


def test_no_pose_returns_none() -> None:
    assert extract_pose_features([]) is None


def test_squat_separates_from_stand() -> None:
    stand = extract_pose_features(_window("stand"))
    squat = extract_pose_features(_window("squat"))
    assert stand is not None and squat is not None
    # Pelvis drops below the front-spine line and the back hunches when squatting.
    assert squat.hip_offset_ratio > stand.hip_offset_ratio
    assert squat.spine_angle_deg < stand.spine_angle_deg
    assert stand.fallback_recommended is False
    assert squat.fallback_recommended is False


def test_features_are_view_and_scale_invariant() -> None:
    left = extract_pose_features(_window("stand", facing="left"))
    right = extract_pose_features(_window("stand", facing="right"))
    big = extract_pose_features(_window("stand", bbox=BBox(0, 0, 800, 600)))
    moved = extract_pose_features(_window("stand", bbox=BBox(900, 700, 1060, 820)))
    for other in (right, big, moved):
        assert abs(other.hip_offset_ratio - left.hip_offset_ratio) < 1e-6
        assert abs(other.spine_angle_deg - left.spine_angle_deg) < 1e-6


def test_squat_depth_change_tracks_transition() -> None:
    box = BBox(0, 0, 160, 120)
    poses = [
        build_synthetic_pose(box, frame_idx=i, mono_ts=i * 0.2,
                             posture="stand" if i < 3 else "squat")
        for i in range(6)
    ]
    feats = extract_pose_features(poses)
    assert feats is not None
    assert feats.squat_depth_change is not None
    assert feats.squat_depth_change > 0.2
    assert feats.dwell_duration_s > 0.0


def test_stationary_vs_moving_centroid_motion() -> None:
    still = extract_pose_features(_window("stand"))
    moving_poses = [
        build_synthetic_pose(BBox(100 + i * 30, 200, 260 + i * 30, 320),
                             frame_idx=i, mono_ts=i * 0.2, posture="stand")
        for i in range(5)
    ]
    moving = extract_pose_features(moving_poses)
    assert still.centroid_motion_ratio is not None
    assert still.centroid_motion_ratio < 1e-6
    assert moving.centroid_motion_ratio > 0.5


def test_missing_tail_yields_none_feature_not_crash() -> None:
    feats = extract_pose_features(_window("stand", missing_roles=(TAIL_END,)))
    assert feats is not None
    assert feats.tail_angle_deg is None
    assert "tail_angle_deg" in feats.missing_reasons
    # Other features remain available.
    assert feats.hip_offset_ratio is not None
    assert feats.support["tail_angle_deg"] == 0


def test_low_coverage_recommends_fallback() -> None:
    # Three good frames, three with the torso gutted (only one torso keypoint left).
    good = _window("stand", n=3)
    bad = _window("stand", n=3, missing_roles=(NECK, WITHERS, SPINE_MID))
    feats = extract_pose_features(good + bad, min_pose_coverage=0.8)
    assert feats is not None
    assert feats.n_frames_valid == 3
    assert feats.coverage == 0.5
    assert feats.fallback_recommended is True


def test_all_frames_unusable_returns_none() -> None:
    bad = _window("stand", n=4, missing_roles=(NECK, WITHERS, SPINE_MID))
    assert extract_pose_features(bad, min_torso_keypoints=3) is None


def test_hind_paw_asymmetry_small_for_symmetric_stance() -> None:
    box = BBox(0, 0, 200, 150)
    feats = extract_pose_features(_window("stand", bbox=box))
    assert feats.hind_paw_asymmetry is not None
    # A normal symmetric stance has near-zero lateral hind-paw asymmetry.
    assert feats.hind_paw_asymmetry < 0.1
