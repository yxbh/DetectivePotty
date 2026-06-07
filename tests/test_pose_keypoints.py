from __future__ import annotations

from datetime import datetime, timezone

from detectivepotty.geometry import BBox
from detectivepotty.pose.keypoints import (
    HIPS,
    KEYPOINT_ALIASES,
    PoseKeypoints,
    QUADRUPED_KEYPOINTS,
    QUADRUPED_SCHEMA,
    TORSO_KEYPOINTS,
    WITHERS,
    resolve_role,
)


def test_schema_has_39_unique_keypoints() -> None:
    assert len(QUADRUPED_KEYPOINTS) == 39
    assert len(set(QUADRUPED_KEYPOINTS)) == 39


def test_aliases_resolve_to_real_keypoint_names() -> None:
    for role, names in KEYPOINT_ALIASES.items():
        assert names, f"role {role} has no candidates"
        for name in names:
            assert name in QUADRUPED_KEYPOINTS
    for name in TORSO_KEYPOINTS:
        assert name in QUADRUPED_KEYPOINTS


def test_resolve_role_unknown_raises() -> None:
    try:
        resolve_role("not_a_role")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown role")


def test_get_role_honors_confidence_and_fallback() -> None:
    # WITHERS resolves to back_base then neck_end; only neck_end is present here.
    pose = PoseKeypoints.from_mapping(
        {"neck_end": (10.0, 20.0, 0.8), "back_end": (50.0, 25.0, 0.3)},
        frame_idx=0,
        mono_ts=0.0,
    )
    withers = pose.get_role(WITHERS, min_conf=0.5)
    assert withers is not None
    assert withers.xy == (10.0, 20.0)

    # HIPS resolves to back_end (conf 0.3) — gated out at min_conf 0.5.
    assert pose.get_role(HIPS, min_conf=0.5) is None
    assert pose.get_role(HIPS, min_conf=0.2) is not None


def test_torso_keypoint_count_and_provenance_roundtrip() -> None:
    pose = PoseKeypoints.from_mapping(
        {"neck_base": (0.0, 0.0, 0.9), "back_middle": (5.0, 1.0, 0.9), "tail_base": (10.0, 2.0, 0.4)},
        frame_idx=7,
        mono_ts=1.5,
        wall_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        backend="mock",
        model_name="synthetic",
        crop_bbox=BBox(0, 0, 10, 10),
        crop_margin_frac=0.4,
        device="cpu",
        latency_ms=12.0,
    )
    assert pose.torso_keypoint_count(min_conf=0.5) == 2
    data = pose.to_dict()
    assert data["frame_idx"] == 7
    assert data["keypoint_schema"] == QUADRUPED_SCHEMA
    assert data["crop_bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert data["backend"] == "mock"
    assert set(data["points"]) == {"neck_base", "back_middle", "tail_base"}
