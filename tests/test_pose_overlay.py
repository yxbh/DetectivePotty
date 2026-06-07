from __future__ import annotations

import cv2
import numpy as np

from detectivepotty.events import CropRecord, FrameRecord
from detectivepotty.geometry import BBox
from detectivepotty.pose.keypoints import Keypoint, PoseKeypoints
from detectivepotty.recording.pose_overlay import (
    _EDGE_COLOR,
    _POINT_COLOR,
    crop_origin,
    draw_pose_on_crop,
    write_pose_overlays,
)


def _pose(points: dict[str, tuple[float, float, float]], frame_idx: int = 0) -> PoseKeypoints:
    return PoseKeypoints(
        points={name: Keypoint(x, y, conf) for name, (x, y, conf) in points.items()},
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
    )


def test_crop_origin_clips_to_frame_bounds() -> None:
    # Expanding a near-edge bbox would push the origin negative; it must clip to 0.
    assert crop_origin(BBox(2, 2, 10, 10), 1.0, 200, 100) == (0, 0)
    # Interior bbox: origin is the integer expanded top-left.
    expanded = BBox(50, 40, 90, 80).expand(0.0, 200, 100).to_int_tuple()
    assert crop_origin(BBox(50, 40, 90, 80), 0.0, 200, 100) == (expanded[0], expanded[1])


def test_draw_pose_maps_keypoint_into_crop_space() -> None:
    crop = np.zeros((40, 50, 3), dtype=np.uint8)
    # Keypoint at original (120, 75); crop origin (100, 60) -> crop pixel (20, 15).
    pose = _pose({"nose": (120.0, 75.0, 0.9)})
    draw_pose_on_crop(crop, pose, (100, 60), min_conf=0.5)
    assert tuple(int(c) for c in crop[15, 20]) == _POINT_COLOR


def test_draw_pose_skips_low_confidence_keypoints() -> None:
    crop = np.zeros((40, 50, 3), dtype=np.uint8)
    pose = _pose({"nose": (120.0, 75.0, 0.2)})
    draw_pose_on_crop(crop, pose, (100, 60), min_conf=0.5)
    assert int(crop.sum()) == 0


def test_draw_pose_draws_skeleton_edge_between_roles() -> None:
    crop = np.zeros((60, 60, 3), dtype=np.uint8)
    # head=nose, neck=neck_base: a vertical green edge at x=10 between y=10..40.
    pose = _pose({"nose": (10.0, 10.0, 0.9), "neck_base": (10.0, 40.0, 0.9)})
    draw_pose_on_crop(crop, pose, (0, 0), min_conf=0.5)
    assert tuple(int(c) for c in crop[25, 10]) == _EDGE_COLOR


def test_write_pose_overlays_only_for_crops_with_pose(tmp_path) -> None:
    event = tmp_path / "event"
    crops = event / "crops"
    crops.mkdir(parents=True)
    cv2.imwrite(str(crops / "000.jpg"), np.zeros((40, 50, 3), dtype=np.uint8))
    cv2.imwrite(str(crops / "001.jpg"), np.zeros((40, 50, 3), dtype=np.uint8))

    crop_records = [
        CropRecord(frame_idx=0, bbox=BBox(100, 60, 140, 90), margin_frac=0.0, path="crops/000.jpg"),
        CropRecord(frame_idx=1, bbox=BBox(100, 60, 140, 90), margin_frac=0.0, path="crops/001.jpg"),
    ]
    frame_records = [
        FrameRecord(0, "src", None, 400, 300),
        FrameRecord(1, "src", None, 400, 300),
    ]
    # Only frame 0 has a pose -> only one overlay is produced.
    poses = [_pose({"nose": (120.0, 75.0, 0.9)}, frame_idx=0)]

    written = write_pose_overlays(event, crop_records, frame_records, poses, min_conf=0.5)

    assert written == ["crops_overlay/000.jpg"]
    assert (event / "crops_overlay" / "000.jpg").exists()
    assert not (event / "crops_overlay" / "001.jpg").exists()
    overlay = cv2.imread(str(event / "crops_overlay" / "000.jpg"))
    assert overlay is not None
    assert int(overlay.sum()) > 0
