"""Draw keypoint skeletons on saved dog crops for human review.

Keypoints are stored in ORIGINAL-frame pixel coordinates, while the crops are a
margin-expanded sub-image of the frame. We recompute each crop's origin exactly as
:func:`detectivepotty.geometry.crop_from_frame` does (same expand+clip), then map
keypoints into crop-local pixels before drawing. Overlay generation is best-effort
and must never break event recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2

from detectivepotty.events import CropRecord, FrameRecord
from detectivepotty.geometry import BBox
from detectivepotty.pose.keypoints import (
    FRONT_LEFT_PAW,
    FRONT_RIGHT_PAW,
    HEAD,
    HIND_LEFT_PAW,
    HIND_RIGHT_PAW,
    HIPS,
    NECK,
    PoseKeypoints,
    SPINE_MID,
    TAIL_END,
    WITHERS,
)

# Backend-agnostic skeleton expressed in semantic roles (resolved via get_role).
_SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    (HEAD, NECK),
    (NECK, WITHERS),
    (WITHERS, SPINE_MID),
    (SPINE_MID, HIPS),
    (HIPS, TAIL_END),
    (WITHERS, FRONT_LEFT_PAW),
    (WITHERS, FRONT_RIGHT_PAW),
    (HIPS, HIND_LEFT_PAW),
    (HIPS, HIND_RIGHT_PAW),
)
_POINT_COLOR = (0, 215, 255)  # amber (BGR)
_EDGE_COLOR = (0, 255, 0)  # green (BGR)
_OVERLAY_SUBDIR = "crops_overlay"


def crop_origin(bbox: BBox, margin_frac: float, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Top-left origin (in original pixels) of the margin-expanded crop box."""

    crop_box = bbox.expand(margin_frac, frame_w, frame_h)
    x1, y1, _, _ = crop_box.to_int_tuple()
    return (min(max(x1, 0), frame_w), min(max(y1, 0), frame_h))


def draw_pose_on_crop(
    crop_bgr,
    keypoints: PoseKeypoints,
    origin_xy: tuple[int, int],
    *,
    min_conf: float = 0.0,
):
    """Draw the skeleton + confident keypoints onto ``crop_bgr`` in place."""

    ox, oy = origin_xy
    height, width = crop_bgr.shape[:2]

    def role_point(role: str) -> tuple[int, int] | None:
        keypoint = keypoints.get_role(role, min_conf)
        if keypoint is None:
            return None
        return (int(round(keypoint.x - ox)), int(round(keypoint.y - oy)))

    for role_a, role_b in _SKELETON_EDGES:
        point_a = role_point(role_a)
        point_b = role_point(role_b)
        if point_a is not None and point_b is not None:
            cv2.line(crop_bgr, point_a, point_b, _EDGE_COLOR, 2, cv2.LINE_AA)

    for keypoint in keypoints.points.values():
        if keypoint.confidence < min_conf:
            continue
        x = int(round(keypoint.x - ox))
        y = int(round(keypoint.y - oy))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(crop_bgr, (x, y), 3, _POINT_COLOR, -1, cv2.LINE_AA)

    return crop_bgr


def write_pose_overlays(
    target_event_dir: str | Path,
    crop_records: Sequence[CropRecord],
    frame_records: Sequence[FrameRecord],
    poses: Sequence[PoseKeypoints],
    *,
    min_conf: float = 0.0,
    jpeg_quality: int = 92,
) -> list[str]:
    """Write skeleton overlays for crops that have a matching pose.

    Returns the list of written overlay paths (relative to the event dir). Crops
    without a pose are skipped, so the overlay set is a subset of the crop set.
    """

    poses_by_frame = {pose.frame_idx: pose for pose in poses}
    dims_by_frame = {
        record.frame_idx: (record.original_width, record.original_height)
        for record in frame_records
    }
    event_path = Path(target_event_dir)
    overlay_dir = event_path / _OVERLAY_SUBDIR
    written: list[str] = []

    for record in crop_records:
        if record.path is None:
            continue
        pose = poses_by_frame.get(record.frame_idx)
        dims = dims_by_frame.get(record.frame_idx)
        if pose is None or dims is None:
            continue
        crop_image = cv2.imread(str(event_path / record.path))
        if crop_image is None:
            continue
        origin = crop_origin(record.bbox, record.margin_frac, dims[0], dims[1])
        draw_pose_on_crop(crop_image, pose, origin, min_conf=min_conf)
        out_rel = Path(_OVERLAY_SUBDIR) / Path(record.path).name
        overlay_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(
            str(event_path / out_rel),
            crop_image,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        ):
            written.append(out_rel.as_posix())

    return written
