"""View-robust posture features from a window of keypoint poses.

Design constraints (validated by a rubber-duck critique + anatomy verification):

* NO global "height above ground" — the camera is fixed but angled/wide-angle, so
  absolute pixel-y confounds depth/orientation with posture. Features use a
  DOG-LOCAL frame instead: segment angles and torso-normalized signed distances,
  plus temporal change across the track window.
* Robust body scale = the torso span (``neck_base``..``tail_base`` /
  ``back_base``..``back_end``), with the bbox diagonal only as a low-quality
  fallback.
* Output is POPULATED-WITH-VALIDITY, never a bare ``None`` for a partial pose: each
  feature may be ``None`` while coverage/quality and ``fallback_recommended`` tell
  the caller whether to trust pose or fall back to the bbox heuristics. A bare
  ``None`` return is reserved for "no usable pose at all".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median
from typing import Any, Sequence

from detectivepotty.pose.keypoints import (
    HIND_LEFT_PAW,
    HIND_RIGHT_PAW,
    HIPS,
    NECK,
    PoseKeypoints,
    SPINE_MID,
    TAIL_BASE,
    TAIL_END,
    TORSO_KEYPOINTS,
    WITHERS,
    BELLY,
)

Point = tuple[float, float]


@dataclass(slots=True)
class PoseFeatures:
    """Window-level posture features with per-feature support and quality flags.

    Higher ``hip_offset_ratio`` means the pelvis sits further to the belly side of
    the front-spine line (a squat); ``spine_angle_deg`` near 180 is a straight back
    and lower values are a hunched/arched back; ``squat_depth_change`` is how much
    the dog lowered across the window. ``fallback_recommended`` is ``True`` when the
    pose is too sparse/low-quality to trust over the bbox heuristics.
    """

    n_frames_total: int
    n_frames_valid: int
    coverage: float
    dwell_duration_s: float
    body_scale_px: float | None
    body_scale_quality: str
    spine_angle_deg: float | None = None
    hip_offset_ratio: float | None = None
    hip_offset_min: float | None = None
    hip_offset_max: float | None = None
    squat_depth_change: float | None = None
    hind_paw_drop_ratio: float | None = None
    tail_angle_deg: float | None = None
    hind_paw_asymmetry: float | None = None
    centroid_motion_ratio: float | None = None
    fallback_recommended: bool = True
    support: dict[str, int] = field(default_factory=dict)
    missing_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frames_total": self.n_frames_total,
            "n_frames_valid": self.n_frames_valid,
            "coverage": self.coverage,
            "dwell_duration_s": self.dwell_duration_s,
            "body_scale_px": self.body_scale_px,
            "body_scale_quality": self.body_scale_quality,
            "spine_angle_deg": self.spine_angle_deg,
            "hip_offset_ratio": self.hip_offset_ratio,
            "hip_offset_min": self.hip_offset_min,
            "hip_offset_max": self.hip_offset_max,
            "squat_depth_change": self.squat_depth_change,
            "hind_paw_drop_ratio": self.hind_paw_drop_ratio,
            "tail_angle_deg": self.tail_angle_deg,
            "hind_paw_asymmetry": self.hind_paw_asymmetry,
            "centroid_motion_ratio": self.centroid_motion_ratio,
            "fallback_recommended": self.fallback_recommended,
            "support": dict(self.support),
            "missing_reasons": list(self.missing_reasons),
        }


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_at(a: Point, vertex: Point, c: Point) -> float | None:
    """Interior angle (degrees) at ``vertex`` formed by ``a`` and ``c``."""

    v1 = _sub(a, vertex)
    v2 = _sub(c, vertex)
    n1, n2 = _norm(v1), _norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _angle_between(v1: Point, v2: Point) -> float | None:
    n1, n2 = _norm(v1), _norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _signed_perp_distance(
    point: Point,
    line_a: Point,
    line_b: Point,
    belly_side: Point | None,
) -> float | None:
    """Perpendicular distance of ``point`` from line ``a→b``.

    Positive on the belly side of the line (if ``belly_side`` is known), else
    positive in the image-down (+y) direction so a dropped pelvis reads positive.
    """

    axis = _sub(line_b, line_a)
    axis_len = _norm(axis)
    if axis_len == 0.0:
        return None
    # Left-normal of the axis; cross product gives signed area / length = distance.
    rel = _sub(point, line_a)
    cross = axis[0] * rel[1] - axis[1] * rel[0]
    signed = cross / axis_len
    # Orient so "belly side" (or image-down when belly unknown) is positive.
    if belly_side is not None:
        rel_belly = _sub(belly_side, line_a)
        belly_cross = axis[0] * rel_belly[1] - axis[1] * rel_belly[0]
        if belly_cross < 0:
            signed = -signed
    else:
        # Down direction (+y) should map to positive: check the normal's y sign.
        if -axis[0] < 0:  # left-normal y-component is -axis_x
            signed = -signed
    return signed


def _role_point(pose: PoseKeypoints, role: str, min_conf: float) -> Point | None:
    kp = pose.get_role(role, min_conf)
    return kp.xy if kp is not None else None


def _body_scale(pose: PoseKeypoints, min_conf: float) -> tuple[float | None, str]:
    neck = _role_point(pose, NECK, min_conf)
    hips = _role_point(pose, HIPS, min_conf)
    withers = _role_point(pose, WITHERS, min_conf)
    candidates: list[float] = []
    if neck is not None and hips is not None:
        candidates.append(_dist(neck, hips))
    if withers is not None and hips is not None:
        candidates.append(_dist(withers, hips))
    if candidates:
        scale = median(candidates)
        if scale > 0.0:
            return scale, "torso"
    if pose.crop_bbox is not None:
        diag = math.hypot(pose.crop_bbox.width, pose.crop_bbox.height)
        if diag > 0.0:
            return diag, "bbox_fallback"
    return None, "none"


def _torso_centroid(pose: PoseKeypoints, min_conf: float) -> Point | None:
    xs: list[float] = []
    ys: list[float] = []
    for name in TORSO_KEYPOINTS:
        kp = pose.get(name, min_conf)
        if kp is not None:
            xs.append(kp.x)
            ys.append(kp.y)
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass(slots=True)
class _FramePosture:
    spine_angle_deg: float | None
    hip_offset_ratio: float | None
    hind_paw_drop_ratio: float | None
    tail_angle_deg: float | None
    hind_paw_asymmetry: float | None
    centroid: Point | None


def _frame_posture(pose: PoseKeypoints, min_conf: float, scale: float) -> _FramePosture:
    neck = _role_point(pose, NECK, min_conf)
    withers = _role_point(pose, WITHERS, min_conf)
    spine_mid = _role_point(pose, SPINE_MID, min_conf)
    hips = _role_point(pose, HIPS, min_conf)
    tail_base = _role_point(pose, TAIL_BASE, min_conf)
    tail_end = _role_point(pose, TAIL_END, min_conf)
    belly = _role_point(pose, BELLY, min_conf)
    hind_l = _role_point(pose, HIND_LEFT_PAW, min_conf)
    hind_r = _role_point(pose, HIND_RIGHT_PAW, min_conf)

    spine_angle = (
        _angle_at(withers, spine_mid, hips)
        if withers and spine_mid and hips
        else None
    )

    hip_offset = None
    if neck and spine_mid and hips:
        signed = _signed_perp_distance(hips, neck, spine_mid, belly)
        if signed is not None:
            hip_offset = signed / scale

    hind_drop = None
    hind_paws = [p for p in (hind_l, hind_r) if p is not None]
    if hips and neck and spine_mid and hind_paws:
        centroid = (
            sum(p[0] for p in hind_paws) / len(hind_paws),
            sum(p[1] for p in hind_paws) / len(hind_paws),
        )
        signed = _signed_perp_distance(centroid, neck, spine_mid, belly)
        hip_signed = _signed_perp_distance(hips, neck, spine_mid, belly)
        if signed is not None and hip_signed is not None:
            hind_drop = (signed - hip_signed) / scale

    tail_angle = None
    if spine_mid and hips and tail_base and tail_end:
        rear = _sub(hips, spine_mid)
        tail = _sub(tail_end, tail_base)
        tail_angle = _angle_between(rear, tail)

    asymmetry = None
    if hind_l and hind_r and neck and spine_mid:
        left = _signed_perp_distance(hind_l, neck, spine_mid, belly)
        right = _signed_perp_distance(hind_r, neck, spine_mid, belly)
        if left is not None and right is not None:
            asymmetry = abs(left - right) / scale

    return _FramePosture(
        spine_angle_deg=spine_angle,
        hip_offset_ratio=hip_offset,
        hind_paw_drop_ratio=hind_drop,
        tail_angle_deg=tail_angle,
        hind_paw_asymmetry=asymmetry,
        centroid=_torso_centroid(pose, min_conf),
    )


def _median_or_none(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def extract_pose_features(
    poses: Sequence[PoseKeypoints],
    *,
    min_keypoint_conf: float = 0.5,
    min_required_frames: int = 3,
    min_pose_coverage: float = 0.5,
    min_torso_keypoints: int = 3,
) -> PoseFeatures | None:
    """Reduce a window of poses to view-robust posture features.

    Returns ``None`` only when there is no usable pose at all; otherwise returns a
    populated object whose ``fallback_recommended`` flag and per-feature ``None``
    values let the caller decide whether to trust pose.
    """

    n_total = len(poses)
    if n_total == 0:
        return None

    valid: list[tuple[PoseKeypoints, float, _FramePosture]] = []
    scales: list[float] = []
    scale_qualities: list[str] = []
    for pose in poses:
        if pose.torso_keypoint_count(min_keypoint_conf) < min_torso_keypoints:
            continue
        scale, quality = _body_scale(pose, min_keypoint_conf)
        if scale is None or scale <= 0.0:
            continue
        posture = _frame_posture(pose, min_keypoint_conf, scale)
        valid.append((pose, scale, posture))
        scales.append(scale)
        scale_qualities.append(quality)

    n_valid = len(valid)
    if n_valid == 0:
        return None

    coverage = n_valid / n_total
    body_scale = median(scales)
    quality = "torso" if any(q == "torso" for q in scale_qualities) else "bbox_fallback"

    valid_mono = [pose.mono_ts for pose, _, _ in valid]
    dwell = max(0.0, max(valid_mono) - min(valid_mono))

    spine = [p.spine_angle_deg for _, _, p in valid if p.spine_angle_deg is not None]
    hip = [p.hip_offset_ratio for _, _, p in valid if p.hip_offset_ratio is not None]
    hind = [p.hind_paw_drop_ratio for _, _, p in valid if p.hind_paw_drop_ratio is not None]
    tail = [p.tail_angle_deg for _, _, p in valid if p.tail_angle_deg is not None]
    asym = [p.hind_paw_asymmetry for _, _, p in valid if p.hind_paw_asymmetry is not None]

    squat_change = (max(hip) - min(hip)) if len(hip) >= 2 else None

    centroids = [p.centroid for _, _, p in valid if p.centroid is not None]
    centroid_motion = None
    if len(centroids) >= 2 and body_scale > 0.0:
        origin = centroids[0]
        max_disp = max(_dist(c, origin) for c in centroids)
        centroid_motion = max_disp / body_scale

    support = {
        "spine_angle_deg": len(spine),
        "hip_offset_ratio": len(hip),
        "hind_paw_drop_ratio": len(hind),
        "tail_angle_deg": len(tail),
        "hind_paw_asymmetry": len(asym),
        "centroid_motion_ratio": len(centroids),
    }
    missing = tuple(name for name, count in support.items() if count == 0)

    fallback = (
        n_valid < min_required_frames
        or coverage < min_pose_coverage
        or quality != "torso"
    )

    return PoseFeatures(
        n_frames_total=n_total,
        n_frames_valid=n_valid,
        coverage=coverage,
        dwell_duration_s=dwell,
        body_scale_px=body_scale,
        body_scale_quality=quality,
        spine_angle_deg=_median_or_none(spine),
        hip_offset_ratio=_median_or_none(hip),
        hip_offset_min=min(hip) if hip else None,
        hip_offset_max=max(hip) if hip else None,
        squat_depth_change=squat_change,
        hind_paw_drop_ratio=_median_or_none(hind),
        tail_angle_deg=_median_or_none(tail),
        hind_paw_asymmetry=_median_or_none(asym),
        centroid_motion_ratio=centroid_motion,
        fallback_recommended=fallback,
        support=support,
        missing_reasons=missing,
    )
