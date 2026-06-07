"""Quadruped keypoint schema, verified anatomical aliases, and the pose record.

The backend is DeepLabCut SuperAnimal-Quadruped (39 keypoints). The anatomical
mapping below was VERIFIED against real model output on our own day clips (see
``files/pose_spike/verify_anatomy.py``): on a side-on dog the front paws cluster
at the head/nose end and the hind paws at the tail end, confirming which torso
keypoints are front (withers/shoulder) vs rear (hips/pelvis). Feature code keys
off the semantic ROLES here, never the raw DeepLabCut names, so a backend swap or
a schema quirk only touches this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from detectivepotty.geometry import BBox

# The 39 SuperAnimal-Quadruped body parts, in model output order. ("thai" is the
# upstream spelling of "thigh"; the "antler" points are irrelevant for dogs.)
QUADRUPED_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
)

QUADRUPED_SCHEMA = "superanimal_quadruped"


# Semantic roles used by feature code. Values are candidate DeepLabCut names in
# priority order; ``PoseKeypoints.get_role`` returns the first one that is present
# and confident enough. Some upstream points coincide (e.g. ``back_base`` and
# ``neck_end``; ``back_end`` and ``tail_base``), so each role lists fallbacks.
HEAD = "head"
NECK = "neck"
WITHERS = "withers"  # front torso top (shoulder)
SPINE_MID = "spine_mid"
HIPS = "hips"  # rear torso top (pelvis)
TAIL_BASE = "tail_base"
TAIL_END = "tail_end"
FRONT_LEFT_PAW = "front_left_paw"
FRONT_RIGHT_PAW = "front_right_paw"
HIND_LEFT_PAW = "hind_left_paw"
HIND_RIGHT_PAW = "hind_right_paw"
BELLY = "belly"

KEYPOINT_ALIASES: dict[str, tuple[str, ...]] = {
    HEAD: ("nose",),
    NECK: ("neck_base",),
    WITHERS: ("back_base", "neck_end"),
    SPINE_MID: ("back_middle",),
    HIPS: ("back_end", "tail_base"),
    TAIL_BASE: ("tail_base", "back_end"),
    TAIL_END: ("tail_end",),
    FRONT_LEFT_PAW: ("front_left_paw",),
    FRONT_RIGHT_PAW: ("front_right_paw",),
    HIND_LEFT_PAW: ("back_left_paw",),
    HIND_RIGHT_PAW: ("back_right_paw",),
    BELLY: ("belly_bottom",),
}

# Torso keypoints used for the dog-local coordinate frame and the body-scale
# estimate. Kept separate from head/leg points which are noisier or gait-driven.
TORSO_KEYPOINTS: tuple[str, ...] = (
    "neck_base",
    "back_base",
    "back_middle",
    "back_end",
    "tail_base",
)


def resolve_role(role: str) -> tuple[str, ...]:
    """Return the candidate DeepLabCut names for a semantic role."""

    try:
        return KEYPOINT_ALIASES[role]
    except KeyError as exc:  # pragma: no cover - programmer error guard.
        raise KeyError(f"Unknown keypoint role: {role!r}") from exc


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A single keypoint in original-source pixel space."""

    x: float
    y: float
    confidence: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "confidence": self.confidence}


@dataclass(slots=True)
class PoseKeypoints:
    """Per-frame keypoints for one dog, with provenance for debug/persistence.

    Coordinates are ORIGINAL-source pixels. ``points`` is keyed by raw DeepLabCut
    name; access via :meth:`get_role` to stay decoupled from the raw schema. Pose
    is per-frame: associate it to a track via ``frame_idx`` rather than embedding
    it inside :class:`~detectivepotty.events.Track`.
    """

    points: dict[str, Keypoint]
    frame_idx: int
    mono_ts: float
    wall_ts: datetime | None = None
    source_id: str | None = None
    backend: str = ""
    model_name: str = ""
    model_version: str | None = None
    device: str | None = None
    image_size: tuple[int, int] | None = None
    crop_bbox: BBox | None = None
    crop_margin_frac: float | None = None
    keypoint_schema: str = QUADRUPED_SCHEMA
    latency_ms: float | None = None
    failure_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, min_conf: float = 0.0) -> Keypoint | None:
        """Return the raw-named keypoint if present and confident enough."""

        point = self.points.get(name)
        if point is None or point.confidence < min_conf:
            return None
        return point

    def get_role(self, role: str, min_conf: float = 0.0) -> Keypoint | None:
        """Return the best keypoint for a semantic role, honoring fallbacks."""

        for name in resolve_role(role):
            point = self.get(name, min_conf)
            if point is not None:
                return point
        return None

    def present_roles(self, roles: tuple[str, ...], min_conf: float = 0.0) -> int:
        """Count how many of ``roles`` resolve to a confident keypoint."""

        return sum(1 for role in roles if self.get_role(role, min_conf) is not None)

    def torso_keypoint_count(self, min_conf: float = 0.0) -> int:
        """Number of confident torso keypoints (pose-quality signal)."""

        return sum(
            1
            for name in TORSO_KEYPOINTS
            if self.get(name, min_conf) is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": {name: kp.to_dict() for name, kp in self.points.items()},
            "frame_idx": self.frame_idx,
            "mono_ts": self.mono_ts,
            "wall_ts": self.wall_ts.isoformat() if self.wall_ts else None,
            "source_id": self.source_id,
            "backend": self.backend,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "image_size": list(self.image_size) if self.image_size else None,
            "crop_bbox": (
                [self.crop_bbox.x1, self.crop_bbox.y1, self.crop_bbox.x2, self.crop_bbox.y2]
                if self.crop_bbox is not None
                else None
            ),
            "crop_margin_frac": self.crop_margin_frac,
            "keypoint_schema": self.keypoint_schema,
            "latency_ms": self.latency_ms,
            "failure_reason": self.failure_reason,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_mapping(
        cls,
        named_points: Mapping[str, tuple[float, float, float]],
        frame_idx: int,
        mono_ts: float,
        **provenance: Any,
    ) -> "PoseKeypoints":
        """Build from a ``name -> (x, y, confidence)`` mapping (test/backends)."""

        points = {
            name: Keypoint(float(x), float(y), float(conf))
            for name, (x, y, conf) in named_points.items()
        }
        return cls(points=points, frame_idx=frame_idx, mono_ts=mono_ts, **provenance)
