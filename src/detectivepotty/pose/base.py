"""PoseEstimator interface plus a deterministic mock for tests.

``PoseEstimator`` mirrors :class:`~detectivepotty.detect.yolo.DogDetector`: it takes
an original-resolution frame plus a dog bbox and returns keypoints in
original-resolution pixels (top-down "detect small, crop big" — pose runs on the
dog crop, not the whole frame). Concrete model backends live in their own modules
and are imported lazily so the core package and the tests never require the heavy
pose dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from detectivepotty.geometry import BBox
from detectivepotty.pose.keypoints import (
    BELLY,
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
    resolve_role,
)
from detectivepotty.pose.telemetry import PoseTelemetrySnapshot


@dataclass(frozen=True, eq=False)
class PoseRequest:
    """One pose estimation request for batched inference.

    ``eq=False`` because the frame is an ``np.ndarray`` (generated ``__eq__`` would
    raise on the ambiguous truth value); identity equality is all callers need.
    """

    frame_bgr_original: np.ndarray
    bbox: BBox
    frame_idx: int = 0
    mono_ts: float | None = None
    wall_ts: datetime | None = None
    source_id: str | None = None


class PoseEstimator(ABC):
    """Estimate dog keypoints for one detection in original-resolution pixels."""

    @abstractmethod
    def estimate(
        self,
        frame_bgr_original: np.ndarray,
        bbox: BBox,
        frame_idx: int = 0,
        mono_ts: float | None = None,
        wall_ts: datetime | None = None,
        source_id: str | None = None,
    ) -> PoseKeypoints | None:
        """Return keypoints for the dog in ``bbox`` or ``None`` if none produced."""

        raise NotImplementedError

    def estimate_batch(
        self, requests: Sequence[PoseRequest]
    ) -> list[PoseKeypoints | None]:
        """Estimate poses for a batch of requests, aligned 1:1 with ``requests``.

        Default impl loops :meth:`estimate` so the mock and any backend without a
        real batched path work unchanged. Backends with GPU batching (e.g.
        SuperAnimal/DeepLabCut) override this to submit crops in one forward.
        """

        return [
            self.estimate(
                request.frame_bgr_original,
                request.bbox,
                frame_idx=request.frame_idx,
                mono_ts=request.mono_ts,
                wall_ts=request.wall_ts,
                source_id=request.source_id,
            )
            for request in requests
        ]

    def telemetry_snapshot(self) -> PoseTelemetrySnapshot | None:
        """Return perf/health telemetry if the backend tracks it, else ``None``."""

        return None


# Canonical side-on dog skeleton as fractions of the bbox (x: head->tail, y down).
# Values mirror the VERIFIED anatomy from the spike: withers in front, hips in rear,
# front paws under the shoulders, hind paws under the pelvis.
_ROLE_FRACTIONS_STAND: dict[str, tuple[float, float]] = {
    HEAD: (0.05, 0.55),
    NECK: (0.20, 0.42),
    WITHERS: (0.28, 0.30),
    SPINE_MID: (0.50, 0.30),
    HIPS: (0.75, 0.33),
    TAIL_END: (0.93, 0.20),
    FRONT_LEFT_PAW: (0.24, 0.96),
    FRONT_RIGHT_PAW: (0.27, 0.96),
    HIND_LEFT_PAW: (0.71, 0.96),
    HIND_RIGHT_PAW: (0.74, 0.96),
    BELLY: (0.46, 0.66),
}

# Squat: pelvis drops and tucks forward, back arches up, hind paws tuck under hips,
# belly lowers. This is what gives pose features a real stand-vs-squat signal.
_ROLE_FRACTIONS_SQUAT: dict[str, tuple[float, float]] = {
    HEAD: (0.07, 0.62),
    NECK: (0.22, 0.48),
    WITHERS: (0.30, 0.40),
    SPINE_MID: (0.52, 0.24),
    HIPS: (0.70, 0.55),
    TAIL_END: (0.86, 0.34),
    FRONT_LEFT_PAW: (0.26, 0.96),
    FRONT_RIGHT_PAW: (0.29, 0.96),
    HIND_LEFT_PAW: (0.66, 0.96),
    HIND_RIGHT_PAW: (0.69, 0.96),
    BELLY: (0.48, 0.82),
}


def build_synthetic_pose(
    bbox: BBox,
    frame_idx: int = 0,
    mono_ts: float = 0.0,
    posture: str = "stand",
    confidence: float = 0.9,
    missing_roles: tuple[str, ...] = (),
    facing: str = "left",
    backend: str = "mock",
    **provenance: object,
) -> PoseKeypoints:
    """Build a deterministic synthetic pose inside ``bbox`` (original-res coords).

    ``posture`` is ``"stand"`` or ``"squat"``; ``missing_roles`` are dropped (to
    exercise the confidence-gated fallback paths); ``facing`` flips the dog left or
    right within the box so tests cover both orientations.
    """

    fractions = _ROLE_FRACTIONS_SQUAT if posture == "squat" else _ROLE_FRACTIONS_STAND
    x1, y1 = bbox.x1, bbox.y1
    width, height = bbox.width, bbox.height
    missing = set(missing_roles)

    named: dict[str, tuple[float, float, float]] = {}
    for role, (fx, fy) in fractions.items():
        if role in missing:
            continue
        use_fx = fx if facing == "left" else (1.0 - fx)
        px = x1 + use_fx * width
        py = y1 + fy * height
        raw_name = resolve_role(role)[0]
        named[raw_name] = (px, py, confidence)

    return PoseKeypoints.from_mapping(
        named,
        frame_idx=frame_idx,
        mono_ts=mono_ts,
        backend=backend,
        model_name="synthetic",
        crop_bbox=bbox,
        **provenance,
    )


class MockPoseEstimator(PoseEstimator):
    """Deterministic estimator that synthesizes a pose from the bbox (no model)."""

    def __init__(
        self,
        posture: str = "stand",
        confidence: float = 0.9,
        missing_roles: tuple[str, ...] = (),
        facing: str = "left",
        per_frame_posture: Mapping[int, str] | None = None,
    ) -> None:
        self.posture = posture
        self.confidence = confidence
        self.missing_roles = tuple(missing_roles)
        self.facing = facing
        self.per_frame_posture = dict(per_frame_posture or {})

    def estimate(
        self,
        frame_bgr_original: np.ndarray,
        bbox: BBox,
        frame_idx: int = 0,
        mono_ts: float | None = None,
        wall_ts: datetime | None = None,
        source_id: str | None = None,
    ) -> PoseKeypoints | None:
        if frame_bgr_original.ndim < 2:
            raise ValueError("frame_bgr_original must be an image array")
        if bbox.width <= 0 or bbox.height <= 0:
            return None
        posture = self.per_frame_posture.get(frame_idx, self.posture)
        height, width = frame_bgr_original.shape[:2]
        return build_synthetic_pose(
            bbox=bbox,
            frame_idx=frame_idx,
            mono_ts=0.0 if mono_ts is None else mono_ts,
            posture=posture,
            confidence=self.confidence,
            missing_roles=self.missing_roles,
            facing=self.facing,
            wall_ts=wall_ts,
            source_id=source_id,
            image_size=(width, height),
        )
