"""Pose-aware augmentation for the detection gate (Phase 4, opt-in).

The bbox-only posture heuristics in :mod:`detectivepotty.potty_event` are noisy.
When ``pose.enable_pose_gate`` is set, :class:`PoseGate` runs the configured pose
estimator on each sampled frame's detections and reduces a track's trailing window
to :class:`~detectivepotty.pose.features.PoseFeatures`, exposing trustworthy
``pose_squat`` / ``pose_stationary`` signals that the state machine combines with
the bbox signals.

Safety properties (so enabling the gate cannot quietly regress the recall fix):

* Pose is **additive**. It can flip a frame's ``is_squat`` to True and relax the
  bbox motion-jitter check, but it never removes the bbox ``covered_long_enough``
  requirement, so the candidate-window timing structure is preserved.
* Pose is only trusted when it is **not sparse**: enough successful poses over the
  window (``min_required_frames``) AND a high enough success fraction of the
  attempted detections (``min_pose_coverage``) AND the window-level
  ``fallback_recommended`` flag is clear. Otherwise the gate returns ``None`` and
  the state machine uses bbox only.
* Estimator failures are swallowed per detection (logged, rate-limited) so one bad
  crop never crashes a camera thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Sequence

from detectivepotty.events import Detection
from detectivepotty.pose.features import PoseFeatures, extract_pose_features
from detectivepotty.pose.keypoints import PoseKeypoints
from detectivepotty.sources.base import Frame

LOGGER = logging.getLogger(__name__)

# How many estimator failures to log before going quiet (one bad crop must not
# spam the logs every frame for a whole clip).
_MAX_FAILURE_LOGS = 5

# Estimate one dog's pose from a frame + detection, or None when unavailable.
PoseEstimateFn = Callable[[Frame, Detection], PoseKeypoints | None]


@dataclass(frozen=True, slots=True)
class PoseGateThresholds:
    """Posture thresholds for the pose-derived squat/stationary signals.

    Mirrors the classifier's squat thresholds (``classify.pose``) so the gate and
    the classifier agree on what "arched"/"deep"/"stationary" mean.
    """

    arched_spine_deg: float = 150.0
    deep_squat_ratio: float = 0.12
    stationary_motion: float = 0.6


@dataclass(frozen=True, slots=True)
class PoseGateResult:
    """Trustworthy pose signals for one track window (None fields = no signal)."""

    pose_squat: bool | None
    pose_stationary: bool | None
    valid_frames: int
    attempted_frames: int
    features: PoseFeatures

    def summary(self) -> dict[str, Any]:
        return {
            "pose_squat": self.pose_squat,
            "pose_stationary": self.pose_stationary,
            "pose_valid_frames": self.valid_frames,
            "pose_attempted_frames": self.attempted_frames,
            "pose_coverage": (
                self.valid_frames / self.attempted_frames
                if self.attempted_frames
                else 0.0
            ),
            "pose_spine_angle_deg": self.features.spine_angle_deg,
            "pose_hip_offset_ratio": self.features.hip_offset_ratio,
            "pose_centroid_motion_ratio": self.features.centroid_motion_ratio,
        }


@dataclass(slots=True)
class _PoseEntry:
    frame_idx: int
    mono_ts: float
    pose: PoseKeypoints | None  # None = attempted this frame but produced no pose.


class PoseGate:
    """Per-camera pose cache + posture reducer for the detection gate."""

    def __init__(
        self,
        estimate_fn: PoseEstimateFn,
        *,
        min_keypoint_conf: float = 0.5,
        min_required_frames: int = 3,
        min_pose_coverage: float = 0.5,
        min_torso_keypoints: int = 3,
        thresholds: PoseGateThresholds | None = None,
    ) -> None:
        self._estimate_fn = estimate_fn
        self._min_keypoint_conf = min_keypoint_conf
        self._min_required_frames = min_required_frames
        self._min_pose_coverage = min_pose_coverage
        self._min_torso_keypoints = min_torso_keypoints
        self._thresholds = thresholds or PoseGateThresholds()
        self._by_id: dict[int, _PoseEntry] = {}
        self._failure_logs = 0

    def observe(self, frame: Frame, detections: Sequence[Detection]) -> None:
        """Run pose for each detection this frame; store keyed by object identity.

        A failure or empty result is stored as an attempted-but-None entry so the
        coverage math below sees the sparsity instead of silently ignoring it.
        """

        for detection in detections:
            try:
                pose = self._estimate_fn(frame, detection)
            except Exception:  # noqa: BLE001 - one bad crop must not kill the thread.
                pose = None
                if self._failure_logs < _MAX_FAILURE_LOGS:
                    self._failure_logs += 1
                    LOGGER.warning(
                        "Pose gate estimate failed (frame %s); using bbox.",
                        detection.frame_idx,
                        exc_info=True,
                    )
            self._by_id[id(detection)] = _PoseEntry(
                detection.frame_idx,
                detection.mono_ts,
                pose,
            )

    def prune(self, keep_ids: set[int]) -> None:
        """Drop cached poses whose detections are no longer in any trailing window."""

        if not keep_ids:
            self._by_id.clear()
            return
        self._by_id = {key: value for key, value in self._by_id.items() if key in keep_ids}

    def posture(self, recent: Sequence[Detection]) -> PoseGateResult | None:
        """Return trustworthy pose signals for a track's trailing window, or None.

        Returns ``None`` (→ bbox-only) when pose is too sparse or low-quality. The
        sparsity check is over *attempted* detections, not just the successful
        poses, so a window where pose succeeded on only a few of many frames is
        correctly distrusted.
        """

        if not recent:
            return None

        attempted = 0
        poses: list[PoseKeypoints] = []
        for detection in recent:
            entry = self._by_id.get(id(detection))
            if (
                entry is None
                or entry.frame_idx != detection.frame_idx
                or entry.mono_ts != detection.mono_ts
            ):
                continue
            attempted += 1
            if entry.pose is not None:
                poses.append(entry.pose)

        if attempted == 0 or len(poses) < self._min_required_frames:
            return None
        if len(poses) / attempted < self._min_pose_coverage:
            return None

        features = extract_pose_features(
            poses,
            min_keypoint_conf=self._min_keypoint_conf,
            min_required_frames=self._min_required_frames,
            min_pose_coverage=self._min_pose_coverage,
            min_torso_keypoints=self._min_torso_keypoints,
        )
        if features is None or features.fallback_recommended:
            return None

        pose_squat: bool | None = None
        if features.spine_angle_deg is not None or features.hip_offset_ratio is not None:
            arched = (
                features.spine_angle_deg is not None
                and features.spine_angle_deg < self._thresholds.arched_spine_deg
            )
            deep = (
                features.hip_offset_ratio is not None
                and features.hip_offset_ratio > self._thresholds.deep_squat_ratio
            )
            pose_squat = arched or deep

        pose_stationary: bool | None = None
        if features.centroid_motion_ratio is not None:
            pose_stationary = (
                features.centroid_motion_ratio < self._thresholds.stationary_motion
            )

        if pose_squat is None and pose_stationary is None:
            return None

        return PoseGateResult(
            pose_squat=pose_squat,
            pose_stationary=pose_stationary,
            valid_frames=len(poses),
            attempted_frames=attempted,
            features=features,
        )
