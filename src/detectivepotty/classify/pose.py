"""Pose-based v0 pee-vs-poop classifier (drop-in for the bbox heuristic).

Runs the configured :class:`~detectivepotty.pose.base.PoseEstimator` over the
candidate track's frames, reduces the window to view-robust
:class:`~detectivepotty.pose.features.PoseFeatures`, and maps the posture
signature to a pee/poop guess. Poop has the stronger postural fingerprint
(sustained, deep, arched, stationary squat — corroborated by independent prior
art), so the decision leans on that signature and defaults to pee otherwise.

The guess remains a weak v0 prefill for human review: ``needs_label`` stays
``True``. The classifier falls back to the bbox heuristic whenever pose is
unavailable (estimator missing, no usable keypoints) or too low-quality to trust
(``fallback_recommended``), so enabling it can never do worse than the heuristic.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Sequence

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.config import PoseConfig
from detectivepotty.events import ClassifierGuess, Detection, Track
from detectivepotty.pose.base import PoseEstimator
from detectivepotty.pose.features import PoseFeatures, extract_pose_features
from detectivepotty.pose.keypoints import PoseKeypoints
from detectivepotty.sources.base import Frame
from detectivepotty.tracking import temporal_box_union


@dataclass(frozen=True, slots=True)
class PoseDecisionThresholds:
    """Tunable thresholds for the pose posture-signature decision.

    Defaults are deliberately modest; pee-vs-poop from a fixed camera is genuinely
    hard and the output is a review prefill, not ground truth.
    """

    poop_dwell_s: float = 5.0
    arched_spine_deg: float = 150.0
    deep_squat_ratio: float = 0.12
    stationary_motion: float = 0.6
    tail_up_deg: float = 110.0
    leg_lift_asym: float = 0.25
    poop_fraction: float = 0.6
    min_posture_signals: int = 2


def classify_pose_features(
    features: PoseFeatures,
    thresholds: PoseDecisionThresholds = PoseDecisionThresholds(),
) -> tuple[ClassifierGuess, float] | None:
    """Map window-level posture features to a (guess, confidence) pair.

    Counts how many of the available poop signals (arched back, deep squat,
    stationary, tail up, long dwell) are present; a high fraction reads as poop
    unless a clear hind-leg lift pulls the guess toward pee.

    Returns ``None`` when there is too little discriminative posture evidence
    (fewer than ``min_posture_signals`` geometric signals, or neither core squat
    signal available) so the caller can fall back to the bbox heuristic rather than
    guess from one stray signal plus dwell.
    """

    geom_signals: list[bool] = []
    has_core = False
    if features.spine_angle_deg is not None:
        geom_signals.append(features.spine_angle_deg < thresholds.arched_spine_deg)
        has_core = True
    if features.hip_offset_ratio is not None:
        geom_signals.append(features.hip_offset_ratio > thresholds.deep_squat_ratio)
        has_core = True
    if features.centroid_motion_ratio is not None:
        geom_signals.append(features.centroid_motion_ratio < thresholds.stationary_motion)
    if features.tail_angle_deg is not None:
        geom_signals.append(features.tail_angle_deg >= thresholds.tail_up_deg)

    if len(geom_signals) < thresholds.min_posture_signals or not has_core:
        return None

    signals = [*geom_signals, features.dwell_duration_s >= thresholds.poop_dwell_s]
    fraction = sum(signals) / len(signals)
    leg_lift = (
        features.hind_paw_asymmetry is not None
        and features.hind_paw_asymmetry >= thresholds.leg_lift_asym
    )

    if fraction >= thresholds.poop_fraction and not leg_lift:
        confidence = min(0.65, 0.40 + 0.25 * fraction)
        return ClassifierGuess.POOP, confidence

    confidence = min(0.6, 0.35 + 0.20 * (1.0 - fraction))
    if leg_lift:
        confidence = min(0.6, confidence + 0.1)
    return ClassifierGuess.PEE, confidence


def _evenly_sample(items: list, cap: int) -> list:
    """Return at most ``cap`` items spread evenly across ``items``."""

    n = len(items)
    if cap <= 0 or n <= cap:
        return list(items)
    step = n / cap
    return [items[int(i * step)] for i in range(cap)]


class PosePottyClassifier(PottyClassifier):
    """Pee-vs-poop guess from keypoint posture, with a heuristic fallback."""

    def __init__(
        self,
        estimator: PoseEstimator,
        pose_config: PoseConfig,
        fallback: PottyClassifier,
        *,
        inference_lock: ContextManager | None = None,
        max_pose_frames: int = 30,
        thresholds: PoseDecisionThresholds | None = None,
    ) -> None:
        self._estimator = estimator
        self._pose = pose_config
        self._fallback = fallback
        # Pose inference shares the accelerator with detection; the pipeline passes
        # its inference lock so a finalization-time pose pass on one camera does not
        # run concurrently with another camera's detection. Lock is acquired per
        # frame (not for the whole window) so detection can still interleave.
        self._lock: ContextManager = nullcontext() if inference_lock is None else inference_lock
        self._max_pose_frames = max_pose_frames
        self._thresholds = thresholds or PoseDecisionThresholds()

    def classify(self, track: Track, frames: Sequence[Frame]) -> ClassifierResult:
        poses = self._estimate_window(track, frames)
        if not poses:
            # No pose at all: defer entirely to the heuristic, with no pose data.
            return self._fallback.classify(track, frames)

        result = self._guess(track, frames, poses)
        # Carry the keypoints (and features when usable) out for persistence and
        # overlay even when the guess itself came from the heuristic fallback.
        result.poses = poses
        return result

    def _guess(
        self,
        track: Track,
        frames: Sequence[Frame],
        poses: list[PoseKeypoints],
    ) -> ClassifierResult:
        features = extract_pose_features(
            poses,
            min_keypoint_conf=self._pose.min_keypoint_conf,
            min_required_frames=self._pose.min_required_frames,
            min_pose_coverage=self._pose.min_pose_coverage,
            min_torso_keypoints=self._pose.min_torso_keypoints,
        )
        if features is None or features.fallback_recommended:
            return self._fallback.classify(track, frames)

        decision = classify_pose_features(features, self._thresholds)
        if decision is None:
            result = self._fallback.classify(track, frames)
            result.pose_features = features
            return result

        guess, confidence = decision
        return ClassifierResult(
            guess=guess,
            confidence=confidence,
            needs_label=True,
            pose_features=features,
        )

    def _estimate_window(
        self,
        track: Track,
        frames: Sequence[Frame],
    ) -> list[PoseKeypoints]:
        frames_by_idx = {frame.frame_idx: frame for frame in frames}
        # Keep one detection per frame_idx (highest confidence) so a tracker that
        # ever emits duplicates can't pose the same image twice and inflate quality.
        by_frame: dict[int, Detection] = {}
        for detection in track.detections:
            if detection.frame_idx not in frames_by_idx:
                continue
            existing = by_frame.get(detection.frame_idx)
            if existing is None or detection.confidence > existing.confidence:
                by_frame[detection.frame_idx] = detection
        if not by_frame:
            return []
        detections = sorted(by_frame.values(), key=lambda det: det.frame_idx)

        poses: list[PoseKeypoints] = []
        for detection in _evenly_sample(detections, self._max_pose_frames):
            frame = frames_by_idx[detection.frame_idx]
            # Recover full dog extent from a short trailing window so an
            # under-segmented IR box does not feed pose a partial crop. The
            # estimator still applies its own crop_margin_frac on top. Disabled
            # (window 0.0) returns the raw detector box unchanged.
            pose_bbox = temporal_box_union(
                detections,
                detection,
                self._pose.box_union_window_s,
            )
            with self._lock:
                pose = self._estimator.estimate(
                    frame.bgr,
                    pose_bbox,
                    frame_idx=detection.frame_idx,
                    mono_ts=detection.mono_ts,
                    wall_ts=detection.wall_ts,
                    source_id=frame.source_id,
                )
            if pose is not None:
                poses.append(pose)
        return poses
