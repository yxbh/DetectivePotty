"""Weak v0 pee-vs-poop classifier for human review prefill."""

from __future__ import annotations

from typing import Sequence

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.events import ClassifierGuess, Track
from detectivepotty.sources.base import Frame


class HeuristicPottyClassifier(PottyClassifier):
    """Return a modest, non-ground-truth pee/poop guess from bbox posture.

    This classifier exists only to pre-populate review UI metadata. It must not be
    treated as a label; ``needs_label`` is always true until a human verifies the
    event.
    """

    def __init__(
        self,
        poop_duration_s: float = 5.0,
        deep_squat_threshold: float = 0.35,
    ) -> None:
        self.poop_duration_s = poop_duration_s
        self.deep_squat_threshold = deep_squat_threshold

    def classify(
        self,
        track: Track,
        frames: Sequence[Frame],
    ) -> ClassifierResult:
        del frames
        detections = track.detections
        if len(detections) < 2:
            return ClassifierResult(
                guess=ClassifierGuess.UNKNOWN,
                confidence=0.0,
                needs_label=True,
            )

        duration_s = max(0.0, detections[-1].mono_ts - detections[0].mono_ts)
        heights = [detection.bbox.height for detection in detections]
        aspects = [
            detection.bbox.width / detection.bbox.height
            for detection in detections
            if detection.bbox.height > 0.0
        ]
        max_height = max(heights, default=0.0)
        min_height = min(heights, default=0.0)
        height_drop = 0.0
        if max_height > 0.0:
            height_drop = max(0.0, (max_height - min_height) / max_height)
        aspect_change = 0.0
        if aspects and min(aspects) > 0.0:
            aspect_change = max(0.0, (max(aspects) - min(aspects)) / min(aspects))
        squat_depth = max(height_drop, aspect_change)

        if duration_s >= self.poop_duration_s and squat_depth >= self.deep_squat_threshold:
            confidence = min(0.65, 0.45 + 0.15 * squat_depth + 0.02 * duration_s)
            return ClassifierResult(
                guess=ClassifierGuess.POOP,
                confidence=confidence,
                needs_label=True,
            )
        if duration_s > 0.0 or squat_depth > 0.0:
            confidence = min(0.6, 0.35 + 0.10 * squat_depth + 0.01 * duration_s)
            return ClassifierResult(
                guess=ClassifierGuess.PEE,
                confidence=confidence,
                needs_label=True,
            )
        return ClassifierResult(
            guess=ClassifierGuess.UNKNOWN,
            confidence=0.2,
            needs_label=True,
        )
