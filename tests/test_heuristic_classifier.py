from __future__ import annotations

from datetime import datetime, timezone

from detectivepotty.classify.base import ClassifierResult
from detectivepotty.classify.heuristic import HeuristicPottyClassifier
from detectivepotty.events import ClassifierGuess, Detection, Track
from detectivepotty.geometry import BBox


def detection(frame_idx: int, bbox: BBox) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=0.9,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_heuristic_classifier_returns_reviewable_result() -> None:
    track = Track(
        track_id="dog-1",
        detections=[
            detection(0, BBox(0, 0, 40, 80)),
            detection(5, BBox(0, 20, 70, 70)),
        ],
    )

    result = HeuristicPottyClassifier().classify(track, frames=[])

    assert isinstance(result, ClassifierResult)
    assert result.needs_label is True
    assert result.guess in set(ClassifierGuess)
    assert 0.0 <= result.confidence <= 1.0
