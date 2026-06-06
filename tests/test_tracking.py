from __future__ import annotations

from datetime import datetime, timezone

import pytest

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.tracking import Tracker, iou


def make_detection(frame_idx: int, bbox: BBox, confidence: float = 0.9) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_iou_correctness() -> None:
    assert iou(BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)) == pytest.approx(1 / 7)
    assert iou(BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)) == 0.0
    assert iou(BBox(0, 0, 0, 10), BBox(0, 0, 10, 10)) == 0.0


def test_stable_ids_across_moving_boxes() -> None:
    tracker = Tracker(iou_threshold=0.2)

    first = tracker.update([make_detection(0, BBox(0, 0, 20, 20))])
    second = tracker.update([make_detection(1, BBox(2, 1, 22, 21))])

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
    assert len(second[0].detections) == 2


def test_track_birth_and_death_after_max_age() -> None:
    tracker = Tracker(max_age_frames=1)

    first = tracker.update([make_detection(0, BBox(0, 0, 10, 10))])
    assert [track.track_id for track in first] == ["1"]

    assert [track.track_id for track in tracker.update([])] == ["1"]
    assert tracker.update([]) == []

    reborn = tracker.update([make_detection(3, BBox(0, 0, 10, 10))])
    assert [track.track_id for track in reborn] == ["2"]
    assert [track.track_id for track in tracker.histories] == ["1", "2"]


def test_overlapping_dogs_have_no_reidentification_guarantee() -> None:
    tracker = Tracker(iou_threshold=0.1, max_age_frames=2)
    first = tracker.update(
        [
            make_detection(0, BBox(0, 0, 20, 20)),
            make_detection(0, BBox(80, 0, 100, 20)),
        ]
    )
    original_ids = {track.track_id for track in first}

    tracker.update(
        [
            make_detection(1, BBox(35, 0, 55, 20)),
            make_detection(1, BBox(45, 0, 65, 20)),
        ]
    )
    crossed = tracker.update(
        [
            make_detection(2, BBox(80, 0, 100, 20)),
            make_detection(2, BBox(0, 0, 20, 20)),
        ]
    )

    final_ids = {track.track_id for track in crossed}
    assert original_ids <= final_ids
    assert len(crossed) >= 2
    assert all(track.detections for track in crossed)
