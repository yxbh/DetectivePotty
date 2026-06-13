from __future__ import annotations

from datetime import datetime, timezone

import pytest

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.tracking import Tracker, iou, temporal_box_union


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


def test_temporal_box_union_disabled_returns_reference_box() -> None:
    dets = [make_detection(0, BBox(0, 0, 10, 10)), make_detection(1, BBox(0, 0, 40, 40))]

    out = temporal_box_union(dets, dets[1], window_s=0.0)

    # Disabled path must return the exact same box object (byte-identical crop).
    assert out is dets[1].bbox


def test_temporal_box_union_recovers_extent_within_window() -> None:
    # The reference frame (f2) under-segmented; earlier in-window frames caught
    # slightly more of the dog, so the union recovers the fuller extent.
    dets = [
        make_detection(0, BBox(8, 8, 36, 36)),
        make_detection(1, BBox(10, 10, 34, 34)),
        make_detection(2, BBox(12, 12, 32, 32)),
    ]

    out = temporal_box_union(dets, dets[2], window_s=5.0)

    assert out == BBox(8, 8, 36, 36)


def test_temporal_box_union_respects_trailing_window() -> None:
    # f2 is a huge box but falls OUTSIDE the trailing window, so it must not leak
    # into the union; only f3/f4/f5 contribute.
    dets = [
        make_detection(2, BBox(0, 0, 100, 100)),
        make_detection(3, BBox(40, 40, 60, 60)),
        make_detection(4, BBox(42, 42, 62, 62)),
        make_detection(5, BBox(44, 44, 64, 64)),
    ]

    out = temporal_box_union(dets, dets[3], window_s=2.0)

    assert out == BBox(40, 40, 64, 64)


def test_temporal_box_union_skips_far_drifted_boxes() -> None:
    # A faraway box (a moving/other dog) within the window is excluded so the crop
    # is not elongated along the path; a nearby box is still merged.
    dets = [
        make_detection(0, BBox(100, 100, 120, 120)),
        make_detection(1, BBox(2, 2, 30, 30)),
        make_detection(2, BBox(0, 0, 20, 20)),
    ]

    out = temporal_box_union(dets, dets[2], window_s=10.0)

    assert out == BBox(0, 0, 30, 30)


def test_temporal_box_union_growth_cap_rejects_ballooning() -> None:
    # A concentric but enormous box passes the center-shift guard; the final area
    # cap then rejects the poisoned union and keeps the raw reference box.
    dets = [
        make_detection(0, BBox(-20, -20, 50, 50)),
        make_detection(1, BBox(10, 10, 20, 20)),
    ]

    out = temporal_box_union(dets, dets[1], window_s=10.0)

    assert out == BBox(10, 10, 20, 20)


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


def test_sparse_source_frame_gaps_age_by_tracker_update() -> None:
    tracker = Tracker(max_age_frames=1)

    tracker.update([make_detection(0, BBox(0, 0, 10, 10))])
    one_miss = tracker.update([make_detection(6, BBox(50, 50, 60, 60))])

    assert {track.track_id for track in one_miss} == {"1", "2"}

    two_misses = tracker.update([make_detection(12, BBox(50, 50, 60, 60))])

    assert {track.track_id for track in two_misses} == {"2"}


def test_inactive_histories_are_bounded() -> None:
    tracker = Tracker(max_age_frames=0, max_history_tracks=2)

    for idx, x in enumerate((0, 50, 100)):
        tracker.update([make_detection(idx * 2, BBox(x, 0, x + 10, 10))])
        tracker.update([])

    assert [track.track_id for track in tracker.histories] == ["2", "3"]
    assert tracker.get_track("1") is None


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


def test_center_dist_gate_reassociates_nonoverlapping_box() -> None:
    # A dog that moved far enough between sparse samples for its box to stop
    # overlapping (IoU == 0) would fragment into a new track under pure IoU. The
    # center-distance gate keeps it as one track (this is the harvest fix).
    moved = BBox(25, 0, 45, 20)
    assert iou(BBox(0, 0, 20, 20), moved) == 0.0

    tracker = Tracker(iou_threshold=0.3, center_dist_gate=1.5)
    first = tracker.update([make_detection(0, BBox(0, 0, 20, 20))])
    second = tracker.update([make_detection(1, moved)])

    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
    assert len(second[0].detections) == 2


def test_center_dist_gate_disabled_by_default_keeps_pure_iou() -> None:
    # Default gate (0.0) must not change pure-IoU behavior: the same
    # non-overlapping jump births a new track, the live-pipeline contract.
    tracker = Tracker(iou_threshold=0.3)
    first = tracker.update([make_detection(0, BBox(0, 0, 20, 20))])
    second = tracker.update([make_detection(1, BBox(25, 0, 45, 20))])

    assert first[0].track_id == "1"
    assert {track.track_id for track in second} == {"1", "2"}
    moved_track = next(track for track in second if track.track_id == "2")
    assert len(moved_track.detections) == 1


def test_center_dist_gate_respects_threshold() -> None:
    # A box beyond the gate (here ~0.88 box-diagonals away, gate 0.5) still
    # fragments — the gate widens association, it does not merge anything nearby.
    tracker = Tracker(iou_threshold=0.3, center_dist_gate=0.5)
    tracker.update([make_detection(0, BBox(0, 0, 20, 20))])
    second = tracker.update([make_detection(1, BBox(25, 0, 45, 20))])

    assert {track.track_id for track in second} == {"1", "2"}


def test_center_dist_gate_rejects_negative() -> None:
    with pytest.raises(ValueError):
        Tracker(center_dist_gate=-0.1)
