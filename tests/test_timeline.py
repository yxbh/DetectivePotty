from __future__ import annotations

import pytest

from detectivepotty.timeline import FrameTimeline, maybe_pts_times


def test_cfr_timeline_matches_existing_frame_math() -> None:
    timeline = FrameTimeline.cfr(fps=10.0, frame_count=21)

    assert timeline.frame_to_seconds(6) == pytest.approx(0.6)
    assert timeline.seconds_to_frame_nearest(1.4) == 14
    assert timeline.sample_frames_by_time(0, 20, stride_s=0.3, max_frames=40) == [
        0,
        3,
        6,
        9,
        12,
        15,
        18,
    ]


def test_pts_timeline_maps_seconds_to_nearest_frames() -> None:
    timeline = FrameTimeline.from_frame_times([0.0, 0.1, 0.4, 0.45, 1.0], fps=10.0)

    assert timeline.frame_to_seconds(2) == pytest.approx(0.4)
    assert timeline.seconds_to_frame_floor(0.39) == 1
    assert timeline.seconds_to_frame_ceil(0.39) == 2
    assert timeline.seconds_to_frame_nearest(0.42) == 2
    assert timeline.sample_frames_by_time(0, 4, stride_s=0.3, max_frames=40) == [0, 2, 4]


def test_pts_timeline_rejects_non_monotonic_times() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        FrameTimeline.from_frame_times([0.0, 0.4, 0.3], fps=10.0)


def test_maybe_pts_times_omits_cfr_equivalent_times() -> None:
    assert maybe_pts_times([0.0, 0.1, 0.2], fps=10.0) is None
    assert maybe_pts_times([0.0, 0.1, 0.25], fps=10.0) == (0.0, 0.1, 0.25)
