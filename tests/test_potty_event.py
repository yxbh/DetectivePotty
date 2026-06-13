from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from detectivepotty.config import CameraConfig, ZoneConfig
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.potty_event import PottyEventDetector
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def camera_config(**overrides: object) -> CameraConfig:
    values = {
        "id": "cam-1",
        "name": "Backyard",
        "detection_conf_threshold": 0.25,
        "event_duration_s": 2.0,
        "stationary_threshold_s": 1.0,
        "dwell_trigger_s": 5.0,
        "sample_rate_fps": 1.0,
    }
    values.update(overrides)
    return CameraConfig(**values)


def frame(frame_idx: int) -> Frame:
    return Frame(
        bgr=np.zeros((120, 160, 3), dtype=np.uint8),
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
        source_id="camera://cam-1",
    )


def detection(frame_idx: int, bbox: BBox, confidence: float = 0.9) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
    )


def standing(x: float = 40.0) -> BBox:
    return BBox(x, 20, x + 40, 100)


def run_sequence(
    detector: PottyEventDetector,
    boxes_by_frame: list[list[BBox]],
) -> list:
    emitted = []
    for frame_idx, boxes in enumerate(boxes_by_frame):
        emitted.extend(
            detector.process(
                frame(frame_idx),
                [detection(frame_idx, bbox) for bbox in boxes],
            )
        )
    emitted.extend(detector.flush())
    return emitted


def test_stationary_hold_emits_one_potty_candidate() -> None:
    # A dog that holds still for >= dwell_trigger_s becomes a potty candidate.
    detector = PottyEventDetector(camera_config())

    emitted = run_sequence(detector, [[standing()] for _ in range(8)])

    assert len(emitted) == 1
    candidate = emitted[0]
    assert candidate.camera_id == "cam-1"
    assert candidate.primary_track_id == "1"
    assert candidate.start_ts == BASE_TS
    assert candidate.near_miss is False
    assert candidate.stationary_duration_s >= 1.0
    assert candidate.posture_summary["dwell_duration_s"] >= 5.0
    assert [track.track_id for track in candidate.tracks] == ["1"]


def test_walking_through_emits_no_event() -> None:
    detector = PottyEventDetector(camera_config())

    emitted = run_sequence(
        detector,
        [
            [standing(0)],
            [standing(35)],
            [standing(70)],
            [standing(105)],
            [standing(140)],
        ],
    )

    assert emitted == []


def test_two_dogs_sets_multi_dog_and_ambiguous_flags() -> None:
    detector = PottyEventDetector(camera_config())

    emitted = run_sequence(
        detector,
        [[standing(25), standing(95)] for _ in range(8)],
    )

    assert len(emitted) == 1
    assert emitted[0].multi_dog is True
    assert emitted[0].ambiguous is True
    assert {track.track_id for track in emitted[0].tracks} == {"1", "2"}


def test_candidate_stats_stay_with_primary_track() -> None:
    detector = PottyEventDetector(
        camera_config(dwell_trigger_s=2.0, event_duration_s=4.0),
    )

    emitted = run_sequence(
        detector,
        [[standing(20)] for _ in range(3)]
        + [[standing(95)] for _ in range(4)],
    )

    assert len(emitted) == 1
    candidate = emitted[0]
    assert candidate.primary_track_id == "1"
    assert candidate.posture_summary["posture_window_start_mono"] == 1.0
    assert candidate.posture_summary["posture_window_end_mono"] == 2.0
    assert candidate.posture_summary["dwell_duration_s"] == 2.0


def test_last_frame_does_not_retain_full_image() -> None:
    detector = PottyEventDetector(camera_config())
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    the_frame = Frame(
        bgr=image,
        frame_idx=0,
        mono_ts=0.0,
        wall_ts=BASE_TS,
        source_id="camera://cam-1",
    )

    detector.process(the_frame, [detection(0, standing())])

    assert detector._last_frame is not None
    assert detector._last_frame.bgr.size == 0
    assert detector._last_frame.bgr.base is None


def test_suppressed_only_track_does_not_grow_window() -> None:
    detector = PottyEventDetector(
        camera_config(dwell_trigger_s=5.0, event_duration_s=1.0),
    )

    emitted = []
    for frame_idx in range(7):
        emitted.extend(detector.process(frame(frame_idx), [detection(frame_idx, standing())]))
    assert len(emitted) == 1

    for frame_idx in range(7, 11):
        assert detector.process(frame(frame_idx), [detection(frame_idx, standing())]) == []
        assert detector._window_detections == []


def test_continuous_track_can_emit_again_after_suppression_cooldown() -> None:
    detector = PottyEventDetector(
        camera_config(dwell_trigger_s=2.0, event_duration_s=1.0),
    )

    emitted = run_sequence(detector, [[standing()] for _ in range(8)])

    assert len(emitted) == 2
    assert [candidate.primary_track_id for candidate in emitted] == ["1", "1"]


def test_roi_and_ignore_zone_filtering() -> None:
    # Zones are normalized [0.0, 1.0] image coordinates; the test frame is 160x120,
    # so a detection's pixel center is normalized before the polygon test. ROI keeps
    # the left ~3/4 of the frame; the ignore box carves out the middle.
    roi = ZoneConfig(points=[(0.0, 0.0), (0.75, 0.0), (0.75, 1.0), (0.0, 1.0)])
    ignore = ZoneConfig(points=[(0.2, 0.1), (0.6, 0.1), (0.6, 0.9), (0.2, 0.9)])
    config = camera_config(roi=[roi], ignore_zones=[ignore])

    # Center (60, 60) -> (0.375, 0.5): inside ROI but inside the ignore box.
    ignored_detector = PottyEventDetector(config)
    ignored = run_sequence(ignored_detector, [[standing(40)] for _ in range(8)])
    assert ignored == []

    # Center (150, 60) -> (0.9375, 0.5): outside the ROI.
    outside_detector = PottyEventDetector(config)
    outside = run_sequence(outside_detector, [[standing(130)] for _ in range(8)])
    assert outside == []

    # Center (20, 60) -> (0.125, 0.5): inside ROI, left of the ignore box.
    allowed_detector = PottyEventDetector(config)
    allowed = run_sequence(allowed_detector, [[standing(0)] for _ in range(8)])
    assert len(allowed) == 1
    assert allowed[0].camera_id == "cam-1"


def test_stationary_window_emits_with_non_integer_fps_timestamps() -> None:
    """Regression: real cameras report non-integer fps (e.g. 30.00004), so a
    stationary window spanning ``stationary_threshold_s`` lands a few microseconds
    below the threshold (1.9999972677...s for a 2.0s window). A strict ``>=``
    comparison rejected every frame and no event ever fired. The detector must
    tolerate that sub-threshold rounding and still read the window as stationary.
    """

    fps = 30.000040983662547  # measured fps of the real sample clip
    sample_every = 3  # 30fps source sampled at ~10fps
    base = 12345.678  # non-zero base mono clock, like time.monotonic()
    threshold_s = 2.0
    config = camera_config(
        stationary_threshold_s=threshold_s,
        dwell_trigger_s=1.0,
        sample_rate_fps=10.0,
        event_duration_s=2.0,
    )
    detector = PottyEventDetector(config)

    def at(sample_idx: int, bbox: BBox) -> tuple[Frame, Detection]:
        frame_idx = sample_idx * sample_every
        mono = base + frame_idx / fps
        wall = BASE_TS + timedelta(seconds=frame_idx / fps)
        the_frame = Frame(
            bgr=np.zeros((120, 160, 3), dtype=np.uint8),
            frame_idx=frame_idx,
            mono_ts=mono,
            wall_ts=wall,
            source_id="camera://cam-1",
        )
        the_detection = Detection(
            bbox=bbox,
            confidence=0.9,
            class_name="dog",
            frame_idx=frame_idx,
            mono_ts=mono,
            wall_ts=wall,
        )
        return the_frame, the_detection

    # ~3.8s of holding still in place so the 2.0s stationary window is fully covered.
    boxes = [standing()] * 38

    durations: list[float] = []
    original_posture_stats = PottyEventDetector._posture_stats

    def spy(self, track, current_mono):  # type: ignore[no-untyped-def]
        stats = original_posture_stats(self, track, current_mono)
        durations.append(stats.stationary_duration_s)
        return stats

    PottyEventDetector._posture_stats = spy  # type: ignore[assignment]
    emitted: list = []
    try:
        for sample_idx, bbox in enumerate(boxes):
            the_frame, the_detection = at(sample_idx, bbox)
            emitted.extend(detector.process(the_frame, [the_detection]))
        emitted.extend(detector.flush())
    finally:
        PottyEventDetector._posture_stats = original_posture_stats

    assert len(emitted) == 1
    # The measured window span is genuinely just below the threshold; the detector
    # reads it as stationary anyway thanks to the sampling/float tolerance.
    assert max(durations) < threshold_s
    assert abs(max(durations) - threshold_s) < 1e-3


def test_stationary_emits_when_window_span_stays_below_threshold() -> None:
    """Regression for the night-clip false negatives: with intermittent detections
    the trailing ``stationary_threshold_s`` window only ever *spans* up to one
    sample interval less than the threshold (e.g. 1.8s for a 2.0s threshold), so a
    strict ``window_span >= threshold`` check never fired even though the dog stood
    in one spot for many seconds. The gate tolerates a bounded coverage shortfall, so
    a long-lived stationary track must still emit exactly one event.
    """

    dt = 0.3  # sample spacing that does not divide the 2.0s threshold evenly
    threshold_s = 2.0
    config = camera_config(
        stationary_threshold_s=threshold_s,
        dwell_trigger_s=1.0,
        sample_rate_fps=10.0,
        event_duration_s=10.0,
    )
    detector = PottyEventDetector(config)

    def at(idx: int, bbox: BBox) -> tuple[Frame, Detection]:
        mono = idx * dt
        wall = BASE_TS + timedelta(seconds=mono)
        the_frame = Frame(
            bgr=np.zeros((120, 160, 3), dtype=np.uint8),
            frame_idx=idx,
            mono_ts=mono,
            wall_ts=wall,
            source_id="camera://cam-1",
        )
        the_detection = Detection(
            bbox=bbox,
            confidence=0.9,
            class_name="dog",
            frame_idx=idx,
            mono_ts=mono,
            wall_ts=wall,
        )
        return the_frame, the_detection

    boxes = [standing()] * 28

    emitted: list = []
    spans: list[float] = []
    original_posture_stats = PottyEventDetector._posture_stats

    def spy(self, track, current_mono):  # type: ignore[no-untyped-def]
        stats = original_posture_stats(self, track, current_mono)
        spans.append(stats.stationary_duration_s)
        return stats

    PottyEventDetector._posture_stats = spy  # type: ignore[assignment]
    try:
        for idx, bbox in enumerate(boxes):
            the_frame, the_detection = at(idx, bbox)
            emitted.extend(detector.process(the_frame, [the_detection]))
        emitted.extend(detector.flush())
    finally:
        PottyEventDetector._posture_stats = original_posture_stats

    # The window span never reaches the threshold; the old span-based gate would
    # have rejected every frame and emitted nothing.
    assert spans, "posture stats should have been evaluated"
    assert max(spans) < threshold_s - 1e-2
    assert len(emitted) == 1


def test_brief_stationary_after_long_walk_does_not_emit() -> None:
    """A long-lived track that walks across the frame and only briefly stops must NOT
    emit: the trailing window then still contains the moving samples (high centroid
    motion), so it is never stationary and never accrues dwell. Guards against the
    coverage tolerance over-crediting an old-but-moving track.
    """

    config = camera_config(
        stationary_threshold_s=2.0,
        dwell_trigger_s=5.0,
        sample_rate_fps=10.0,
        event_duration_s=2.0,
    )
    detector = PottyEventDetector(config)

    dt = 0.1
    emitted: list = []
    idx = 0

    def make(idx: int, bbox: BBox) -> tuple[Frame, Detection]:
        return (
            Frame(
                bgr=np.zeros((120, 600, 3), dtype=np.uint8),
                frame_idx=idx,
                mono_ts=idx * dt,
                wall_ts=BASE_TS + timedelta(seconds=idx * dt),
                source_id="camera://cam-1",
            ),
            Detection(
                bbox=bbox,
                confidence=0.9,
                class_name="dog",
                frame_idx=idx,
                mono_ts=idx * dt,
                wall_ts=BASE_TS + timedelta(seconds=idx * dt),
            ),
        )

    stationary_flags: list[bool] = []
    original_posture_stats = PottyEventDetector._posture_stats

    def spy(self, track, current_mono):  # type: ignore[no-untyped-def]
        stats = original_posture_stats(self, track, current_mono)
        stationary_flags.append(stats.is_stationary)
        return stats

    PottyEventDetector._posture_stats = spy  # type: ignore[assignment]
    try:
        # ~3s walking across the frame (tall standing box moves every sample).
        for step in range(30):
            x = float(step) * 15.0
            frame_obj, det = make(idx, BBox(x, 20, x + 40, 100))
            emitted.extend(detector.process(frame_obj, [det]))
            idx += 1

        # Then only ~0.3s of standing in place.
        stop_x = 30.0 * 15.0
        for _ in range(3):
            frame_obj, det = make(idx, BBox(stop_x, 20, stop_x + 40, 100))
            emitted.extend(detector.process(frame_obj, [det]))
            idx += 1
        emitted.extend(detector.flush())
    finally:
        PottyEventDetector._posture_stats = original_posture_stats

    # The trailing window always contains moving samples, so it never reads stationary.
    assert not any(stationary_flags)
    assert emitted == []


def test_long_stationary_hold_dwell_triggers_event() -> None:
    # A dog that holds still long enough (>= dwell_trigger_s) triggers a potty
    # candidate -- the viewpoint-invariant cue for high/top-down cameras.
    detector = PottyEventDetector(camera_config(dwell_trigger_s=5.0))

    emitted = run_sequence(detector, [[standing()] for _ in range(8)])

    assert len(emitted) == 1
    candidate = emitted[0]
    assert candidate.near_miss is False
    assert candidate.posture_summary["dwell_trigger_s"] == 5.0
    # The recorded dwell duration is the real continuous hold (>= the trigger).
    assert candidate.posture_summary["dwell_duration_s"] >= 5.0
    # Dwell confidence is moderate and scales with the hold length.
    assert 0.5 <= candidate.confidence <= 0.7


def test_dwell_confidence_grows_with_hold_length() -> None:
    short = PottyEventDetector(camera_config(dwell_trigger_s=5.0, event_duration_s=1.0))
    short_event = run_sequence(short, [[standing()] for _ in range(8)])[0]

    long = PottyEventDetector(camera_config(dwell_trigger_s=5.0, event_duration_s=5.0))
    long_event = run_sequence(long, [[standing()] for _ in range(12)])[0]

    assert long_event.posture_summary["dwell_duration_s"] > short_event.posture_summary[
        "dwell_duration_s"
    ]
    assert long_event.confidence > short_event.confidence


def test_short_stationary_hold_below_dwell_threshold_emits_nothing() -> None:
    # Stationary but for less than dwell_trigger_s: no event.
    detector = PottyEventDetector(camera_config(dwell_trigger_s=5.0))

    emitted = run_sequence(detector, [[standing()] for _ in range(4)])

    assert emitted == []


def test_motion_breaks_dwell_accumulation() -> None:
    # Two sub-threshold stationary holds split by a move must not sum into a trigger:
    # the dwell accumulator resets the moment the dog stops reading stationary.
    detector = PottyEventDetector(camera_config(dwell_trigger_s=5.0))

    emitted = run_sequence(
        detector,
        [[standing()]] * 4 + [[standing(120)]] + [[standing(120)]] * 4,
    )

    assert emitted == []


def test_two_separate_dwell_events() -> None:
    # A dwell event, then the dog leaves (tracks clear) and a second dog holds still:
    # the detector resets cleanly and emits a second independent dwell event.
    detector = PottyEventDetector(camera_config(dwell_trigger_s=5.0))

    emitted = run_sequence(
        detector,
        [[standing()] for _ in range(8)]
        + [[], [], []]
        + [[standing()] for _ in range(8)],
    )

    assert len(emitted) == 2
    assert all(c.near_miss is False for c in emitted)
    assert all(c.posture_summary["dwell_duration_s"] >= 5.0 for c in emitted)
