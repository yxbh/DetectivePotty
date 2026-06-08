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
        "squat_threshold": 0.3,
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


def squat(x: float = 40.0) -> BBox:
    return BBox(x - 15, 35, x + 55, 85)


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


def test_stationary_squat_emits_one_potty_candidate() -> None:
    detector = PottyEventDetector(camera_config())

    emitted = run_sequence(
        detector,
        [[standing()], [standing()], [squat()], [squat()], [squat()], [squat()]],
    )

    assert len(emitted) == 1
    candidate = emitted[0]
    assert candidate.camera_id == "cam-1"
    assert candidate.primary_track_id == "1"
    assert candidate.start_ts == BASE_TS
    assert candidate.end_ts == BASE_TS + timedelta(seconds=4)
    assert candidate.near_miss is False
    assert candidate.stationary_duration_s >= 1.0
    assert candidate.squat_metric >= 0.3
    assert [track.track_id for track in candidate.tracks] == ["1"]
    assert len(candidate.detections) == 5


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
        [
            [standing(25), standing(95)],
            [standing(25), standing(95)],
            [squat(25), squat(95)],
            [squat(25), squat(95)],
            [squat(25), squat(95)],
        ],
    )

    assert len(emitted) == 1
    assert emitted[0].multi_dog is True
    assert emitted[0].ambiguous is True
    assert {track.track_id for track in emitted[0].tracks} == {"1", "2"}


def test_roi_and_ignore_zone_filtering() -> None:
    # Zones are normalized [0.0, 1.0] image coordinates; the test frame is 160x120,
    # so a detection's pixel center is normalized before the polygon test. ROI keeps
    # the left ~3/4 of the frame; the ignore box carves out the middle.
    roi = ZoneConfig(points=[(0.0, 0.0), (0.75, 0.0), (0.75, 1.0), (0.0, 1.0)])
    ignore = ZoneConfig(points=[(0.2, 0.1), (0.6, 0.1), (0.6, 0.9), (0.2, 0.9)])
    config = camera_config(roi=[roi], ignore_zones=[ignore])

    # Center (60, 60) -> (0.375, 0.5): inside ROI but inside the ignore box.
    ignored_detector = PottyEventDetector(config)
    ignored = run_sequence(
        ignored_detector,
        [[standing(40)], [standing(40)], [squat(40)], [squat(40)], [squat(40)]],
    )
    assert ignored == []

    # Center (150, 60) -> (0.9375, 0.5): outside the ROI.
    outside_detector = PottyEventDetector(config)
    outside = run_sequence(
        outside_detector,
        [[standing(130)], [standing(130)], [squat(130)], [squat(130)], [squat(130)]],
    )
    assert outside == []

    # Center (20, 60) -> (0.125, 0.5): inside ROI, left of the ignore box.
    allowed_detector = PottyEventDetector(config)
    allowed = run_sequence(
        allowed_detector,
        [[standing(0)], [standing(0)], [squat(0)], [squat(0)], [squat(0)]],
    )
    assert len(allowed) == 1
    assert allowed[0].camera_id == "cam-1"


def test_stationary_window_emits_with_non_integer_fps_timestamps() -> None:
    """Regression: real cameras report non-integer fps (e.g. 30.00004), so a
    stationary window spanning ``stationary_threshold_s`` lands a few microseconds
    below the threshold (1.9999972677...s for a 2.0s window). A strict ``>=``
    comparison rejected every frame and no event ever fired. The detector must
    tolerate that sub-threshold rounding and still emit one event.
    """

    fps = 30.000040983662547  # measured fps of the real sample clip
    sample_every = 3  # 30fps source sampled at ~10fps
    base = 12345.678  # non-zero base mono clock, like time.monotonic()
    threshold_s = 2.0
    config = camera_config(
        stationary_threshold_s=threshold_s,
        squat_threshold=0.3,
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

    # A few standing frames establish the tall baseline, then ~3s of in-place
    # squatting so the 2.0s stationary window is fully inside the squat phase.
    boxes = [standing()] * 3 + [squat()] * 35

    emitted: list = []
    for sample_idx, bbox in enumerate(boxes):
        the_frame, the_detection = at(sample_idx, bbox)
        emitted.extend(detector.process(the_frame, [the_detection]))
    emitted.extend(detector.flush())

    assert len(emitted) == 1
    assert emitted[0].squat_metric >= config.squat_threshold
    # The measured duration is genuinely just below the threshold; the detector
    # emits anyway thanks to the sampling/float tolerance.
    assert emitted[0].stationary_duration_s < threshold_s
    assert abs(emitted[0].stationary_duration_s - threshold_s) < 1e-3


def test_stationary_emits_when_window_span_stays_below_threshold() -> None:
    """Regression for the night-clip false negatives: with intermittent detections
    the trailing ``stationary_threshold_s`` window only ever *spans* up to one
    sample interval less than the threshold (e.g. 1.8s for a 2.0s threshold), so a
    strict ``window_span >= threshold`` check never fired even though the dog stood
    in one spot and squatted for many seconds. The gate tolerates a bounded coverage
    shortfall, so a long-lived stationary, squatting track must still emit exactly
    one event.
    """

    dt = 0.3  # sample spacing that does not divide the 2.0s threshold evenly
    threshold_s = 2.0
    config = camera_config(
        stationary_threshold_s=threshold_s,
        squat_threshold=0.3,
        sample_rate_fps=10.0,
        event_duration_s=2.0,
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

    boxes = [standing()] * 3 + [squat()] * 25

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
    assert emitted[0].squat_metric >= config.squat_threshold


def test_brief_squat_after_long_walk_does_not_emit() -> None:
    """A long-lived track that walks across the frame and only briefly drops into a
    squat must NOT emit: the trailing window then still contains the moving samples
    (high centroid motion), so it is not stationary. Guards against the coverage
    tolerance over-crediting an old-but-moving track.
    """

    config = camera_config(
        stationary_threshold_s=2.0,
        squat_threshold=0.3,
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

    # ~3s walking across the frame (tall standing box moves every sample).
    squat_flags: list[bool] = []
    original_posture_stats = PottyEventDetector._posture_stats

    def spy(self, track, current_mono):  # type: ignore[no-untyped-def]
        stats = original_posture_stats(self, track, current_mono)
        squat_flags.append(stats.is_squat)
        return stats

    PottyEventDetector._posture_stats = spy  # type: ignore[assignment]
    try:
        for step in range(30):
            x = float(step) * 15.0
            frame_obj, det = make(idx, BBox(x, 20, x + 40, 100))
            emitted.extend(detector.process(frame_obj, [det]))
            idx += 1

        # Then only ~0.3s of in-place squatting (short box => genuine squat posture).
        squat_x = 30.0 * 15.0
        for _ in range(3):
            frame_obj, det = make(idx, BBox(squat_x - 15, 40, squat_x + 55, 90))
            emitted.extend(detector.process(frame_obj, [det]))
            idx += 1
        emitted.extend(detector.flush())
    finally:
        PottyEventDetector._posture_stats = original_posture_stats

    assert any(squat_flags), "the brief posture should register as a squat"
    assert emitted == []
