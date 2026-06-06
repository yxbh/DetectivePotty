from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from detectivepotty.events import Detection, Track
from detectivepotty.geometry import BBox
from detectivepotty.recording.dataset import (
    DEFAULT_CROP_MARGIN_FRAC,
    event_dir,
    format_event_timestamp,
    sanitize_path_component,
    write_event_images,
)
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 6, 6, 9, 10, 47, tzinfo=timezone.utc)


def make_frame(frame_idx: int) -> Frame:
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[10:30, 20:40] = (20, 120, 220)
    return Frame(
        bgr=bgr,
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
        source_id="rtsp://user:pass@cam.local/stream?token=secret&keep=1",
    )


def make_detection(frame_idx: int, confidence: float = 0.8) -> Detection:
    return Detection(
        bbox=BBox(18, 8, 42, 32),
        confidence=confidence,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
    )


def test_event_dir_is_sortable_secret_free_and_idempotent(tmp_path) -> None:
    start = datetime(
        2026,
        6,
        6,
        19,
        10,
        47,
        tzinfo=timezone(timedelta(hours=10)),
    )

    first = event_dir(
        tmp_path / "dataset",
        "cam/1",
        "Backyard Grass",
        start,
        "dog 1/primary",
        "123e4567-e89b-12d3-a456-426614174000",
    )
    second = event_dir(
        tmp_path / "dataset",
        "cam/1",
        "Backyard Grass",
        start,
        "dog 1/primary",
        "123e4567-e89b-12d3-a456-426614174000",
    )

    assert first == second
    assert first.name.startswith("20260606T091047Z_Backyard_Grass_dog_1_primary_")
    assert first.parts[-4:] == (
        "Backyard_Grass",
        "2026-06-06",
        "events",
        first.name,
    )
    assert " " not in first.name
    assert ":" not in first.name

    safe_source = sanitize_path_component(
        "rtsp://user:pass@cam.local/stream?token=secret&keep=1",
    )
    assert "user" not in safe_source
    assert "pass" not in safe_source
    assert "token" not in safe_source
    assert "secret" not in safe_source


def test_format_event_timestamp_is_utc_and_colon_free() -> None:
    local_dt = datetime(
        2026,
        6,
        6,
        19,
        10,
        47,
        tzinfo=timezone(timedelta(hours=10)),
    )

    formatted = format_event_timestamp(local_dt)

    assert formatted == "20260606T091047Z"
    assert ":" not in formatted


def test_write_event_images_uses_zero_padded_names_and_relative_crops(tmp_path) -> None:
    frames = [make_frame(idx) for idx in range(3)]
    detections = [make_detection(0), make_detection(2, confidence=0.9)]
    track = Track(track_id="dog-1", detections=detections)
    target = tmp_path / "event"

    frame_records, crop_records = write_event_images(
        target,
        frames,
        detections,
        [track],
        "dog-1",
        substream="high",
    )

    assert [path.name for path in sorted((target / "frames").iterdir())] == [
        "000.jpg",
        "001.jpg",
        "002.jpg",
    ]
    assert [path.name for path in sorted((target / "crops").iterdir())] == [
        "000.jpg",
        "001.jpg",
    ]
    assert len(frame_records) == 3
    assert frame_records[0].substream == "high"
    assert "token" not in frame_records[0].source_id
    assert [record.path for record in crop_records] == ["crops/000.jpg", "crops/001.jpg"]
    assert all(record.margin_frac == DEFAULT_CROP_MARGIN_FRAC for record in crop_records)
    assert all(record.path is not None and not record.path.startswith("/") for record in crop_records)
