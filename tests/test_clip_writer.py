from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from detectivepotty.recording.clip_writer import write_frames_to_mp4
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_frame(frame_idx: int, *, width: int = 64, height: int = 48) -> Frame:
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = frame_idx * 20
    bgr[:, :, 1] = 120
    return Frame(
        bgr=bgr,
        frame_idx=frame_idx,
        mono_ts=float(frame_idx) / 5.0,
        wall_ts=BASE_TS + timedelta(milliseconds=200 * frame_idx),
        source_id="camera://cam-1",
    )


def test_write_frames_to_mp4_writes_readable_clip(tmp_path) -> None:
    frames = [make_frame(idx) for idx in range(5)]

    clip_path = write_frames_to_mp4(frames, tmp_path / "clip.mp4")

    assert clip_path.exists()
    capture = cv2.VideoCapture(str(clip_path))
    try:
        assert capture.isOpened()
        read_count = 0
        first_shape = None
        while True:
            ok, image = capture.read()
            if not ok:
                break
            first_shape = first_shape or image.shape
            read_count += 1
    finally:
        capture.release()

    assert abs(read_count - len(frames)) <= 1
    assert first_shape[:2] == (48, 64)


def test_write_frames_to_mp4_rejects_empty_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="frames must not be empty"):
        write_frames_to_mp4([], tmp_path / "clip.mp4")
