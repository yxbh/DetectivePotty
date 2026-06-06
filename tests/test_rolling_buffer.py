from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading

import numpy as np
import pytest

from detectivepotty.sources.base import Frame
from detectivepotty.sources.rolling_buffer import RollingBuffer

BASE_TS = datetime(2026, 6, 6, tzinfo=timezone.utc)


def make_frame(idx: int, seconds: float) -> Frame:
    return Frame(
        bgr=np.full((1, 2, 3), idx, dtype=np.uint8),
        frame_idx=idx,
        mono_ts=1000.0 + seconds,
        wall_ts=BASE_TS + timedelta(seconds=seconds),
        source_id="camera-1",
    )


def frame_indices(frames: list[Frame]) -> list[int]:
    return [frame.frame_idx for frame in frames]


def test_rolling_buffer_evicts_by_duration() -> None:
    buffer = RollingBuffer(window_s=2.0)

    for seconds in [0.0, 1.0, 3.0]:
        buffer.append(make_frame(int(seconds), seconds))

    assert len(buffer) == 2
    assert frame_indices(buffer.snapshot()) == [1, 3]


def test_rolling_buffer_get_last_uses_latest_monotonic_timestamp() -> None:
    buffer = RollingBuffer(window_s=10.0)
    for seconds in [0.0, 1.0, 2.0, 4.0]:
        buffer.append(make_frame(int(seconds), seconds))

    assert frame_indices(buffer.get_last(2.0)) == [2, 4]


def test_rolling_buffer_get_window_uses_tz_aware_wall_timestamps() -> None:
    buffer = RollingBuffer(window_s=10.0)
    for seconds in [0.0, 1.0, 2.0, 3.0, 4.0]:
        buffer.append(make_frame(int(seconds), seconds))

    local_tz = timezone(timedelta(hours=10))
    start = (BASE_TS + timedelta(seconds=1)).astimezone(local_tz)
    end = (BASE_TS + timedelta(seconds=3)).astimezone(local_tz)

    assert frame_indices(buffer.get_window(start, end)) == [1, 2, 3]
    assert buffer.get_window(end, start) == []
    with pytest.raises(ValueError, match="timezone-aware"):
        buffer.get_window(datetime(2026, 6, 6), end)


def test_rolling_buffer_honors_max_frames_cap() -> None:
    buffer = RollingBuffer(window_s=100.0, max_frames=3)
    for idx in range(5):
        buffer.append(make_frame(idx, float(idx)))

    assert frame_indices(buffer.snapshot()) == [2, 3, 4]


def test_rolling_buffer_clear_and_thread_safety_smoke() -> None:
    buffer = RollingBuffer(window_s=1000.0, max_frames=1000)

    def append_many(offset: int) -> None:
        for idx in range(50):
            frame_idx = offset + idx
            buffer.append(make_frame(frame_idx, float(frame_idx)))

    threads = [threading.Thread(target=append_many, args=(offset,)) for offset in range(0, 200, 50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(buffer) == 200
    assert len(buffer.get_last(1000.0)) == 200

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.snapshot() == []
