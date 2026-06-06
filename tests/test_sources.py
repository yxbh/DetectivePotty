from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pytest

from detectivepotty.sources.file import FileSource
from detectivepotty.sources.rtsp import RTSPSource

SAMPLE_CLIP = Path(
    "data/unifi_direct_clip_downloads/"
    "Backyard Grass 6-6-2026, 09.10.47 GMT+10 - "
    "6-6-2026, 09.11.03 GMT+10.mp4"
)


def test_file_source_reads_sample_clip_and_eof() -> None:
    if not SAMPLE_CLIP.exists():
        pytest.skip("sample clip is not available")

    with FileSource(SAMPLE_CLIP) as source:
        assert source.is_live is False
        assert source.fps is not None and source.fps > 0
        assert source.resolution == (2688, 1512)

        frames = [source.read() for _ in range(3)]
        assert all(frame is not None for frame in frames)
        first, second, third = frames
        assert first is not None
        assert second is not None
        assert third is not None
        assert [first.frame_idx, second.frame_idx, third.frame_idx] == [0, 1, 2]
        assert first.width == 2688
        assert first.height == 1512
        assert first.wall_ts.tzinfo is not None
        assert first.wall_ts.utcoffset() is not None
        assert first.source_id == str(SAMPLE_CLIP)
        assert second.wall_ts > first.wall_ts

        for _ in range(1000):
            if source.read() is None:
                break
        else:
            pytest.fail("sample clip did not reach EOF within expected frame count")
        assert source.read() is None


class FakeCapture:
    def __init__(self, values: list[int], *, opened: bool = True) -> None:
        self.values = deque(values)
        self.opened = opened
        self.released = False
        self.read_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self.values:
            value = self.values.popleft()
            return True, np.full((2, 3, 3), value, dtype=np.uint8)
        return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 12.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 3.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 2.0
        return 0.0

    def release(self) -> None:
        self.released = True


def wait_until(predicate: Any, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("condition was not met before timeout")


def test_rtsp_source_latest_frame_only_and_sanitizes_source_id() -> None:
    capture = FakeCapture([1, 2, 3])
    raw_url = "rtsps://user:pass@host:7441/abc?token=secret&profile=low"
    source = RTSPSource(
        raw_url,
        capture_factory=lambda _url: capture,
        stale_timeout_s=0.2,
        reconnect_initial_s=0.2,
        reconnect_max_s=0.2,
    )

    try:
        source.open()
        wait_until(lambda: capture.read_count >= 3)
        frame = source.read()
    finally:
        source.close()

    assert frame is not None
    assert int(frame.bgr[0, 0, 0]) == 3
    assert frame.frame_idx == 2
    assert source.is_live is True
    assert source.fps == 12.0
    assert source.resolution == (3, 2)
    assert "user" not in source.source_id
    assert "pass" not in source.source_id
    assert "token" not in source.source_id
    assert source.source_id == "rtsps://host:7441/abc?profile=low"
    assert frame.source_id == source.source_id


def test_rtsp_source_reconnects_after_stale_failures() -> None:
    captures = [FakeCapture([]), FakeCapture([7])]
    created: list[FakeCapture] = []

    def capture_factory(_url: str) -> FakeCapture:
        capture = captures.pop(0) if captures else FakeCapture([7])
        created.append(capture)
        return capture

    source = RTSPSource(
        "rtsp://host/stream",
        capture_factory=capture_factory,
        stale_timeout_s=0.02,
        reconnect_initial_s=0.01,
        reconnect_max_s=0.01,
        read_retry_s=0.005,
    )

    try:
        source.open()
        wait_until(lambda: source._latest_frame is not None)  # noqa: SLF001
        frame = source.read()
    finally:
        source.close()

    assert frame is not None
    assert int(frame.bgr[0, 0, 0]) == 7
    assert len(created) >= 2
    assert created[0].released is True
