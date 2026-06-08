from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pytest

from detectivepotty.sources.file import (
    FileSource,
    derive_base_wall_ts,
    parse_filename_start_ts,
)
from detectivepotty.sources.rtsp import (
    _DEFAULT_FFMPEG_CAPTURE_OPTIONS,
    RTSPSource,
    _configure_ffmpeg_transport,
)

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


def test_configure_ffmpeg_transport_defaults_to_tcp() -> None:
    env: dict[str, str] = {}
    _configure_ffmpeg_transport(env)
    assert env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == _DEFAULT_FFMPEG_CAPTURE_OPTIONS
    assert "rtsp_transport;tcp" == _DEFAULT_FFMPEG_CAPTURE_OPTIONS


def test_configure_ffmpeg_transport_respects_existing_value() -> None:
    env = {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;udp"}
    _configure_ffmpeg_transport(env)
    assert env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;udp"


def test_rtsp_module_import_sets_ffmpeg_transport_default() -> None:
    assert os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == _DEFAULT_FFMPEG_CAPTURE_OPTIONS


UNIFI_NAME = (
    "Backyard Grass 6-6-2026, 09.10.47 GMT+10 - 6-6-2026, 09.11.03 GMT+10.mp4"
)


def test_parse_filename_start_ts_unifi_export() -> None:
    parsed = parse_filename_start_ts(UNIFI_NAME)
    # 09:10:47 at GMT+10 == 23:10:47 UTC on the previous day.
    assert parsed == datetime(2026, 6, 5, 23, 10, 47, tzinfo=timezone.utc)


def test_parse_filename_start_ts_rejects_impossible_dates() -> None:
    assert parse_filename_start_ts("clip 13-40-2026, 09.10.47 GMT+10.mp4") is None
    assert parse_filename_start_ts("just-a-clip.mp4") is None


def test_derive_base_wall_ts_prefers_filename(tmp_path: Path) -> None:
    path = tmp_path / UNIFI_NAME
    path.write_bytes(b"x")
    base, basis = derive_base_wall_ts(path)
    assert basis == "filename"
    assert base == datetime(2026, 6, 5, 23, 10, 47, tzinfo=timezone.utc)


def test_derive_base_wall_ts_falls_back_to_mtime(tmp_path: Path) -> None:
    path = tmp_path / "no_timestamp_here.mp4"
    path.write_bytes(b"x")
    fixed = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
    os.utime(path, (fixed, fixed))
    base, basis = derive_base_wall_ts(path)
    assert basis == "file_mtime"
    assert base == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_derive_base_wall_ts_runtime_now_for_missing_unparseable(tmp_path: Path) -> None:
    path = tmp_path / "missing_no_ts.mp4"  # never created
    before = datetime.now(timezone.utc)
    base, basis = derive_base_wall_ts(path)
    assert basis == "runtime_now"
    assert base >= before


def test_derive_base_wall_ts_override_wins(tmp_path: Path) -> None:
    path = tmp_path / UNIFI_NAME
    path.write_bytes(b"x")
    override = datetime(2030, 5, 5, 5, 5, 5, tzinfo=timezone.utc)
    base, basis = derive_base_wall_ts(path, override=override)
    assert basis == "config"
    assert base == override


def test_file_source_timeline_is_deterministic_across_opens(tmp_path: Path) -> None:
    path = tmp_path / UNIFI_NAME
    path.write_bytes(b"x")

    def read_three() -> list[datetime]:
        source = FileSource(path, capture_factory=lambda _p: FakeCapture([1, 2, 3]))
        source.open()
        try:
            frames = [source.read() for _ in range(3)]
        finally:
            source.close()
        assert all(frame is not None for frame in frames)
        assert source.time_basis == "filename"
        return [frame.wall_ts for frame in frames]  # type: ignore[union-attr]

    first = read_three()
    second = read_three()
    assert first == second
    # Anchor equals the real recording time parsed from the filename.
    assert first[0] == datetime(2026, 6, 5, 23, 10, 47, tzinfo=timezone.utc)


def test_file_source_accepts_explicit_base() -> None:
    base = datetime(2025, 3, 3, 12, 0, 0, tzinfo=timezone.utc)
    source = FileSource(
        "anything.mp4",
        base_wall_ts=base,
        capture_factory=lambda _p: FakeCapture([1, 2]),
    )
    source.open()
    try:
        frame = source.read()
    finally:
        source.close()
    assert frame is not None
    assert source.time_basis == "explicit"
    assert frame.wall_ts == base
