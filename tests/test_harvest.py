from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.harvest import (
    DogSpan,
    FrameSample,
    compute_spans,
    harvest_clips,
    make_span_id,
)


def _sample(frame_idx: int, fps: float, box: tuple[float, float, float, float]) -> FrameSample:
    return FrameSample(
        frame_idx=frame_idx,
        time_s=frame_idx / fps,
        bbox=BBox(*box),
        confidence=0.9,
    )


def test_compute_spans_groups_contiguous_presence() -> None:
    fps = 10.0
    samples = [_sample(i, fps, (10, 10, 50, 50)) for i in range(0, 30, 5)]
    spans = compute_spans(
        {"1": samples},
        fps=fps,
        total_frames=100,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.track_id == "1"
    assert span.start_frame == 0
    assert span.end_frame == 25


def test_compute_spans_splits_on_gap_exceeding_tolerance() -> None:
    fps = 10.0
    # Two clusters separated by a 3s gap (> 2s tolerance).
    first = [_sample(i, fps, (10, 10, 50, 50)) for i in (0, 5, 10)]
    second = [_sample(i, fps, (10, 10, 50, 50)) for i in (40, 45, 50)]
    spans = compute_spans(
        {"1": first + second},
        fps=fps,
        total_frames=100,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
    )
    assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 10), (40, 50)]


def test_compute_spans_pads_and_clamps_to_bounds() -> None:
    fps = 10.0
    samples = [_sample(i, fps, (10, 10, 50, 50)) for i in (2, 4)]
    spans = compute_spans(
        {"1": samples},
        fps=fps,
        total_frames=60,
        merge_gap_s=2.0,
        pad_s=1.0,
        min_len_s=0.0,
        max_len_s=60.0,
    )
    span = spans[0]
    # 0.2s - 1s padding clamps to 0; 0.4s + 1s = 1.4s -> frame 14.
    assert span.start_frame == 0
    assert span.start_s == pytest.approx(0.0)
    assert span.end_frame == 14


def test_compute_spans_drops_below_min_len() -> None:
    fps = 10.0
    spans = compute_spans(
        {"1": [_sample(5, fps, (10, 10, 50, 50))]},
        fps=fps,
        total_frames=100,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=1.0,
        max_len_s=60.0,
    )
    assert spans == []


def test_compute_spans_splits_on_max_len() -> None:
    fps = 10.0
    samples = [_sample(i, fps, (10, 10, 50, 50)) for i in range(0, 260, 5)]
    spans = compute_spans(
        {"1": samples},
        fps=fps,
        total_frames=300,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=10.0,
    )
    # 0..25.5s of presence split into <=10s chunks.
    assert len(spans) >= 2
    for span in spans:
        assert span.end_s - span.start_s <= 10.0 + 1e-6


def test_compute_spans_tracks_are_independent_and_overlap() -> None:
    fps = 10.0
    a = [_sample(i, fps, (10, 10, 50, 50)) for i in (0, 5, 10)]
    b = [_sample(i, fps, (200, 200, 260, 260)) for i in (5, 10, 15)]
    spans = compute_spans(
        {"1": a, "2": b},
        fps=fps,
        total_frames=100,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
    )
    by_track = {s.track_id: s for s in spans}
    assert set(by_track) == {"1", "2"}
    assert (by_track["1"].start_frame, by_track["1"].end_frame) == (0, 10)
    assert (by_track["2"].start_frame, by_track["2"].end_frame) == (5, 15)


def test_make_span_id_is_deterministic() -> None:
    span = DogSpan("1", 0, 25, 0.0, 2.5)
    assert make_span_id("clip.mp4", span) == make_span_id("clip.mp4", span)
    other = DogSpan("2", 0, 25, 0.0, 2.5)
    assert make_span_id("clip.mp4", span) != make_span_id("clip.mp4", other)


# --- orchestrator with fakes ------------------------------------------------


class FakeCapture:
    """Yields ``n_frames`` solid frames, reporting ``fps`` and a fixed size."""

    def __init__(self, n_frames: int, *, fps: float = 10.0, size=(64, 48)) -> None:
        self.n_frames = n_frames
        self.fps = fps
        self.width, self.height = size
        self._remaining = deque(range(n_frames))
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._remaining:
            idx = self._remaining.popleft()
            frame = np.full((self.height, self.width, 3), idx % 255, dtype=np.uint8)
            return True, frame
        return False, None

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self) -> None:
        self.released = True


class FakeDetector:
    """Returns a dog box for frames in ``[present_start, present_end]``."""

    def __init__(self, present_start: int, present_end: int) -> None:
        self.present_start = present_start
        self.present_end = present_end

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        if not (self.present_start <= frame_idx <= self.present_end):
            return []
        return [
            Detection(
                bbox=BBox(10, 10, 30, 30),
                confidence=0.9,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=0.0,
                wall_ts=datetime.now(timezone.utc),
            )
        ]


class FakeClipWriter:
    written: dict[str, int] = {}

    def __init__(self, path: Path, fps: float, size) -> None:
        self.path = path
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")  # so checksum + existence checks pass

    def write(self, frame: np.ndarray) -> None:
        self.count += 1

    def release(self) -> None:
        FakeClipWriter.written[self.path.parent.name] = self.count


def test_harvest_clips_writes_span_dir_and_metadata(tmp_path: Path) -> None:
    FakeClipWriter.written = {}
    out_dir = tmp_path / "harvest"

    def capture_factory(_path: str) -> Any:
        return FakeCapture(60, fps=10.0)

    results = harvest_clips(
        tmp_path / "fake.mp4",
        out_dir,
        detector=FakeDetector(present_start=10, present_end=30),
        sample_every=5,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
        source_start_utc=datetime(2026, 6, 6, 9, 0, tzinfo=timezone.utc),
        capture_factory=capture_factory,
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )

    assert len(results) == 1
    result = results[0]
    assert result.clip_path.exists()
    assert result.metadata_path.exists()

    meta = json.loads(result.metadata_path.read_text())
    assert meta["fps"] == 10.0
    assert meta["track_id"] == result.span.track_id
    assert meta["source_start_frame"] == result.span.start_frame
    assert meta["checksum"]  # computed from the (fake) clip bytes
    # Detections are keyed by clip-frame index (source - start_frame).
    assert meta["detections"]
    first = meta["detections"][0]
    assert first["clip_frame_idx"] == first["source_frame_idx"] - result.span.start_frame
    # The clip writer received one frame per source frame in the span.
    assert FakeClipWriter.written[result.span_id] == result.span.frame_count


def test_harvest_clips_no_dogs_returns_empty(tmp_path: Path) -> None:
    def capture_factory(_path: str) -> Any:
        return FakeCapture(30, fps=10.0)

    results = harvest_clips(
        tmp_path / "fake.mp4",
        tmp_path / "harvest",
        detector=FakeDetector(present_start=100, present_end=200),
        sample_every=5,
        capture_factory=capture_factory,
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )
    assert results == []


def test_harvest_clips_is_idempotent(tmp_path: Path) -> None:
    out_dir = tmp_path / "harvest"

    def run() -> list[str]:
        return [
            r.span_id
            for r in harvest_clips(
                tmp_path / "fake.mp4",
                out_dir,
                detector=FakeDetector(present_start=10, present_end=30),
                sample_every=5,
                pad_s=0.0,
                min_len_s=0.0,
                source_start_utc=datetime(2026, 6, 6, tzinfo=timezone.utc),
                capture_factory=lambda _p: FakeCapture(60, fps=10.0),
                clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
            )
        ]

    first = run()
    second = run()
    assert first == second
    assert len(first) == 1
