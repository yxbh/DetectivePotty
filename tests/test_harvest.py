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
    _extract_span_clips,
    _merge_frame_ranges,
    _scan_for_dogs,
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


class FakeBatchDetector(FakeDetector):
    """``FakeDetector`` that also exposes ``detect_batch`` (per-image-independent).

    Delegates each batch entry to the per-frame :meth:`detect`, so a batched scan
    must yield identical detections (and thus identical spans) to the single-frame
    path. Records the batch sizes it was called with for boundary assertions.
    """

    def __init__(self, present_start: int, present_end: int) -> None:
        super().__init__(present_start, present_end)
        self.batch_sizes: list[int] = []

    def detect_batch(self, frames, metas=None):
        frames = list(frames)
        if metas is None:
            metas = [None] * len(frames)
        self.batch_sizes.append(len(frames))
        out = []
        for frame, meta in zip(frames, metas):
            idx = 0 if meta is None else meta.frame_idx
            out.append(self.detect(frame, frame_idx=idx))
        return out


def _scan(detector, *, detect_batch_size, n_frames=60, sample_every=5, fps=10.0):
    return _scan_for_dogs(
        Path("fake.mp4"),
        detector=detector,
        sample_every=sample_every,
        iou_threshold=0.3,
        max_age_frames=15,
        center_dist_gate=1.5,
        detect_batch_size=detect_batch_size,
        capture_factory=lambda _p: FakeCapture(n_frames, fps=fps),
    )


@pytest.mark.parametrize("batch_size", [2, 4, 5, 32])
def test_scan_batched_matches_single_frame(batch_size: int) -> None:
    # A detector without detect_batch always takes the single-frame path; the
    # batched path must reproduce its (fps, total_frames, presence) exactly.
    single = _scan(FakeDetector(10, 30), detect_batch_size=1)
    batched = _scan(FakeBatchDetector(10, 30), detect_batch_size=batch_size)
    assert batched == single


def test_scan_batch_size_one_uses_single_frame_path() -> None:
    # detect_batch_size=1 disables batching even when detect_batch exists.
    det = FakeBatchDetector(10, 30)
    single = _scan(FakeDetector(10, 30), detect_batch_size=1)
    assert _scan(det, detect_batch_size=1) == single
    assert det.batch_sizes == []  # detect_batch never called


def test_scan_batch_flush_sizes() -> None:
    # 60 frames @ sample_every=5 -> 12 sampled frames.
    # Larger-than-count -> one partial flush; exact multiple -> even flushes;
    # remainder -> a short tail flush. All paths still match single-frame.
    single = _scan(FakeDetector(10, 30), detect_batch_size=1)

    big = FakeBatchDetector(10, 30)
    assert _scan(big, detect_batch_size=32) == single
    assert big.batch_sizes == [12]

    exact = FakeBatchDetector(10, 30)
    assert _scan(exact, detect_batch_size=4) == single
    assert exact.batch_sizes == [4, 4, 4]

    remainder = FakeBatchDetector(10, 30)
    assert _scan(remainder, detect_batch_size=5) == single
    assert remainder.batch_sizes == [5, 5, 2]


def test_harvest_clips_batched_matches_single_frame_spans(tmp_path: Path) -> None:
    def run(detector, *, detect_batch_size) -> list[tuple[str, int, int]]:
        results = harvest_clips(
            tmp_path / "fake.mp4",
            tmp_path / f"harvest_{detect_batch_size}",
            detector=detector,
            sample_every=5,
            pad_s=0.0,
            min_len_s=0.0,
            detect_batch_size=detect_batch_size,
            source_start_utc=datetime(2026, 6, 6, tzinfo=timezone.utc),
            capture_factory=lambda _p: FakeCapture(60, fps=10.0),
            clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
        )
        return [(r.span_id, r.span.start_frame, r.span.end_frame) for r in results]

    single = run(FakeDetector(10, 30), detect_batch_size=1)
    batched = run(FakeBatchDetector(10, 30), detect_batch_size=32)
    assert single == batched
    assert len(single) == 1


class _CountingCapture(FakeCapture):
    """``FakeCapture`` that counts ``read()`` calls that returned a frame."""

    def __init__(self, n_frames: int, *, fps: float = 10.0, size=(64, 48)) -> None:
        super().__init__(n_frames, fps=fps, size=size)
        self._n = n_frames
        self.read_count = 0

    def read(self):
        ok, frame = super().read()
        if ok:
            self.read_count += 1
        return ok, frame


class FakeSeekCapture(_CountingCapture):
    """Seekable fake: ``set(CAP_PROP_POS_FRAMES, n)`` repositions the reader."""

    def __init__(self, n_frames: int, *, fps: float = 10.0, size=(64, 48)) -> None:
        super().__init__(n_frames, fps=fps, size=size)
        self.seeks: list[int] = []

    def set(self, prop: int, value: float) -> bool:  # noqa: N802 - mirror cv2
        import cv2

        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        n = int(value)
        if not (0 <= n <= self._n):
            return False
        self.seeks.append(n)
        self._remaining = deque(range(n, self._n))
        return True


def _span(track_id: str, start: int, end: int, fps: float = 10.0) -> DogSpan:
    return DogSpan(
        track_id=track_id,
        start_frame=start,
        end_frame=end,
        start_s=start / fps,
        end_s=end / fps,
    )


def _plans(tmp_path: Path, spans):
    out: list[tuple[DogSpan, str, Path]] = []
    for i, span in enumerate(spans):
        span_id = f"s{i}"
        clip_dir = tmp_path / span_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        out.append((span, span_id, clip_dir))
    return out


def test_merge_frame_ranges() -> None:
    # disjoint stays separate; overlapping and adjacent (end+1 == next start) fuse.
    assert _merge_frame_ranges([(0, 5), (20, 25)]) == [(0, 5), (20, 25)]
    assert _merge_frame_ranges([(0, 10), (5, 15)]) == [(0, 15)]
    assert _merge_frame_ranges([(0, 5), (6, 10)]) == [(0, 10)]
    # unsorted input + a contained range.
    assert _merge_frame_ranges([(20, 25), (0, 5), (2, 3)]) == [(0, 5), (20, 25)]


def test_extract_decodes_only_union(tmp_path: Path) -> None:
    FakeClipWriter.written = {}
    spans = [_span("1", 10, 20), _span("2", 50, 60)]  # two disjoint segments
    plans = _plans(tmp_path, spans)
    cap = FakeSeekCapture(100)

    _extract_span_clips(
        tmp_path / "fake.mp4",
        plans,
        fps=10.0,
        capture_factory=lambda _p: cap,
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )

    # Only the union of the two ranges is decoded (11 + 11), not all 100 frames.
    assert cap.read_count == 22
    assert cap.seeks == [10, 50]
    assert FakeClipWriter.written["s0"] == 11  # frames 10..20 inclusive
    assert FakeClipWriter.written["s1"] == 11  # frames 50..60 inclusive


def test_extract_overlapping_spans_share_one_segment(tmp_path: Path) -> None:
    FakeClipWriter.written = {}
    spans = [_span("1", 10, 30), _span("2", 20, 40)]  # overlap -> one segment
    plans = _plans(tmp_path, spans)
    cap = FakeSeekCapture(100)

    _extract_span_clips(
        tmp_path / "fake.mp4",
        plans,
        fps=10.0,
        capture_factory=lambda _p: cap,
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )

    # Union [10..40] decoded once; each writer still gets only its own frames.
    assert cap.read_count == 31
    assert cap.seeks == [10]
    assert FakeClipWriter.written["s0"] == 21  # 10..30
    assert FakeClipWriter.written["s1"] == 21  # 20..40


def test_extract_sequential_fallback_without_seek(tmp_path: Path) -> None:
    # A capture without ``set`` decodes the whole file (legacy path) but still
    # writes exactly each span's frames.
    FakeClipWriter.written = {}
    spans = [_span("1", 10, 20)]
    plans = _plans(tmp_path, spans)
    cap = _CountingCapture(100)
    assert not hasattr(cap, "set")

    _extract_span_clips(
        tmp_path / "fake.mp4",
        plans,
        fps=10.0,
        capture_factory=lambda _p: cap,
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )

    assert cap.read_count == 100  # full sequential pass
    assert FakeClipWriter.written["s0"] == 11


def test_harvest_clips_seek_matches_sequential(tmp_path: Path) -> None:
    def run(capture_cls, sub: str):
        FakeClipWriter.written = {}
        results = harvest_clips(
            tmp_path / "fake.mp4",
            tmp_path / sub,
            detector=FakeDetector(present_start=10, present_end=30),
            sample_every=5,
            pad_s=0.0,
            min_len_s=0.0,
            source_start_utc=datetime(2026, 6, 6, tzinfo=timezone.utc),
            capture_factory=lambda _p: capture_cls(60, fps=10.0),
            clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
        )
        ids = [(r.span_id, r.span.start_frame, r.span.end_frame) for r in results]
        return ids, dict(FakeClipWriter.written)

    seek_ids, seek_counts = run(FakeSeekCapture, "seek")
    seq_ids, seq_counts = run(FakeCapture, "seq")
    assert seek_ids == seq_ids
    assert seek_counts == seq_counts


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

class AliasModelDetector:
    """Detector exposing ``model_name`` that emits dog + one alias class.

    Frames in ``[present_start, present_end]`` get a box; even source frames are
    the real ``dog`` class, odd ones come in as the accepted alias ``sheep`` (as
    the committed class-agnostic-NMS recovery would). Used to assert harvest now
    records detector provenance + per-detection class without shifting spans.
    """

    model_name = "models/yolo11m.pt"

    def __init__(self, present_start: int, present_end: int) -> None:
        self.present_start = present_start
        self.present_end = present_end

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        if not (self.present_start <= frame_idx <= self.present_end):
            return []
        class_name = "dog" if frame_idx % 2 == 0 else "sheep"
        return [
            Detection(
                bbox=BBox(10, 10, 30, 30),
                confidence=0.9,
                class_name=class_name,
                frame_idx=frame_idx,
                mono_ts=0.0,
                wall_ts=datetime.now(timezone.utc),
            )
        ]


def _harvest_one(detector: Any, tmp_path: Path) -> Any:
    FakeClipWriter.written = {}
    results = harvest_clips(
        tmp_path / "fake.mp4",
        tmp_path / "harvest",
        detector=detector,
        sample_every=5,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
        source_start_utc=datetime(2026, 6, 6, 9, 0, tzinfo=timezone.utc),
        capture_factory=lambda _p: FakeCapture(60, fps=10.0),
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
    )
    assert len(results) == 1
    return results[0]


def test_harvest_records_model_name_and_per_detection_class(tmp_path: Path) -> None:
    result = _harvest_one(AliasModelDetector(10, 30), tmp_path)
    meta = json.loads(result.metadata_path.read_text())
    assert meta["schema_version"] == "harvest-1.1"
    assert meta["model_name"] == "models/yolo11m.pt"
    classes = {d["class_name"] for d in meta["detections"]}
    assert classes == {"dog", "sheep"}  # alias preserved for audit


def test_harvest_model_name_none_when_detector_lacks_it(tmp_path: Path) -> None:
    # FakeDetector has no ``model_name`` attr -> recorded as None (legacy/fake).
    result = _harvest_one(FakeDetector(present_start=10, present_end=30), tmp_path)
    meta = json.loads(result.metadata_path.read_text())
    assert meta["model_name"] is None
    assert all(d["class_name"] == "dog" for d in meta["detections"])


def test_harvest_class_provenance_does_not_shift_spans(tmp_path: Path) -> None:
    """Extra-fields-only invariant: alias/model recording can't move span math."""
    dog_only = _harvest_one(FakeDetector(present_start=10, present_end=30), tmp_path / "a")
    aliased = _harvest_one(AliasModelDetector(10, 30), tmp_path / "b")
    assert aliased.span.start_frame == dog_only.span.start_frame
    assert aliased.span.end_frame == dog_only.span.end_frame
    assert aliased.span.frame_count == dog_only.span.frame_count
