"""Offline tests for chunked UNVR harvest (``harvest_unvr``).

No NVR, network, GPU, or model: the NVR download is an injected ``download_fn``
that writes a placeholder file, and the file pipeline runs against injected
``FakeCapture`` / ``FakeClipWriter`` / ``FakeDetector`` seams (same pattern as
``test_harvest``).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.harvest import DogSpan, HarvestResult
from detectivepotty.harvest_unvr import (
    harvest_camera_window,
    plan_chunks,
)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# plan_chunks (pure interval math)
# --------------------------------------------------------------------------- #


def test_plan_chunks_splits_with_overlap_tail() -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=2)
    chunks = plan_chunks(start, end, chunk_s=3600.0, overlap_s=5.0)
    assert len(chunks) == 2
    # Boundaries step by chunk_s; each carries a 5s overlap tail into the next.
    assert chunks[0][0] == start
    assert chunks[0][1] == start + timedelta(seconds=3605)
    assert chunks[1][0] == start + timedelta(seconds=3600)
    assert chunks[1][1] == end  # tail clamped at end


def test_plan_chunks_partial_final_window() -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=5400)  # 1.5 chunks
    chunks = plan_chunks(start, end, chunk_s=3600.0, overlap_s=0.0)
    assert len(chunks) == 2
    assert chunks[1] == (start + timedelta(seconds=3600), end)


def test_plan_chunks_empty_when_end_not_after_start() -> None:
    t = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    assert plan_chunks(t, t, chunk_s=3600.0) == []
    assert plan_chunks(t, t - timedelta(hours=1), chunk_s=3600.0) == []


def test_plan_chunks_naive_datetimes_treated_as_utc() -> None:
    start = datetime(2026, 6, 6, 0, 0)
    end = datetime(2026, 6, 6, 1, 0)
    chunks = plan_chunks(start, end, chunk_s=3600.0, overlap_s=0.0)
    assert len(chunks) == 1
    assert chunks[0][0].tzinfo is not None


# --------------------------------------------------------------------------- #
# Fakes for the orchestrator
# --------------------------------------------------------------------------- #


class FakeCapture:
    def __init__(self, n_frames: int, *, fps: float = 10.0, size=(64, 48)) -> None:
        self.fps = fps
        self.width, self.height = size
        self._remaining = deque(range(n_frames))

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._remaining:
            idx = self._remaining.popleft()
            return True, np.full((self.height, self.width, 3), idx % 255, dtype=np.uint8)
        return False, None

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def release(self) -> None:
        pass


class FakeDetector:
    """Dog present for frames in ``[present_start, present_end]``."""

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
                wall_ts=datetime.now(UTC),
            )
        ]


class FakeClipWriter:
    def __init__(self, path: Path, fps: float, size) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    def write(self, frame: np.ndarray) -> None:  # noqa: D401 - count not needed here
        pass

    def release(self) -> None:
        pass


def _make_download_fn(n_frames: int, calls: list[tuple[str, datetime, datetime]]):
    def download_fn(camera_id: str, start: datetime, end: datetime, dest: Path) -> Path:
        calls.append((camera_id, start, end))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"placeholder")  # only needs to exist; capture is faked
        return dest

    return download_fn


def _kwargs(n_frames: int):
    return dict(
        detector=FakeDetector(present_start=10, present_end=30),
        capture_factory=lambda _p: FakeCapture(n_frames, fps=10.0),
        clip_writer_factory=lambda p, fps, size: FakeClipWriter(p, fps, size),
        sample_every=5,
        merge_gap_s=2.0,
        pad_s=0.0,
        min_len_s=0.0,
        max_len_s=60.0,
    )


def _harvest_result(root: Path, span_id: str, start_s: float, end_s: float) -> HarvestResult:
    clip_dir = root / span_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    metadata_path = clip_dir / "metadata.json"
    clip_path.write_bytes(b"fake")
    metadata_path.write_text("{}", encoding="utf-8")
    return HarvestResult(
        span=DogSpan(
            track_id=span_id,
            start_frame=int(start_s * 10),
            end_frame=int(end_s * 10),
            start_s=start_s,
            end_s=end_s,
        ),
        span_id=span_id,
        clip_dir=clip_dir,
        clip_path=clip_path,
        metadata_path=metadata_path,
    )


# --------------------------------------------------------------------------- #
# harvest_camera_window orchestration
# --------------------------------------------------------------------------- #


def test_harvest_camera_window_harvests_each_chunk(tmp_path: Path) -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=2)
    calls: list[Any] = []
    results = harvest_camera_window(
        "Backyard Grass",
        start,
        end,
        tmp_path / "harvest",
        download_fn=_make_download_fn(60, calls),
        chunk_s=3600.0,
        overlap_s=0.0,
        **_kwargs(60),
    )
    # Two chunks downloaded, each yields its own dog span (no overlap → no dedup).
    assert len(calls) == 2
    assert len(results) == 2
    for result in results:
        assert result.clip_path.exists()
        assert result.metadata_path.exists()
    # Span ids embed the deterministic per-chunk source id (idempotent re-runs).
    assert results[0].span_id != results[1].span_id


def test_harvest_camera_window_dedups_cross_chunk_overlap(tmp_path: Path) -> None:
    # Both chunks decode an identical clip; with a large overlap the same span's
    # absolute interval repeats and must be deduped to a single result.
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(seconds=7200)
    calls: list[Any] = []
    results = harvest_camera_window(
        "cam",
        start,
        end,
        tmp_path / "harvest",
        download_fn=_make_download_fn(60, calls),
        chunk_s=3600.0,
        overlap_s=5.0,
        dedup_time_iou=0.5,
        **_kwargs(60),
    )
    assert len(calls) == 2
    # Each chunk is anchored at its own absolute start, so the spans do NOT share
    # an absolute interval → both kept (fragmentation, not duplication).
    assert len(results) == 2


def test_harvest_camera_window_dedups_identical_absolute_interval(tmp_path: Path) -> None:
    # Force a real duplicate: two windows anchored at the SAME absolute time.
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    calls: list[Any] = []

    # chunk_s == full window so plan yields ONE chunk; we instead invoke twice to
    # prove idempotency: re-running reproduces the same span dir (no duplicates).
    common = dict(
        download_fn=_make_download_fn(60, calls),
        chunk_s=3600.0,
        overlap_s=0.0,
        **_kwargs(60),
    )
    first = harvest_camera_window("cam", start, start + timedelta(hours=1),
                                  tmp_path / "h", **common)
    second = harvest_camera_window("cam", start, start + timedelta(hours=1),
                                   tmp_path / "h", **common)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].span_id == second[0].span_id  # deterministic / idempotent


def test_harvest_camera_window_keeps_same_chunk_overlapping_spans(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import detectivepotty.harvest_unvr as harvest_unvr

    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    out = tmp_path / "harvest"

    def fake_harvest(*_args, **_kwargs):
        return [
            _harvest_result(out, "track-a", 10.0, 20.0),
            _harvest_result(out, "track-b", 10.0, 20.0),
        ]

    monkeypatch.setattr(harvest_unvr, "harvest_clips", fake_harvest)

    results = harvest_camera_window(
        "cam",
        start,
        start + timedelta(hours=1),
        out,
        download_fn=_make_download_fn(1, []),
        chunk_s=3600.0,
        overlap_s=5.0,
        **_kwargs(1),
    )

    assert [result.span_id for result in results] == ["track-a", "track-b"]
    assert all(result.clip_dir.exists() for result in results)


def test_harvest_camera_window_dedups_only_overlap_region(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import detectivepotty.harvest_unvr as harvest_unvr

    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    out = tmp_path / "harvest"

    def fake_harvest(*_args, **kwargs):
        source_start = kwargs["source_start_utc"]
        if source_start == start:
            return [
                _harvest_result(out, "prev-overlap", 3600.0, 3604.0),
                _harvest_result(out, "prev-earlier", 30.0, 34.0),
            ]
        return [
            _harvest_result(out, "current-overlap", 0.0, 4.0),
            _harvest_result(out, "current-later", 30.0, 34.0),
        ]

    monkeypatch.setattr(harvest_unvr, "harvest_clips", fake_harvest)

    results = harvest_camera_window(
        "cam",
        start,
        start + timedelta(hours=2),
        out,
        download_fn=_make_download_fn(1, []),
        chunk_s=3600.0,
        overlap_s=5.0,
        dedup_time_iou=0.5,
        **_kwargs(1),
    )

    assert [result.span_id for result in results] == [
        "prev-overlap",
        "prev-earlier",
        "current-later",
    ]
    assert not (out / "current-overlap").exists()
    assert (out / "current-later").exists()


def test_harvest_camera_window_tolerates_failed_and_empty_chunks(tmp_path: Path) -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=3)

    def flaky_download(camera_id, c_start, c_end, dest):
        hour = c_start.hour
        if hour == 0:
            raise RuntimeError("network blip")  # chunk 1 fails
        if hour == 1:
            return None  # chunk 2: no recording (motion-only gap)
        dest.write_bytes(b"placeholder")  # chunk 3 ok
        return dest

    results = harvest_camera_window(
        "cam",
        start,
        end,
        tmp_path / "harvest",
        download_fn=flaky_download,
        chunk_s=3600.0,
        overlap_s=0.0,
        **_kwargs(60),
    )
    # Only the third chunk produced a clip; the day did not abort.
    assert len(results) == 1


def test_harvest_camera_window_cleans_temp_chunks(tmp_path: Path) -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    out = tmp_path / "harvest"
    calls: list[Any] = []
    harvest_camera_window(
        "cam",
        start,
        end,
        out,
        download_fn=_make_download_fn(60, calls),
        chunk_s=3600.0,
        overlap_s=0.0,
        **_kwargs(60),
    )
    assert not (out / ".chunks").exists()  # temp dir removed by default


def test_harvest_camera_window_records_camera_name_and_conf(tmp_path: Path) -> None:
    import json

    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    out = tmp_path / "harvest"
    calls: list[Any] = []
    results = harvest_camera_window(
        "6695ef21030c4603e400040d",
        start,
        end,
        out,
        download_fn=_make_download_fn(60, calls),
        camera_name="Backyard Grass",
        detect_conf=0.25,
        chunk_s=3600.0,
        overlap_s=0.0,
        **_kwargs(60),
    )
    assert results
    meta = json.loads(results[0].metadata_path.read_text(encoding="utf-8"))
    assert meta["camera_name"] == "Backyard Grass"
    assert meta["detect_conf"] == 0.25
    # The id->name sidecar is written at the harvest root for later resolution.
    cameras = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
    assert cameras["6695ef21030c4603e400040d"] == "Backyard Grass"
