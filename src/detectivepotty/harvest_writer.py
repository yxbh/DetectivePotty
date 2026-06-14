"""Clip extraction and metadata writing for harvest spans."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from detectivepotty.events import _jsonify
from detectivepotty.harvest_spans import DogSpan, HarvestResult, make_span_id
from detectivepotty.timeline import (
    TIME_BASIS_CLIP_FRAMES,
    TIME_BASIS_CLIP_PTS,
    clip_frame_times,
)
from detectivepotty.video_encode import open_h264_writer

CLIP_NAME = "clip.mp4"
METADATA_NAME = "metadata.json"
SCHEMA_VERSION = "harvest-1.1"
PTS_SCHEMA_VERSION = "harvest-1.2"


class ClipWriter(Protocol):
    def write(self, frame: np.ndarray) -> Any: ...

    def release(self) -> Any: ...


def default_clip_writer_factory(
    path: Path, fps: float, size: tuple[int, int]
) -> ClipWriter:
    """Write immutable clips as browser-playable H.264 (see ``video_encode``)."""

    return open_h264_writer(path, fps, size)


def write_spans(
    spans: Sequence[DogSpan],
    *,
    input_path: Path,
    out_dir: Path,
    fps: float,
    source_id: str,
    source_start_utc: datetime,
    source_frame_times_s: Sequence[float] | None = None,
    sample_every: int,
    camera_name: str | None = None,
    detect_conf: float | None = None,
    model_name: str | None = None,
    capture_factory: Callable[[str], Any],
    clip_writer_factory: Callable[[Path, float, tuple[int, int]], ClipWriter],
) -> list[HarvestResult]:
    plans: list[tuple[DogSpan, str, Path]] = []
    for span in spans:
        span_id = make_span_id(source_id, span)
        clip_dir = out_dir / span_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        plans.append((span, span_id, clip_dir))

    sizes = extract_span_clips(
        input_path,
        plans,
        fps=fps,
        capture_factory=capture_factory,
        clip_writer_factory=clip_writer_factory,
    )

    results: list[HarvestResult] = []
    for span, span_id, clip_dir in plans:
        clip_path = clip_dir / CLIP_NAME
        size = sizes.get(span_id)
        if size is None:
            raise RuntimeError(f"failed to extract clip for span {span_id}")
        width, height = size
        metadata_path = write_clip_metadata(
            clip_dir,
            span=span,
            span_id=span_id,
            source_id=source_id,
            source_start_utc=source_start_utc,
            fps=fps,
            width=width,
            height=height,
            frame_times_s=clip_frame_times(
                source_frame_times_s,
                span.start_frame,
                span.end_frame,
            ),
            sample_every=sample_every,
            camera_name=camera_name,
            detect_conf=detect_conf,
            model_name=model_name,
        )
        results.append(
            HarvestResult(
                span=span,
                span_id=span_id,
                clip_dir=clip_dir,
                clip_path=clip_path,
                metadata_path=metadata_path,
            )
        )
    return results


def merge_frame_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge ``(start, end)`` frame ranges into disjoint, sorted segments.

    Overlapping *or adjacent* ranges fuse (``end + 1 >= next start``) so one
    seek+decode covers every span that shares those frames.
    """

    segments: list[list[int]] = []
    for start, end in sorted(ranges):
        if segments and start <= segments[-1][1] + 1:
            segments[-1][1] = max(segments[-1][1], end)
        else:
            segments.append([start, end])
    return [(s, e) for s, e in segments]


def seek_capture(capture: Any, frame_idx: int) -> bool:
    """Position ``capture`` so the next ``read()`` returns ``frame_idx``.

    Uses the ``cv2.CAP_PROP_POS_FRAMES`` setter (native on OpenCV, frame-accurate
    forward seek on :class:`PyAvCapture`). Returns ``False`` when the capture has
    no usable setter so callers can fall back to a sequential pass.
    """

    setter = getattr(capture, "set", None)
    if not callable(setter):
        return False
    try:
        return bool(setter(cv2.CAP_PROP_POS_FRAMES, frame_idx))
    except Exception:  # pragma: no cover - defensive
        return False


def capture_can_seek(capture: Any) -> bool:
    return callable(getattr(capture, "set", None))


def extract_span_clips(
    input_path: Path,
    plans: Sequence[tuple[DogSpan, str, Path]],
    *,
    fps: float,
    capture_factory: Callable[[str], Any],
    clip_writer_factory: Callable[[Path, float, tuple[int, int]], ClipWriter],
) -> dict[str, tuple[int, int]]:
    """Cut each span's clip by decoding **only** the dog-present windows.

    Spans cover a small fraction of a long recording, so instead of decoding the
    whole file we merge the span frame-ranges into disjoint decode-segments and
    seek to each one, decoding only ``[start, end]``. Overlapping spans (different
    tracks sharing frames) live in the same segment, so one decode still feeds all
    their writers. Writers are created lazily on the first frame so we know the
    real size, and each span's first written frame is exactly ``start_frame`` so
    the clip-frame / source-frame metadata mapping stays correct. Returns
    ``span_id -> (width, height)``.

    When the capture cannot seek (no ``CAP_PROP_POS_FRAMES`` support), falls back
    to a single sequential pass over the whole file.
    """

    capture = capture_factory(str(input_path))
    if not capture_opened(capture):
        release_capture(capture)
        raise RuntimeError(f"failed to open video file: {input_path}")

    segments = merge_frame_ranges([(span.start_frame, span.end_frame) for span, _, _ in plans])
    writers: dict[str, ClipWriter] = {}
    sizes: dict[str, tuple[int, int]] = {}

    def _emit(frame: np.ndarray, decoded_idx: int) -> None:
        height, width = frame.shape[:2]
        for span, span_id, clip_dir in plans:
            if span.start_frame <= decoded_idx <= span.end_frame:
                writer = writers.get(span_id)
                if writer is None:
                    writer = clip_writer_factory(clip_dir / CLIP_NAME, fps, (width, height))
                    writers[span_id] = writer
                    sizes[span_id] = (width, height)
                writer.write(frame)

    try:
        seekable = False
        if segments and capture_can_seek(capture):
            seekable = seek_capture(capture, segments[0][0])
            if not seekable:
                raise RuntimeError(
                    f"seek to frame {segments[0][0]} failed before extraction"
                )
        if seekable:
            for seg_index, (seg_start, seg_end) in enumerate(segments):
                if seg_index > 0 and not seek_capture(capture, seg_start):
                    raise RuntimeError(f"seek to frame {seg_start} failed mid-extraction")
                decoded_idx = seg_start
                while decoded_idx <= seg_end:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            "decode ended before expected frame "
                            f"{seg_end} in segment starting at {seg_start}"
                        )
                    _emit(frame, decoded_idx)
                    decoded_idx += 1
        else:
            # No seek support: one sequential pass writes each frame into every
            # span it falls inside (legacy behavior).
            decoded_idx = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                _emit(frame, decoded_idx)
                decoded_idx += 1
    finally:
        release_errors: list[Exception] = []
        try:
            for writer in writers.values():
                try:
                    writer.release()
                except Exception as exc:
                    release_errors.append(exc)
        finally:
            release_capture(capture)
        if release_errors:
            raise release_errors[0]
    return sizes


def write_clip_metadata(
    clip_dir: Path,
    *,
    span: DogSpan,
    span_id: str,
    source_id: str,
    source_path: Path | None = None,
    source_start_utc: datetime,
    fps: float,
    width: int,
    height: int,
    frame_times_s: Sequence[float] | None = None,
    sample_every: int,
    camera_name: str | None = None,
    detect_conf: float | None = None,
    model_name: str | None = None,
) -> Path:
    del source_path
    source_span_start_utc = source_start_utc + timedelta(seconds=span.start_s)
    source_span_end_utc = source_start_utc + timedelta(seconds=span.end_s)
    normalized_frame_times = (
        [round(float(value), 9) for value in frame_times_s]
        if frame_times_s is not None
        else None
    )

    detections: list[dict[str, Any]] = []
    for sample in sorted(span.samples, key=lambda s: s.frame_idx):
        detections.append(
            {
                "clip_frame_idx": sample.frame_idx - span.start_frame,
                "source_frame_idx": sample.frame_idx,
                "time_s": sample.time_s,
                "track_id": span.track_id,
                "bbox": _jsonify(sample.bbox),
                "confidence": sample.confidence,
                "class_name": sample.class_name,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": PTS_SCHEMA_VERSION
        if normalized_frame_times
        else SCHEMA_VERSION,
        "span_id": span_id,
        "source_id": source_id,
        "source_path": source_id,
        "camera_name": camera_name,
        "detect_conf": detect_conf,
        "model_name": model_name,
        "source_start_utc": source_start_utc.isoformat(),
        "source_span_start_utc": source_span_start_utc.isoformat(),
        "source_span_end_utc": source_span_end_utc.isoformat(),
        "fps": fps,
        "frame_count": span.frame_count,
        "width": width,
        "height": height,
        "timebase": TIME_BASIS_CLIP_PTS
        if normalized_frame_times
        else TIME_BASIS_CLIP_FRAMES,
        "sample_every": sample_every,
        "track_id": span.track_id,
        "source_start_frame": span.start_frame,
        "source_end_frame": span.end_frame,
        "start_s": span.start_s,
        "end_s": span.end_s,
        "detections": detections,
        "checksum": sha256_file(clip_dir / CLIP_NAME),
    }
    if normalized_frame_times:
        payload["frame_times_s"] = normalized_frame_times

    target = clip_dir / METADATA_NAME
    tmp = target.parent / f".metadata.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def capture_opened(capture: Any) -> bool:
    is_opened = getattr(capture, "isOpened", None)
    return bool(is_opened()) if callable(is_opened) else True


def release_capture(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()
