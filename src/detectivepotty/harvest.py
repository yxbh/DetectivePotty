"""Historical footage harvester: cut dog-present spans out of long clips.

This module is the data-engine entry point. Given a long recording (a local
file for now; UNVR time-range download is wired separately), it:

1. decodes the clip and samples every Nth frame,
2. runs a dog detector + greedy tracker over the samples,
3. groups each track's sampled detections into **track-aware spans** (merge with
   a time tolerance, pad, clamp, enforce min/max length), and
4. writes one immutable ``clip.mp4`` per span at the source resolution/fps plus a
   ``metadata.json`` describing it (checksum, fps, frame count, timebase, source
   UTC range, and the per-sampled-frame detection boxes keyed by the *harvested
   clip's* frame numbering so the exporter can bind a label back to the dog).

The span math (:func:`compute_spans`) is a pure function and is unit-tested in
isolation. The orchestrator (:func:`harvest_clips`) takes injectable
``capture_factory`` / ``detector`` / ``clip_writer_factory`` seams so the offline
test suite never opens a real camera, model, or codec.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from detectivepotty.detect.yolo import FrameMeta
from detectivepotty.events import Detection, _jsonify
from detectivepotty.harvest_spans import (
    DEFAULT_MAX_LEN_S,
    DEFAULT_MERGE_GAP_S,
    DEFAULT_MIN_LEN_S,
    DEFAULT_PAD_S,
    DogSpan,
    FrameSample,
    HarvestResult,
    compute_spans,
    make_span_id,
)
from detectivepotty.sources.base import sanitize_source_id
from detectivepotty.sources.file import derive_base_wall_ts
from detectivepotty.sources.prefetch import prefetch
from detectivepotty.sources.pyav_capture import open_capture
from detectivepotty.tracking import Tracker
from detectivepotty.video_encode import open_h264_writer

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_EVERY = 5
# Tracking defaults are deliberately looser than the live pipeline's: harvest
# samples every Nth frame, so a dog drifts further between tracker updates and
# the detector misses it more often. A larger max-age survives a few missed
# samples (DEFAULT_MAX_AGE_FRAMES ≈ 3 misses at the default stride) and the
# center-distance gate re-associates boxes that stopped overlapping, which keeps
# one walking dog as one track instead of fragmenting it into many short spans.
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_AGE_FRAMES = 15
DEFAULT_CENTER_DIST_GATE = 1.5
# Sampled frames are detected in one batched forward when the detector exposes
# ``detect_batch`` (real ``DogDetector``); the live pipeline proves CoreML/MPS
# true-batches here. Measured on the production CoreML export, batch 32 runs the
# scan's inference ~3.4x faster than single-frame ``detect`` (the dominant cost
# of a decode-overlapped scan). ``1`` reproduces the legacy single-frame path,
# as does any detector lacking ``detect_batch`` (e.g. test fakes).
DEFAULT_DETECT_BATCH_SIZE = 32

CLIP_NAME = "clip.mp4"
METADATA_NAME = "metadata.json"
SCHEMA_VERSION = "harvest-1.1"
TIME_BASIS_CLIP_FRAMES = "clip_frames"


class DetectorLike(Protocol):
    """Minimal detector surface the harvester needs (matches ``DogDetector``).

    The scan uses :meth:`detect_batch` when present (batched forward, much faster
    on accelerated backends) and otherwise falls back to per-frame :meth:`detect`,
    so a fake exposing only ``detect`` still works unchanged.
    """

    def detect(
        self, frame_bgr_original: np.ndarray, frame_idx: int = ...
    ) -> list[Detection]: ...


class ClipWriter(Protocol):
    def write(self, frame: np.ndarray) -> Any: ...

    def release(self) -> Any: ...


def _default_clip_writer_factory(
    path: Path, fps: float, size: tuple[int, int]
) -> ClipWriter:
    """Write immutable clips as browser-playable H.264 (see ``video_encode``)."""

    return open_h264_writer(path, fps, size)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def harvest_clips(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    detector: DetectorLike,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    pad_s: float = DEFAULT_PAD_S,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    max_len_s: float = DEFAULT_MAX_LEN_S,
    source_start_utc: datetime | None = None,
    source_id: str | None = None,
    camera_name: str | None = None,
    detect_conf: float | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_age_frames: int = DEFAULT_MAX_AGE_FRAMES,
    center_dist_gate: float = DEFAULT_CENTER_DIST_GATE,
    detect_batch_size: int = DEFAULT_DETECT_BATCH_SIZE,
    capture_factory: Callable[[str], Any] = open_capture,
    clip_writer_factory: Callable[[Path, float, tuple[int, int]], ClipWriter] = (
        _default_clip_writer_factory
    ),
) -> list[HarvestResult]:
    """Harvest dog-present spans from ``input_path`` into ``out_dir``.

    Returns one :class:`HarvestResult` per written span. Re-running is idempotent:
    span directories are keyed by a deterministic id, so an unchanged clip yields
    the same outputs. ``source_start_utc`` anchors absolute time; when omitted it
    is derived from the filename / mtime via :func:`derive_base_wall_ts`.
    ``source_id`` overrides the provenance/span-id key (derived from
    ``input_path`` when omitted); pass a stable value for chunked sources so
    span ids stay deterministic regardless of the temp file path.
    ``camera_name`` (friendly NVR name) and ``detect_conf`` (the detector's
    confidence gate) are recorded verbatim in each span's ``metadata.json`` for
    the labeling UI; both are optional provenance and never affect span math.
    """

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    if source_start_utc is None:
        source_start_utc, _ = derive_base_wall_ts(input_path)
    source_start_utc = source_start_utc.astimezone(timezone.utc)
    if source_id is None:
        source_id = sanitize_source_id(str(input_path))

    fps, total_frames, presence = _scan_for_dogs(
        input_path,
        detector=detector,
        sample_every=sample_every,
        iou_threshold=iou_threshold,
        max_age_frames=max_age_frames,
        center_dist_gate=center_dist_gate,
        detect_batch_size=detect_batch_size,
        capture_factory=capture_factory,
    )
    if total_frames == 0:
        logger.warning("harvest: no frames decoded from %s", input_path)
        return []

    spans = compute_spans(
        presence,
        fps=fps,
        total_frames=total_frames,
        merge_gap_s=merge_gap_s,
        pad_s=pad_s,
        min_len_s=min_len_s,
        max_len_s=max_len_s,
    )
    if not spans:
        logger.info("harvest: no dog spans in %s", input_path)
        return []

    return _write_spans(
        spans,
        input_path=input_path,
        out_dir=out_dir,
        fps=fps,
        source_id=source_id,
        source_start_utc=source_start_utc,
        sample_every=sample_every,
        camera_name=camera_name,
        detect_conf=detect_conf,
        model_name=getattr(detector, "model_name", None),
        capture_factory=capture_factory,
        clip_writer_factory=clip_writer_factory,
    )


def _scan_for_dogs(
    input_path: Path,
    *,
    detector: DetectorLike,
    sample_every: int,
    iou_threshold: float,
    max_age_frames: int,
    center_dist_gate: float = 0.0,
    detect_batch_size: int = DEFAULT_DETECT_BATCH_SIZE,
    capture_factory: Callable[[str], Any],
) -> tuple[float, int, dict[str, list[FrameSample]]]:
    capture = capture_factory(str(input_path))
    if not _capture_opened(capture):
        _release(capture)
        raise RuntimeError(f"failed to open video file: {input_path}")

    fps = _capture_value(capture, cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker(
        iou_threshold=iou_threshold,
        max_age_frames=max_age_frames,
        center_dist_gate=center_dist_gate,
    )
    presence: dict[str, list[FrameSample]] = {}

    def _decode_frames():
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame

    # Feed one sampled frame's detections through the tracker and record presence.
    # Detections are per-image-independent, so replaying a batch's results in
    # frame order here yields the same tracks (and thus the same spans) as the
    # per-frame path -- batching only changes how the forward pass is issued.
    def _ingest(frame_idx: int, detections: list[Detection]) -> None:
        tracks = tracker.update(list(detections))
        time_s = frame_idx / fps
        for track in tracks:
            latest = _latest_detection_at(track, frame_idx)
            if latest is None:
                continue
            presence.setdefault(track.track_id, []).append(
                FrameSample(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    bbox=latest.bbox,
                    confidence=latest.confidence,
                    class_name=latest.class_name,
                )
            )

    detect_batch = getattr(detector, "detect_batch", None)
    batch_size = max(1, detect_batch_size)
    use_batch = detect_batch is not None and batch_size > 1

    pending_frames: list[np.ndarray] = []
    pending_idx: list[int] = []

    def _flush_batch() -> None:
        if not pending_frames:
            return
        metas = [FrameMeta(frame_idx=i) for i in pending_idx]
        results = detect_batch(pending_frames, metas)
        for idx, dets in zip(pending_idx, results):
            _ingest(idx, list(dets))
        pending_frames.clear()
        pending_idx.clear()

    decoded_idx = 0
    try:
        # Pipeline decode against detect+track: a background thread reads ahead
        # while the (GIL-releasing) detector runs, turning a decode-bound scan into
        # an inference-bound one. Tracking stays sequential on this thread, and
        # sampled frames are detected in batches of ``batch_size`` (when the
        # detector supports it) so the accelerator runs one forward per batch.
        for frame in prefetch(_decode_frames()):
            if decoded_idx % sample_every == 0:
                if use_batch:
                    pending_frames.append(frame)
                    pending_idx.append(decoded_idx)
                    if len(pending_frames) >= batch_size:
                        _flush_batch()
                else:
                    detections = detector.detect(frame, frame_idx=decoded_idx)
                    _ingest(decoded_idx, list(detections))
            decoded_idx += 1
        if use_batch:
            _flush_batch()
    finally:
        _release(capture)

    return fps, decoded_idx, presence


def _write_spans(
    spans: Sequence[DogSpan],
    *,
    input_path: Path,
    out_dir: Path,
    fps: float,
    source_id: str,
    source_start_utc: datetime,
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

    sizes = _extract_span_clips(
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
        metadata_path = _write_clip_metadata(
            clip_dir,
            span=span,
            span_id=span_id,
            source_id=source_id,
            source_path=input_path,
            source_start_utc=source_start_utc,
            fps=fps,
            width=width,
            height=height,
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


def _merge_frame_ranges(
    ranges: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
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


def _seek_capture(capture: Any, frame_idx: int) -> bool:
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


def _capture_can_seek(capture: Any) -> bool:
    return callable(getattr(capture, "set", None))


def _extract_span_clips(
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
    if not _capture_opened(capture):
        _release(capture)
        raise RuntimeError(f"failed to open video file: {input_path}")

    segments = _merge_frame_ranges(
        [(span.start_frame, span.end_frame) for span, _, _ in plans]
    )
    writers: dict[str, ClipWriter] = {}
    sizes: dict[str, tuple[int, int]] = {}

    def _emit(frame: np.ndarray, decoded_idx: int) -> None:
        height, width = frame.shape[:2]
        for span, span_id, clip_dir in plans:
            if span.start_frame <= decoded_idx <= span.end_frame:
                writer = writers.get(span_id)
                if writer is None:
                    writer = clip_writer_factory(
                        clip_dir / CLIP_NAME, fps, (width, height)
                    )
                    writers[span_id] = writer
                    sizes[span_id] = (width, height)
                writer.write(frame)

    try:
        seekable = False
        if segments and _capture_can_seek(capture):
            seekable = _seek_capture(capture, segments[0][0])
            if not seekable:
                raise RuntimeError(
                    f"seek to frame {segments[0][0]} failed before extraction"
                )
        if seekable:
            for seg_index, (seg_start, seg_end) in enumerate(segments):
                if seg_index > 0 and not _seek_capture(capture, seg_start):
                    raise RuntimeError(
                        f"seek to frame {seg_start} failed mid-extraction"
                    )
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
            _release(capture)
        if release_errors:
            raise release_errors[0]
    return sizes


def _write_clip_metadata(
    clip_dir: Path,
    *,
    span: DogSpan,
    span_id: str,
    source_id: str,
    source_path: Path,
    source_start_utc: datetime,
    fps: float,
    width: int,
    height: int,
    sample_every: int,
    camera_name: str | None = None,
    detect_conf: float | None = None,
    model_name: str | None = None,
) -> Path:
    source_span_start_utc = source_start_utc + timedelta(seconds=span.start_s)
    source_span_end_utc = source_start_utc + timedelta(seconds=span.end_s)

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
        "schema_version": SCHEMA_VERSION,
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
        "timebase": TIME_BASIS_CLIP_FRAMES,
        "sample_every": sample_every,
        "track_id": span.track_id,
        "source_start_frame": span.start_frame,
        "source_end_frame": span.end_frame,
        "start_s": span.start_s,
        "end_s": span.end_s,
        "detections": detections,
        "checksum": _sha256_file(clip_dir / CLIP_NAME),
    }

    import json
    import os

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


def _latest_detection_at(track: Any, frame_idx: int) -> Detection | None:
    best: Detection | None = None
    for det in track.detections:
        if det.frame_idx == frame_idx:
            if best is None or det.confidence > best.confidence:
                best = det
    return best


def _capture_opened(capture: Any) -> bool:
    is_opened = getattr(capture, "isOpened", None)
    return bool(is_opened()) if callable(is_opened) else True


def _capture_value(capture: Any, prop: int) -> float | None:
    value = capture.get(prop)
    return float(value) if value and value > 0 else None


def _release(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()
