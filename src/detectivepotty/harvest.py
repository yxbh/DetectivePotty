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

from collections.abc import Callable
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from detectivepotty.harvest_scan import (
    DEFAULT_DETECT_BATCH_SIZE,
    DetectorLike,
    _latest_detection_at as _latest_detection_at,
    scan_for_dogs as _scan_for_dogs,
)
from detectivepotty.harvest_spans import (
    DEFAULT_MAX_LEN_S,
    DEFAULT_MERGE_GAP_S,
    DEFAULT_MIN_LEN_S,
    DEFAULT_PAD_S,
    DogSpan,
    FrameSample as FrameSample,
    HarvestResult,
    compute_spans,
    make_span_id,
)
from detectivepotty.harvest_writer import (
    CLIP_NAME,
    METADATA_NAME,
    SCHEMA_VERSION,
    TIME_BASIS_CLIP_FRAMES,
    ClipWriter,
    capture_can_seek as _capture_can_seek,
    capture_opened as _capture_opened,
    default_clip_writer_factory as _default_clip_writer_factory,
    extract_span_clips as _extract_span_clips,
    merge_frame_ranges as _merge_frame_ranges,
    release_capture as _release,
    seek_capture as _seek_capture,
    sha256_file as _sha256_file,
    write_clip_metadata as _write_clip_metadata,
    write_spans as _write_spans,
)
from detectivepotty.sources.base import sanitize_source_id
from detectivepotty.sources.file import derive_base_wall_ts
from detectivepotty.sources.pyav_capture import open_capture

logger = logging.getLogger(__name__)

__all__ = [
    "CLIP_NAME",
    "DEFAULT_CENTER_DIST_GATE",
    "DEFAULT_DETECT_BATCH_SIZE",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_MAX_AGE_FRAMES",
    "DEFAULT_MAX_LEN_S",
    "DEFAULT_MERGE_GAP_S",
    "DEFAULT_MIN_LEN_S",
    "DEFAULT_PAD_S",
    "DEFAULT_SAMPLE_EVERY",
    "METADATA_NAME",
    "SCHEMA_VERSION",
    "TIME_BASIS_CLIP_FRAMES",
    "ClipWriter",
    "DetectorLike",
    "DogSpan",
    "FrameSample",
    "HarvestResult",
    "_capture_can_seek",
    "_capture_opened",
    "_default_clip_writer_factory",
    "_extract_span_clips",
    "_latest_detection_at",
    "_merge_frame_ranges",
    "_release",
    "_scan_for_dogs",
    "_seek_capture",
    "_sha256_file",
    "_write_clip_metadata",
    "_write_spans",
    "compute_spans",
    "harvest_clips",
    "make_span_id",
]

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
