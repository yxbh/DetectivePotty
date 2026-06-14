"""Pure span math and types for historical clip harvesting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from detectivepotty.geometry import BBox
from detectivepotty.timeline import FrameTimeline

DEFAULT_MERGE_GAP_S = 2.0
DEFAULT_PAD_S = 1.0
DEFAULT_MIN_LEN_S = 0.5
DEFAULT_MAX_LEN_S = 60.0


@dataclass(frozen=True, slots=True)
class FrameSample:
    """A single sampled detection of one track, in the *source* clip's numbering."""

    frame_idx: int
    time_s: float
    bbox: BBox
    confidence: float
    class_name: str = "dog"


@dataclass(slots=True)
class DogSpan:
    """A contiguous dog-present window for one track, in source-clip coordinates.

    ``start_frame``/``end_frame`` are inclusive source-frame indices; ``start_s``/
    ``end_s`` are the padded/clamped seconds. ``samples`` are the track's sampled
    detections inside the (unpadded) span, retained as reference boxes for the
    exporter's dense re-detection binding.
    """

    track_id: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    samples: list[FrameSample] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(slots=True)
class HarvestResult:
    span: DogSpan
    span_id: str
    clip_dir: Path
    clip_path: Path
    metadata_path: Path


def compute_spans(
    presence: Mapping[str, Sequence[FrameSample]],
    *,
    fps: float,
    total_frames: int,
    timeline: FrameTimeline | None = None,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    pad_s: float = DEFAULT_PAD_S,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    max_len_s: float = DEFAULT_MAX_LEN_S,
) -> list[DogSpan]:
    """Group per-track sampled detections into padded, clamped, capped spans.

    ``presence`` maps ``track_id`` -> that track's sampled detections (any order).
    Spans are computed independently per track (overlaps across tracks are
    expected and allowed). A new span starts whenever the gap between consecutive
    samples of a track exceeds ``merge_gap_s``. Each raw span is padded by
    ``pad_s`` on both sides, clamped to ``[0, total_frames-1]`` / ``[0, duration]``,
    dropped if shorter than ``min_len_s``, and split into ``max_len_s`` chunks if
    longer. Returned spans are sorted by ``(start_frame, track_id)``.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if total_frames <= 0:
        return []
    timeline = timeline or FrameTimeline.cfr(fps=fps, frame_count=total_frames)
    duration_s = timeline.duration_s

    spans: list[DogSpan] = []
    for track_id, raw_samples in presence.items():
        samples = sorted(raw_samples, key=lambda s: s.frame_idx)
        if not samples:
            continue

        group: list[FrameSample] = [samples[0]]
        for sample in samples[1:]:
            if sample.time_s - group[-1].time_s > merge_gap_s:
                spans.extend(
                    _finalize_group(
                        track_id,
                        group,
                        fps,
                        total_frames,
                        duration_s,
                        timeline,
                        pad_s,
                        min_len_s,
                        max_len_s,
                    )
                )
                group = [sample]
            else:
                group.append(sample)
        spans.extend(
            _finalize_group(
                track_id,
                group,
                fps,
                total_frames,
                duration_s,
                timeline,
                pad_s,
                min_len_s,
                max_len_s,
            )
        )

    spans.sort(key=lambda span: (span.start_frame, _track_sort_key(span.track_id)))
    return spans


def _finalize_group(
    track_id: str,
    group: Sequence[FrameSample],
    fps: float,
    total_frames: int,
    duration_s: float,
    timeline: FrameTimeline,
    pad_s: float,
    min_len_s: float,
    max_len_s: float,
) -> list[DogSpan]:
    raw_start_s = group[0].time_s
    raw_end_s = group[-1].time_s
    start_s = _clamp(raw_start_s - pad_s, 0.0, duration_s)
    end_s = _clamp(raw_end_s + pad_s, 0.0, duration_s)
    if end_s - start_s < min_len_s:
        return []

    chunks: list[tuple[float, float]] = []
    if max_len_s > 0 and end_s - start_s > max_len_s:
        cursor = start_s
        while cursor < end_s - 1e-9:
            chunk_end = min(cursor + max_len_s, end_s)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
    else:
        chunks.append((start_s, end_s))

    out: list[DogSpan] = []
    _ = fps
    for chunk_start_s, chunk_end_s in chunks:
        start_frame = int(
            _clamp(timeline.seconds_to_frame_nearest(chunk_start_s), 0, total_frames - 1)
        )
        end_frame = int(
            _clamp(timeline.seconds_to_frame_nearest(chunk_end_s), 0, total_frames - 1)
        )
        if end_frame < start_frame:
            end_frame = start_frame
        chunk_samples = [
            s for s in group if chunk_start_s - 1e-9 <= s.time_s <= chunk_end_s + 1e-9
        ]
        out.append(
            DogSpan(
                track_id=track_id,
                start_frame=start_frame,
                end_frame=end_frame,
                start_s=chunk_start_s,
                end_s=chunk_end_s,
                samples=chunk_samples,
            )
        )
    return out


def make_span_id(source_id: str, span: DogSpan) -> str:
    """Deterministic, idempotent span id from source + frame range + track."""

    digest = hashlib.sha1(
        f"{source_id}|{span.start_frame}|{span.end_frame}|{span.track_id}".encode()
    ).hexdigest()[:10]
    return f"{span.start_frame:07d}_{span.end_frame:07d}_t{span.track_id}_{digest}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _track_sort_key(track_id: str) -> tuple[int, str]:
    try:
        return (int(track_id), track_id)
    except ValueError:
        return (0, track_id)
