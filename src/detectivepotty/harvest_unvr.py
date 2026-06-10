"""Historical UNVR / Protect harvest in time chunks.

Pulls a long wall-clock window (e.g. a full 24 h day) off a UniFi Protect NVR in
bounded **chunks** (hourly by default), runs the file-based :mod:`harvest`
pipeline on each downloaded chunk, and stitches the per-chunk spans together by
absolute source time — deduping the small overlap between adjacent chunks.

Design (mirrors :mod:`harvest`'s injected-seam pattern so it stays offline-testable):

- ``plan_chunks`` is pure interval math: split ``[start, end]`` into windows of
  ``chunk_s`` that each carry a small ``overlap_s`` tail into the next window, so a
  dog crossing a boundary is fully captured in at least one chunk.
- ``harvest_camera_window`` orchestrates download → per-chunk
  :func:`harvest.harvest_clips` → cross-chunk dedup. The NVR download is injected
  as ``download_fn`` (defaulting to a thin ``ProtectClient.download_recording``
  wrapper at the CLI), so tests drive it with a synthetic-clip writer and a fake
  detector — no NVR, network, GPU, or model.

Robustness: a chunk whose download fails or returns nothing (motion-only gap, no
recording) is logged and skipped; harvesting continues with the remaining chunks.
Chunk source ids are deterministic (``camera|chunk_start``) so re-running a day is
idempotent — unchanged chunks reproduce the same span dirs.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2

from detectivepotty.harvest import (
    DEFAULT_MAX_LEN_S,
    DEFAULT_MERGE_GAP_S,
    DEFAULT_MIN_LEN_S,
    DEFAULT_PAD_S,
    DEFAULT_SAMPLE_EVERY,
    ClipWriter,
    DetectorLike,
    HarvestResult,
    _default_clip_writer_factory,
    harvest_clips,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_S = 3600.0
DEFAULT_OVERLAP_S = 5.0
DEFAULT_DEDUP_TIME_IOU = 0.5

# download_fn(camera_id, start_utc, end_utc, dest) -> written path or None.
DownloadFn = Callable[[str, datetime, datetime, Path], Path | None]


def plan_chunks(
    start: datetime,
    end: datetime,
    *,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
) -> list[tuple[datetime, datetime]]:
    """Split ``[start, end)`` into ``chunk_s`` windows with an ``overlap_s`` tail.

    Each returned window is ``[t, min(t + chunk_s + overlap_s, end)]`` and the
    next starts at ``t + chunk_s``, so adjacent windows overlap by ``overlap_s``
    (clamped at ``end``). Returns ``[]`` when ``end <= start``.
    """

    if chunk_s <= 0:
        raise ValueError("chunk_s must be > 0")
    if overlap_s < 0:
        raise ValueError("overlap_s must be >= 0")
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return []

    step = timedelta(seconds=chunk_s)
    tail = timedelta(seconds=overlap_s)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        boundary = min(cursor + step, end)
        window_end = min(boundary + tail, end)
        chunks.append((cursor, window_end))
        cursor = boundary
    return chunks


def harvest_camera_window(
    camera_id: str,
    start: datetime,
    end: datetime,
    out_dir: str | Path,
    *,
    detector: DetectorLike,
    download_fn: DownloadFn,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    dedup_time_iou: float = DEFAULT_DEDUP_TIME_IOU,
    tmp_dir: str | Path | None = None,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    pad_s: float = DEFAULT_PAD_S,
    min_len_s: float = DEFAULT_MIN_LEN_S,
    max_len_s: float = DEFAULT_MAX_LEN_S,
    iou_threshold: float = 0.3,
    max_age_frames: int = 5,
    keep_chunks: bool = False,
    capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    clip_writer_factory: Callable[
        [Path, float, tuple[int, int]], ClipWriter
    ] = _default_clip_writer_factory,
) -> list[HarvestResult]:
    """Harvest dog spans across ``[start, end)`` for ``camera_id`` in chunks.

    Downloads each planned chunk via ``download_fn`` to a temp MP4, harvests it
    with the file pipeline (anchored at the chunk's absolute start), then dedupes
    spans whose absolute source-time intervals overlap an already-kept span by at
    least ``dedup_time_iou`` (the cross-chunk overlap region). Failed/empty chunk
    downloads are skipped. Temp chunk files are deleted unless ``keep_chunks``.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tmp_dir) if tmp_dir is not None else out_dir / ".chunks"
    tmp_root.mkdir(parents=True, exist_ok=True)

    chunks = plan_chunks(start, end, chunk_s=chunk_s, overlap_s=overlap_s)
    if not chunks:
        return []

    kept: list[HarvestResult] = []
    kept_intervals: list[tuple[float, float]] = []  # absolute epoch seconds

    for index, (chunk_start, chunk_end) in enumerate(chunks):
        dest = tmp_root / f"{_safe(camera_id)}_{chunk_start:%Y%m%dT%H%M%S}.mp4"
        try:
            path = download_fn(camera_id, chunk_start, chunk_end, dest)
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort the day
            logger.warning(
                "harvest-unvr: chunk %d/%d download failed (%s-%s): %s",
                index + 1, len(chunks), chunk_start.isoformat(), chunk_end.isoformat(), exc,
            )
            continue
        if path is None or not Path(path).exists():
            logger.info(
                "harvest-unvr: no recording for chunk %d/%d (%s-%s)",
                index + 1, len(chunks), chunk_start.isoformat(), chunk_end.isoformat(),
            )
            continue

        source_id = f"{camera_id}@{chunk_start.strftime('%Y%m%dT%H%M%SZ')}"
        try:
            results = harvest_clips(
                path,
                out_dir,
                detector=detector,
                sample_every=sample_every,
                merge_gap_s=merge_gap_s,
                pad_s=pad_s,
                min_len_s=min_len_s,
                max_len_s=max_len_s,
                source_start_utc=chunk_start,
                source_id=source_id,
                iou_threshold=iou_threshold,
                max_age_frames=max_age_frames,
                capture_factory=capture_factory,
                clip_writer_factory=clip_writer_factory,
            )
        finally:
            if not keep_chunks:
                _unlink(Path(path))

        base = chunk_start.timestamp()
        for result in results:
            span = result.span
            interval = (base + span.start_s, base + span.end_s)
            if _is_duplicate(interval, kept_intervals, dedup_time_iou):
                logger.debug(
                    "harvest-unvr: dropping cross-chunk duplicate span %s", result.span_id
                )
                _rmtree(result.clip_dir)
                continue
            kept.append(result)
            kept_intervals.append(interval)

    if not keep_chunks:
        _rmtree(tmp_root, missing_ok=True)
    return kept


def _is_duplicate(
    interval: tuple[float, float],
    kept: list[tuple[float, float]],
    min_iou: float,
) -> bool:
    return any(_time_iou(interval, other) >= min_iou for other in kept)


def _time_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0.0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name) or "camera"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _rmtree(path: Path, *, missing_ok: bool = False) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        if not missing_ok:
            raise
    except OSError:
        pass
