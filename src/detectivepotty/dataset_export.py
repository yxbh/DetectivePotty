"""Dataset exporter: harvested clips + ``labels.json`` -> classifier crops.

Walks harvested clip directories (``clip.mp4`` + ``metadata.json`` +
``labels.json``) and turns each labeled range into cropped training images.

Correctness rules this module enforces (from the design critique):

* **Dense re-detection.** Frames are decoded sequentially from the clip and the
  detector is re-run on each exported frame; we never crop from the sparse boxes
  recorded at harvest time. Those harvest boxes are used only as a *reference* to
  pick which freshly-detected box the labeled track corresponds to (max IoU).
* **Track binding.** A range labels one dog (``track_id``); the crop comes from
  the re-detected box matching that track's reference, so multi-dog frames crop
  the right animal. Frames where no box matches are dropped (and counted), never
  cropped from a stale box.
* **Within-range sampling.** Consecutive frames are near-duplicates, so each range
  is sub-sampled by a time stride and capped per range.
* **Split by day/source-recording**, not by random frame, so adjacent frames and
  same-recording background can't leak across train/val.
* ``excluded`` behavior never produces crops; ``unknown`` dog is excluded from the
  dog-ID tree but can still feed the behavior tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detectivepotty.detect import FrameDetector
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox, crop_from_frame
from detectivepotty.harvest import CLIP_NAME, METADATA_NAME
from detectivepotty.labels import (
    LABELS_NAME,
    ClipLabels,
    Dog,
    LabelRange,
    load_labels,
)
from detectivepotty.recording.dataset import sanitize_path_component
from detectivepotty.sources.pyav_capture import open_capture
from detectivepotty.timeline import FrameTimeline, timeline_from_metadata
from detectivepotty.tracking import iou

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_STRIDE_S = 0.3
DEFAULT_MAX_FRAMES_PER_RANGE = 40
DEFAULT_CROP_MARGIN_FRAC = 0.35
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_MIN_IOU = 0.3
DEFAULT_JPEG_QUALITY = 92

MANIFEST_NAME = "manifest.csv"
MANIFEST_HEADER = (
    "crop_path,dog_crop_path,behavior,dog,track_id,clip_id,source_id,date,split,matched"
)


DetectorLike = FrameDetector


@dataclass(slots=True)
class ExportStats:
    clips: int = 0
    ranges: int = 0
    frames_sampled: int = 0
    crops_written: int = 0
    dropped_unmatched: int = 0
    excluded_ranges: int = 0
    behavior_counts: dict[str, int] = field(default_factory=dict)
    dog_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clips": self.clips,
            "ranges": self.ranges,
            "frames_sampled": self.frames_sampled,
            "crops_written": self.crops_written,
            "dropped_unmatched": self.dropped_unmatched,
            "excluded_ranges": self.excluded_ranges,
            "behavior_counts": dict(sorted(self.behavior_counts.items())),
            "dog_counts": dict(sorted(self.dog_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
        }


def sample_range_frames(
    start_frame: int,
    end_frame: int,
    fps: float,
    *,
    stride_s: float = DEFAULT_SAMPLE_STRIDE_S,
    max_frames: int = DEFAULT_MAX_FRAMES_PER_RANGE,
    timeline: FrameTimeline | None = None,
) -> list[int]:
    """Sub-sample a frame range by time stride, capped at ``max_frames``.

    Sampling by time (not raw index) keeps density fps-independent. Always yields
    at least the first frame; if the strided list still exceeds ``max_frames`` it
    is evenly thinned. Returned indices are unique and sorted.
    """

    if end_frame < start_frame:
        return []
    if timeline is None or (not timeline.has_pts and timeline.frame_count <= end_frame):
        timeline = FrameTimeline.cfr(fps=fps, frame_count=end_frame + 1)
    return timeline.sample_frames_by_time(
        start_frame,
        end_frame,
        stride_s=stride_s,
        max_frames=max_frames,
    )


def assign_split(key: str, val_fraction: float) -> str:
    """Deterministically map a day/source key to ``train``/``val``."""

    if val_fraction <= 0:
        return "train"
    if val_fraction >= 1:
        return "val"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    bucket = int(digest, 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def discover_clip_dirs(clips_root: str | Path) -> list[Path]:
    """Return harvested clip dirs (have ``labels.json`` + ``clip.mp4``), sorted."""

    root = Path(clips_root)
    found: list[Path] = []
    for labels_path in root.rglob(LABELS_NAME):
        clip_dir = labels_path.parent
        if (clip_dir / CLIP_NAME).exists():
            found.append(clip_dir)
    return sorted(set(found))


@dataclass(slots=True)
class _ClipContext:
    clip_dir: Path
    clip_path: Path
    fps: float
    timeline: FrameTimeline
    source_id: str
    date: str
    split: str
    # track_id -> sorted [(clip_frame_idx, bbox)]
    reference_boxes: dict[str, list[tuple[int, BBox]]]


def _load_metadata(clip_dir: Path) -> dict[str, Any]:
    import json

    with (clip_dir / METADATA_NAME).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_clip_context(clip_dir: Path, val_fraction: float) -> _ClipContext:
    meta = _load_metadata(clip_dir)
    fps = float(meta.get("fps") or 30.0)
    timeline = timeline_from_metadata(meta)
    source_id = str(meta.get("source_id") or clip_dir.name)
    start_utc = str(meta.get("source_span_start_utc") or "")
    date = start_utc[:10] if start_utc else "unknown-date"

    references: dict[str, list[tuple[int, BBox]]] = {}
    for det in meta.get("detections", []):
        track_id = str(det.get("track_id"))
        box = det.get("bbox") or {}
        bbox = BBox(
            x1=float(box["x1"]),
            y1=float(box["y1"]),
            x2=float(box["x2"]),
            y2=float(box["y2"]),
        )
        references.setdefault(track_id, []).append(
            (int(det["clip_frame_idx"]), bbox)
        )
    for entries in references.values():
        entries.sort(key=lambda item: item[0])

    split_key = f"{sanitize_path_component(source_id)}|{date}"
    return _ClipContext(
        clip_dir=clip_dir,
        clip_path=clip_dir / CLIP_NAME,
        fps=fps,
        timeline=timeline,
        source_id=source_id,
        date=date,
        split=assign_split(split_key, val_fraction),
        reference_boxes=references,
    )


def _reference_box_for(
    ctx: _ClipContext, track_id: str | None, clip_frame_idx: int
) -> BBox | None:
    if track_id is None:
        return None
    entries = ctx.reference_boxes.get(str(track_id))
    if not entries:
        return None
    best: tuple[int, BBox] | None = None
    best_dist = None
    for ref_idx, bbox in entries:
        dist = abs(ref_idx - clip_frame_idx)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (ref_idx, bbox)
    return best[1] if best is not None else None


def _pick_box(
    detections: list[Detection], reference: BBox | None, min_iou: float
) -> tuple[BBox | None, bool]:
    """Return ``(box, matched)``. With a reference, pick the max-IoU detection.

    Without a reference (single-dog clip / no track bound), fall back to the
    highest-confidence detection. ``matched`` is False when nothing cleared the
    IoU gate (caller decides whether to drop).
    """

    if not detections:
        return None, False
    if reference is None:
        best = max(detections, key=lambda d: d.confidence)
        return best.bbox, True
    scored = [(iou(reference, d.bbox), d) for d in detections]
    best_iou, best_det = max(scored, key=lambda item: item[0])
    if best_iou >= min_iou:
        return best_det.bbox, True
    return None, False


def export_dataset(
    clips_root: str | Path,
    out_dir: str | Path,
    *,
    detector: DetectorLike,
    sample_stride_s: float = DEFAULT_SAMPLE_STRIDE_S,
    max_frames_per_range: int = DEFAULT_MAX_FRAMES_PER_RANGE,
    crop_margin_frac: float = DEFAULT_CROP_MARGIN_FRAC,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    min_iou: float = DEFAULT_MIN_IOU,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    capture_factory: Callable[[str], Any] = open_capture,
) -> ExportStats:
    """Export classifier crops + a CSV manifest from labeled harvested clips."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = [MANIFEST_HEADER]
    stats = ExportStats()
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    for clip_dir in discover_clip_dirs(clips_root):
        labels = load_labels(clip_dir)
        ctx = _build_clip_context(clip_dir, val_fraction)
        stats.clips += 1
        _export_clip(
            ctx,
            labels,
            out_path=out_path,
            detector=detector,
            sample_stride_s=sample_stride_s,
            max_frames_per_range=max_frames_per_range,
            crop_margin_frac=crop_margin_frac,
            min_iou=min_iou,
            jpeg_params=jpeg_params,
            capture_factory=capture_factory,
            manifest_rows=manifest_rows,
            stats=stats,
        )

    (out_path / MANIFEST_NAME).write_text("\n".join(manifest_rows) + "\n", "utf-8")
    _write_stats(out_path, stats)
    return stats


def _export_clip(
    ctx: _ClipContext,
    labels: ClipLabels,
    *,
    out_path: Path,
    detector: DetectorLike,
    sample_stride_s: float,
    max_frames_per_range: int,
    crop_margin_frac: float,
    min_iou: float,
    jpeg_params: list[int],
    capture_factory: Callable[[str], Any],
    manifest_rows: list[str],
    stats: ExportStats,
) -> None:
    # Gather the frames we need and which ranges each frame feeds, so one
    # sequential decode pass (+ one detect per frame) serves every range.
    needed: dict[int, list[LabelRange]] = {}
    for rng in labels.ranges:
        stats.ranges += 1
        if not rng.is_trainable:
            stats.excluded_ranges += 1
            continue
        frames = sample_range_frames(
            rng.start_frame,
            rng.end_frame,
            ctx.fps,
            stride_s=sample_stride_s,
            max_frames=max_frames_per_range,
            timeline=ctx.timeline,
        )
        for clip_frame_idx in frames:
            needed.setdefault(clip_frame_idx, []).append(rng)

    if not needed:
        return

    capture = capture_factory(str(ctx.clip_path))
    is_opened = getattr(capture, "isOpened", None)
    if callable(is_opened) and not is_opened():
        logger.warning("export: could not open clip %s", ctx.clip_path)
        _release(capture)
        return

    decoded_idx = 0
    remaining = len(needed)
    try:
        while remaining > 0:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            ranges = needed.get(decoded_idx)
            if ranges is not None:
                remaining -= 1
                detections = detector.detect(frame, frame_idx=decoded_idx)
                for rng in ranges:
                    stats.frames_sampled += 1
                    _emit_crop(
                        frame,
                        detections,
                        rng,
                        ctx,
                        clip_frame_idx=decoded_idx,
                        out_path=out_path,
                        crop_margin_frac=crop_margin_frac,
                        min_iou=min_iou,
                        jpeg_params=jpeg_params,
                        manifest_rows=manifest_rows,
                        stats=stats,
                    )
            decoded_idx += 1
    finally:
        _release(capture)


def _emit_crop(
    frame: np.ndarray,
    detections: list[Detection],
    rng: LabelRange,
    ctx: _ClipContext,
    *,
    clip_frame_idx: int,
    out_path: Path,
    crop_margin_frac: float,
    min_iou: float,
    jpeg_params: list[int],
    manifest_rows: list[str],
    stats: ExportStats,
) -> None:
    reference = _reference_box_for(ctx, rng.track_id, clip_frame_idx)
    box, matched = _pick_box(detections, reference, min_iou)
    if not matched or box is None:
        stats.dropped_unmatched += 1
        return

    crop = crop_from_frame(frame, box, margin_frac=crop_margin_frac)
    if crop.size == 0:
        stats.dropped_unmatched += 1
        return

    stem = f"{ctx.clip_dir.name}_f{clip_frame_idx:06d}_t{rng.track_id}"
    behavior = rng.behavior.value
    behavior_rel = Path("behavior") / ctx.split / behavior / f"{stem}.jpg"
    _write_jpeg(out_path / behavior_rel, crop, jpeg_params)
    stats.behavior_counts[behavior] = stats.behavior_counts.get(behavior, 0) + 1
    stats.split_counts[ctx.split] = stats.split_counts.get(ctx.split, 0) + 1
    stats.crops_written += 1

    dog_rel: Path | None = None
    if rng.dog is not Dog.UNKNOWN:
        dog = rng.dog.value
        dog_rel = Path("dog") / ctx.split / dog / f"{stem}.jpg"
        _write_jpeg(out_path / dog_rel, crop, jpeg_params)
        stats.dog_counts[dog] = stats.dog_counts.get(dog, 0) + 1

    manifest_rows.append(
        ",".join(
            _csv_field(value)
            for value in (
                behavior_rel.as_posix(),
                dog_rel.as_posix() if dog_rel else "",
                behavior,
                rng.dog.value,
                rng.track_id or "",
                ctx.clip_dir.name,
                ctx.source_id,
                ctx.date,
                ctx.split,
                "1" if matched else "0",
            )
        )
    )


def _write_stats(out_path: Path, stats: ExportStats) -> None:
    import json

    (out_path / "export_stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8"
    )


def _write_jpeg(path: Path, image: np.ndarray, params: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"failed to write JPEG: {path}")


def _csv_field(value: str) -> str:
    if any(ch in value for ch in (",", '"', "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _release(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()
