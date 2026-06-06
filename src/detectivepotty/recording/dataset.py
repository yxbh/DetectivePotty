"""Dataset layout and image artifact helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from detectivepotty.events import CropRecord, Detection, FrameRecord, Track
from detectivepotty.geometry import crop_from_frame
from detectivepotty.sources.base import Frame, sanitize_source_id

DEFAULT_JPEG_QUALITY = 92
DEFAULT_CROP_MARGIN_FRAC = 0.35
_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_UNDERSCORE_RE = re.compile(r"_+")


def sanitize_path_component(value: object) -> str:
    """Return a filesystem-safe, secret-stripped path component."""

    safe = sanitize_source_id(str(value or ""))
    safe = _COMPONENT_RE.sub("_", safe)
    safe = _UNDERSCORE_RE.sub("_", safe).strip("._")
    return safe or "unknown"


def format_event_timestamp(dt: datetime) -> str:
    """Format a timestamp as sortable UTC, e.g. 20260606T091047Z."""

    return _ensure_utc(dt).strftime("%Y%m%dT%H%M%SZ")


def camera_dataset_dir(
    dataset_dir: str | Path,
    camera_id: str,
    camera_name: str | None = None,
) -> Path:
    return Path(dataset_dir) / _camera_component(camera_id, camera_name)


def event_dir(
    dataset_dir: str | Path,
    camera_id: str,
    camera_name: str | None,
    start_ts: datetime,
    track_id: str,
    event_id: str,
) -> Path:
    """Build dataset/<camera>/<YYYY-MM-DD>/events/<ts>_<camera>_<track>_<uuid>."""

    utc_start = _ensure_utc(start_ts)
    camera = _camera_component(camera_id, camera_name)
    track = sanitize_path_component(track_id)
    safe_event_id = sanitize_path_component(event_id)
    event_name = f"{format_event_timestamp(utc_start)}_{camera}_{track}_{safe_event_id}"
    return Path(dataset_dir) / camera / utc_start.strftime("%Y-%m-%d") / "events" / event_name


def write_event_images(
    target_event_dir: str | Path,
    frames: Sequence[Frame],
    detections: Sequence[Detection],
    tracks: Sequence[Track],
    primary_track_id: str,
    *,
    substream: str | None = None,
    crop_margin_frac: float = DEFAULT_CROP_MARGIN_FRAC,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[list[FrameRecord], list[CropRecord]]:
    """Write full frames and primary dog crops.

    The default 0.35 crop margin keeps context around the dog while preserving
    high-resolution body detail for later pee/poop classifier training.
    """

    event_path = Path(target_event_dir)
    frames_dir = event_path / "frames"
    crops_dir = event_path / "crops"
    _reset_output_dir(frames_dir)
    _reset_output_dir(crops_dir)

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    best_by_frame = _best_detection_by_frame(detections, tracks, primary_track_id)
    frame_records: list[FrameRecord] = []
    crop_records: list[CropRecord] = []

    for index, frame in enumerate(frames):
        frame_name = f"{index:03d}.jpg"
        _write_jpeg(frames_dir / frame_name, frame.bgr, jpeg_params)
        frame_records.append(
            FrameRecord(
                frame_idx=frame.frame_idx,
                source_id=sanitize_source_id(frame.source_id),
                substream=substream,
                original_width=frame.width,
                original_height=frame.height,
            ),
        )

        detection = best_by_frame.get(frame.frame_idx)
        if detection is None:
            continue
        crop = crop_from_frame(frame.bgr, detection.bbox, margin_frac=crop_margin_frac)
        if crop.size == 0:
            continue
        crop_name = f"{len(crop_records):03d}.jpg"
        crop_rel = Path("crops") / crop_name
        _write_jpeg(event_path / crop_rel, crop, jpeg_params)
        crop_records.append(
            CropRecord(
                frame_idx=frame.frame_idx,
                bbox=detection.bbox,
                margin_frac=crop_margin_frac,
                path=crop_rel.as_posix(),
            ),
        )

    return frame_records, crop_records


def _best_detection_by_frame(
    detections: Sequence[Detection],
    tracks: Sequence[Track],
    primary_track_id: str,
) -> dict[int, Detection]:
    primary = []
    for track in tracks:
        if track.track_id == primary_track_id:
            primary.extend(track.detections)
    primary_by_frame = _group_best(primary)
    all_by_frame = _group_best(detections)
    return {**all_by_frame, **primary_by_frame}


def _group_best(detections: Sequence[Detection]) -> dict[int, Detection]:
    best: dict[int, Detection] = {}
    for detection in detections:
        current = best.get(detection.frame_idx)
        if current is None or _detection_rank(detection) > _detection_rank(current):
            best[detection.frame_idx] = detection
    return best


def _detection_rank(detection: Detection) -> tuple[float, float]:
    return (detection.confidence, detection.bbox.area)


def _camera_component(camera_id: str, camera_name: str | None) -> str:
    camera = sanitize_path_component(camera_name or camera_id)
    if camera == "unknown":
        return sanitize_path_component(camera_id)
    return camera


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _write_jpeg(path: Path, image: np.ndarray, params: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"failed to write JPEG: {path}")
