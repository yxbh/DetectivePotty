"""Shared event, detection, and metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from detectivepotty.geometry import BBox


class TriggerReason(str, Enum):
    PROTECT_ANIMAL = "protect_animal"
    YOLO = "yolo"


class LabelStatus(str, Enum):
    UNLABELED = "unlabeled"
    LABELED = "labeled"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class Label(str, Enum):
    PEE = "pee"
    POOP = "poop"
    NOT_POTTY = "not_potty"
    UNKNOWN = "unknown"


class ClassifierGuess(str, Enum):
    PEE = "pee"
    POOP = "poop"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_name: str
    frame_idx: int
    mono_ts: float
    wall_ts: datetime

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass(slots=True)
class Track:
    track_id: str
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self)


@dataclass(slots=True)
class FrameRecord:
    frame_idx: int
    source_id: str
    substream: str | None
    original_width: int
    original_height: int
    inference_width: int | None = None
    inference_height: int | None = None
    original_to_inference_scale_x: float | None = None
    original_to_inference_scale_y: float | None = None


@dataclass(slots=True)
class CropRecord:
    frame_idx: int
    bbox: BBox
    margin_frac: float
    path: str | None = None


@dataclass(slots=True)
class EventMetadata:
    schema_version: str = "1.1"
    event_id: str = field(default_factory=lambda: str(uuid4()))
    camera_id: str = ""
    camera_name: str = ""
    sanitized_source_id: str = ""
    utc_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_ts: datetime | None = None
    recorded_at: datetime | None = None
    source_start_s: float | None = None
    source_end_s: float | None = None
    time_basis: str | None = None
    local_tz_offset: str = field(default_factory=lambda: _local_tz_offset())
    protect_event_id: str | None = None
    smartdetect_score: float | None = None
    smartdetect_bbox: BBox | None = None
    detection_ts: datetime | None = None
    notification_ts: datetime | None = None
    trigger_latency_s: float | None = None
    model_name: str = ""
    model_version: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    trigger_reason: TriggerReason = TriggerReason.YOLO
    pre_roll_s: float = 0.0
    post_roll_s: float = 0.0
    frame_records: list[FrameRecord] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    crop_boxes: list[CropRecord] = field(default_factory=list)
    multi_dog: bool = False
    ambiguous: bool = False
    classifier_guess: ClassifierGuess = ClassifierGuess.UNKNOWN
    classifier_confidence: float | None = None
    label_status: LabelStatus = LabelStatus.UNLABELED
    label: Label = Label.UNKNOWN
    dog: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.utc_ts = _ensure_aware_utc(self.utc_ts)
        if self.end_ts is not None:
            self.end_ts = _ensure_aware_utc(self.end_ts)
        if self.recorded_at is not None:
            self.recorded_at = _ensure_aware_utc(self.recorded_at)
        if self.detection_ts is not None:
            self.detection_ts = _ensure_aware_utc(self.detection_ts)
        if self.notification_ts is not None:
            self.notification_ts = _ensure_aware_utc(self.notification_ts)
        if (
            self.trigger_latency_s is None
            and self.detection_ts is not None
            and self.notification_ts is not None
        ):
            delta = self.notification_ts - self.detection_ts
            self.trigger_latency_s = delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self)

    def write_json(self, event_dir_or_path: str | Path) -> Path:
        return write_metadata_json(self, event_dir_or_path)


def write_metadata_json(
    metadata: EventMetadata,
    event_dir_or_path: str | Path,
) -> Path:
    target = Path(event_dir_or_path)
    if target.suffix.lower() != ".json":
        target = target / "metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".metadata.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def _local_tz_offset() -> str:
    offset = datetime.now().astimezone().utcoffset()
    if offset is None:
        return "+00:00"
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _jsonify(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonify(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _ensure_aware_utc(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    return value
