"""Dataset scanning and label updates for the local web app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from detectivepotty.events import Label, LabelStatus


_UNSET = object()


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    dir_path: Path
    metadata: dict[str, Any]
    relative_dir: str
    camera_dir: str | None
    date_dir: str | None


class DatasetIndex:
    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()

    def scan(self) -> list[EventRecord]:
        if not self.dataset_dir.exists():
            return []

        records: list[EventRecord] = []
        root = self.dataset_dir.resolve()
        for metadata_path in root.rglob("metadata.json"):
            event_dir = metadata_path.parent
            if event_dir.parent.name != "events":
                continue
            try:
                metadata_real = metadata_path.resolve(strict=True)
                metadata_real.relative_to(root)
                with metadata_real.open("r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue

            event_id = str(metadata.get("event_id") or event_dir.name)
            relative = event_dir.relative_to(root)
            parts = relative.parts
            camera_dir = parts[0] if len(parts) >= 4 else None
            date_dir = parts[1] if len(parts) >= 4 else None
            records.append(
                EventRecord(
                    event_id=event_id,
                    dir_path=event_dir,
                    metadata=metadata,
                    relative_dir=relative.as_posix(),
                    camera_dir=camera_dir,
                    date_dir=date_dir,
                )
            )

        records.sort(key=_record_sort_key, reverse=True)
        return records

    def list_summaries(
        self,
        *,
        camera: str | None = None,
        label_status: LabelStatus | None = None,
        date: str | None = None,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for record in self.scan():
            summary = self.summary(record)
            if not _matches_camera(camera, record, summary):
                continue
            if label_status is not None and summary["label_status"] != label_status.value:
                continue
            if date is not None and not _matches_date(date, record, summary):
                continue
            summaries.append(summary)
        return summaries

    def get_event(self, event_id: str) -> EventRecord | None:
        for record in self.scan():
            if record.event_id == event_id:
                return record
        return None

    def summary(self, record: EventRecord) -> dict[str, Any]:
        metadata = record.metadata
        frames = media_names(record, "frames")
        crops = media_names(record, "crops")
        thumbnail_url = None
        if crops:
            thumbnail_url = media_url(record.event_id, "crops", crops[0])
        elif frames:
            thumbnail_url = media_url(record.event_id, "frames", frames[0])

        camera_name = _optional_str(metadata.get("camera_name"))
        camera_id = _optional_str(metadata.get("camera_id"))
        camera = camera_name or camera_id or record.camera_dir or "unknown"
        protect_path = fixed_media_path(record, "protect_recording.mp4", missing_ok=True)

        return {
            "event_id": record.event_id,
            "camera": camera,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "utc_ts": metadata.get("utc_ts"),
            "trigger_reason": metadata.get("trigger_reason"),
            "classifier_guess": metadata.get("classifier_guess"),
            "classifier_confidence": metadata.get("classifier_confidence"),
            "label": metadata.get("label", Label.UNKNOWN.value),
            "label_status": metadata.get(
                "label_status",
                LabelStatus.UNLABELED.value,
            ),
            "multi_dog": bool(metadata.get("multi_dog", False)),
            "ambiguous": bool(metadata.get("ambiguous", False)),
            "dog": _optional_str(metadata.get("dog")),
            "thumbnail_url": thumbnail_url,
            "frames_count": len(frames),
            "crops_count": len(crops),
            "protect_recording_exists": protect_path is not None,
            "relative_dir": record.relative_dir,
        }

    def detail(self, record: EventRecord) -> dict[str, Any]:
        frames = media_names(record, "frames")
        crops = media_names(record, "crops")
        clip = fixed_media_path(record, "clip.mp4", missing_ok=True)
        protect = fixed_media_path(record, "protect_recording.mp4", missing_ok=True)

        return {
            "summary": self.summary(record),
            "metadata": record.metadata,
            "media": {
                "clip": fixed_media_url(record.event_id, "clip") if clip else None,
                "protect_recording": (
                    fixed_media_url(record.event_id, "protect") if protect else None
                ),
                "frames": [
                    {"name": name, "url": media_url(record.event_id, "frames", name)}
                    for name in frames
                ],
                "crops": [
                    {"name": name, "url": media_url(record.event_id, "crops", name)}
                    for name in crops
                ],
            },
        }

    def update_label(
        self,
        record: EventRecord,
        *,
        label: Label,
        label_status: LabelStatus,
        note: str | None,
        dog: str | None | object = _UNSET,
    ) -> dict[str, Any]:
        event_real = record.dir_path.resolve(strict=True)
        metadata_path = record.dir_path / "metadata.json"
        metadata_real = metadata_path.resolve(strict=True)
        metadata_real.relative_to(event_real)

        with metadata_real.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must contain an object")

        metadata["label"] = label.value
        metadata["label_status"] = label_status.value
        if dog is not _UNSET:
            metadata["dog"] = dog
        extra = metadata.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            metadata["extra"] = extra
        if note is not None:
            extra["label_note"] = note
        extra["labeled_at"] = datetime.now(timezone.utc).isoformat()

        tmp_path = record.dir_path / f".metadata.{os.getpid()}.{os.urandom(8).hex()}.tmp"
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, metadata_path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

        updated = EventRecord(
            event_id=record.event_id,
            dir_path=record.dir_path,
            metadata=metadata,
            relative_dir=record.relative_dir,
            camera_dir=record.camera_dir,
            date_dir=record.date_dir,
        )
        return self.summary(updated)


def media_names(record: EventRecord, kind: str) -> list[str]:
    base = record.dir_path / kind
    try:
        event_real = record.dir_path.resolve(strict=True)
        base_real = base.resolve(strict=True)
        base_real.relative_to(event_real)
    except (OSError, ValueError):
        return []
    if not base_real.is_dir():
        return []

    names: list[str] = []
    for path in base_real.iterdir():
        try:
            real_path = path.resolve(strict=True)
            real_path.relative_to(base_real)
        except (OSError, ValueError):
            continue
        if real_path.is_file():
            names.append(path.name)
    return sorted(names)


def media_path(record: EventRecord, kind: str, name: str) -> Path | None:
    if kind not in {"frames", "crops"}:
        raise ValueError("unsupported media collection")
    _validate_media_name(name)
    event_real = record.dir_path.resolve(strict=True)
    base = record.dir_path / kind
    try:
        base_real = base.resolve(strict=True)
        base_real.relative_to(event_real)
    except (OSError, ValueError):
        return None

    candidate = (base / name).resolve(strict=False)
    candidate.relative_to(base_real)
    if not candidate.is_file():
        return None
    return candidate


def fixed_media_path(
    record: EventRecord,
    filename: str,
    *,
    missing_ok: bool = False,
) -> Path | None:
    event_real = record.dir_path.resolve(strict=True)
    path = record.dir_path / filename
    try:
        real_path = path.resolve(strict=True)
        real_path.relative_to(event_real)
    except (OSError, ValueError):
        if missing_ok:
            return None
        raise
    if not real_path.is_file():
        return None
    return real_path


def media_url(event_id: str, kind: str, name: str) -> str:
    path_kind = "frames" if kind == "frames" else "crops"
    return f"/api/events/{quote(event_id, safe='')}/{path_kind}/{quote(name, safe='')}"


def fixed_media_url(event_id: str, kind: str) -> str:
    return f"/api/events/{quote(event_id, safe='')}/media/{kind}"


def _validate_media_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ValueError("invalid media filename")
    path = Path(name)
    if path.is_absolute() or path.name != name or "/" in name or "\\" in name:
        raise ValueError("invalid media filename")


def _record_sort_key(record: EventRecord) -> tuple[datetime, str]:
    return (_parse_timestamp(record.metadata.get("utc_ts")), record.dir_path.name)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _matches_camera(
    camera: str | None,
    record: EventRecord,
    summary: dict[str, Any],
) -> bool:
    if camera is None:
        return True
    wanted = camera.casefold()
    values = (
        summary.get("camera"),
        summary.get("camera_id"),
        summary.get("camera_name"),
        record.camera_dir,
    )
    return any(str(value).casefold() == wanted for value in values if value)


def _matches_date(date: str, record: EventRecord, summary: dict[str, Any]) -> bool:
    utc_ts = summary.get("utc_ts")
    return record.date_dir == date or (isinstance(utc_ts, str) and utc_ts.startswith(date))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
