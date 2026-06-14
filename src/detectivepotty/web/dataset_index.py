"""Dataset scanning and label updates for the local web app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import threading
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
        self._lock = threading.Lock()
        self._event_dirs: dict[str, Path] = {}

    def scan(self) -> list[EventRecord]:
        if not self.dataset_dir.exists():
            with self._lock:
                self._event_dirs = {}
            return []

        # Events live at a fixed depth (<camera>/<date>/events/<event>/), so a
        # depth-bounded glob avoids descending into the large frames/crops dirs
        # that rglob would walk on every request.
        root = self.dataset_dir
        records: list[EventRecord] = []
        for metadata_path in root.glob("*/*/events/*/metadata.json"):
            record = _record_from_event_dir(metadata_path.parent, root)
            if record is not None:
                records.append(record)

        records.sort(key=_record_sort_key, reverse=True)
        event_dirs: dict[str, Path] = {}
        for record in records:
            event_dirs.setdefault(record.event_id, record.dir_path)
        with self._lock:
            self._event_dirs = event_dirs
        return records

    def list_summaries(
        self,
        *,
        camera: str | None = None,
        label_status: LabelStatus | None = None,
        date: str | None = None,
        records: list[EventRecord] | None = None,
    ) -> list[dict[str, Any]]:
        if records is None:
            records = self.scan()
        summaries: list[dict[str, Any]] = []
        for record in records:
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
        with self._lock:
            event_dir = self._event_dirs.get(event_id)
        if event_dir is not None:
            record = _record_from_event_dir(event_dir, self.dataset_dir)
            if record is not None and record.event_id == event_id:
                return record
        for record in self.scan():
            if record.event_id == event_id:
                return record
        return None

    def summary(self, record: EventRecord) -> dict[str, Any]:
        metadata = record.metadata
        frames_count, first_frame = _count_and_first(record, "frames")
        crops_count, first_crop = _count_and_first(record, "crops")
        thumbnail_url = None
        if first_crop is not None:
            thumbnail_url = media_url(record.event_id, "crops", first_crop)
        elif first_frame is not None:
            thumbnail_url = media_url(record.event_id, "frames", first_frame)

        camera_name = _optional_str(metadata.get("camera_name"))
        camera_id = _optional_str(metadata.get("camera_id"))
        camera = camera_name or camera_id or record.camera_dir or "unknown"
        return {
            "event_id": record.event_id,
            "camera": camera,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "utc_ts": metadata.get("utc_ts"),
            "end_ts": metadata.get("end_ts"),
            "recorded_at": metadata.get("recorded_at"),
            "source_start_s": metadata.get("source_start_s"),
            "source_end_s": metadata.get("source_end_s"),
            "time_basis": metadata.get("time_basis"),
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
            "frames_count": frames_count,
            "crops_count": crops_count,
            "relative_dir": record.relative_dir,
            "media_version": _media_version(record),
        }

    def detail(self, record: EventRecord) -> dict[str, Any]:
        frames = media_names(record, "frames")
        crops = media_names(record, "crops")
        crops_overlay = media_names(record, "crops_overlay")
        clip = fixed_media_path(record, "clip.mp4", missing_ok=True)

        return {
            "summary": self.summary(record),
            "metadata": record.metadata,
            "media": {
                "clip": fixed_media_url(record.event_id, "clip") if clip else None,
                "frames": [
                    {"name": name, "url": media_url(record.event_id, "frames", name)}
                    for name in frames
                ],
                "crops": [
                    {"name": name, "url": media_url(record.event_id, "crops", name)}
                    for name in crops
                ],
                "crops_overlay": [
                    {
                        "name": name,
                        "url": media_url(record.event_id, "crops_overlay", name),
                    }
                    for name in crops_overlay
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


def _record_from_event_dir(event_dir: Path, root: Path) -> EventRecord | None:
    """Build a validated :class:`EventRecord` from a single event directory.

    Shared by :meth:`DatasetIndex.scan` (full walk) and the
    :meth:`DatasetIndex.get_event` fast path so both apply the same
    metadata/path-containment validation.
    """

    if event_dir.parent.name != "events":
        return None
    metadata_path = event_dir / "metadata.json"
    try:
        metadata_real = metadata_path.resolve(strict=True)
        metadata_real.relative_to(root)
        with metadata_real.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    try:
        relative = event_dir.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return None

    parts = relative.parts
    camera_dir = parts[0] if len(parts) >= 4 else None
    date_dir = parts[1] if len(parts) >= 4 else None
    event_id = str(metadata.get("event_id") or event_dir.name)
    return EventRecord(
        event_id=event_id,
        dir_path=event_dir,
        metadata=metadata,
        relative_dir=relative.as_posix(),
        camera_dir=camera_dir,
        date_dir=date_dir,
    )


def _media_version(record: EventRecord) -> int:
    """Cache-busting token that changes whenever an event's media is rewritten.

    Reruns supersede media under a reused event_id, so a fixed per-event cache
    key would serve stale clips/crops. metadata.json is rewritten on every
    (re)record, so its mtime is a cheap, monotonic-enough freshness token the
    frontend appends as ``?v=`` to media URLs.
    """

    try:
        return (record.dir_path / "metadata.json").stat().st_mtime_ns
    except OSError:
        return 0


def _count_and_first(record: EventRecord, kind: str) -> tuple[int, str | None]:
    """Return (file count, lexicographically-first name) for a media subdir.

    Cheap variant of :func:`media_names` for the list view: a single
    ``os.scandir`` with no per-file ``resolve`` so counting hundreds of frames
    across every event stays fast.
    """

    base = record.dir_path / kind
    try:
        event_real = record.dir_path.resolve(strict=True)
        base_real = base.resolve(strict=True)
        base_real.relative_to(event_real)
    except (OSError, ValueError):
        return (0, None)
    if not base_real.is_dir():
        return (0, None)

    count = 0
    first: str | None = None
    try:
        with os.scandir(base_real) as entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                count += 1
                if first is None or entry.name < first:
                    first = entry.name
    except OSError:
        return (0, None)
    return (count, first)


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
    try:
        with os.scandir(base_real) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        names.append(entry.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(names)


def media_path(record: EventRecord, kind: str, name: str) -> Path | None:
    if kind not in {"frames", "crops", "crops_overlay"}:
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
    path_kind = kind if kind in {"frames", "crops", "crops_overlay"} else "crops"
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
