"""UniFi Protect smart-detect trigger."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import is_dataclass
from datetime import datetime, timezone
import logging
from typing import Any

from detectivepotty.events import TriggerReason
from detectivepotty.geometry import BBox
from detectivepotty.protect.client import ProtectClient
from detectivepotty.triggers.base import Trigger, TriggerEvent

LOGGER = logging.getLogger(__name__)
MAX_DEDUP_KEYS = 4096


class ProtectAnimalTrigger(Trigger):
    """Yield one trigger per UniFi Protect animal smart-detect event."""

    def __init__(
        self,
        client: ProtectClient,
        *,
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 30.0,
    ) -> None:
        self.client = client
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self._seen: set[str] = set()
        self._seen_order: deque[str] = deque()

    async def events(self) -> AsyncIterator[TriggerEvent]:
        backoff = self.reconnect_initial_s
        while True:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            state_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
            unsubscribe = _noop
            unsubscribe_state = _noop
            try:
                await self.client.connect()

                def on_packet(packet: Any) -> None:
                    queue.put_nowait(packet)

                def on_state(state: Any) -> None:
                    if _string_value(state).lower() in {"false", "disconnected"}:
                        try:
                            state_queue.put_nowait(state)
                        except asyncio.QueueFull:
                            pass

                unsubscribe = self.client.subscribe_websocket(on_packet)
                unsubscribe_state = self.client.subscribe_websocket_state(on_state)
                backoff = self.reconnect_initial_s

                while True:
                    packet_task = asyncio.create_task(queue.get())
                    state_task = asyncio.create_task(state_queue.get())
                    done, pending = await asyncio.wait(
                        {packet_task, state_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    if state_task in done:
                        raise ConnectionError("Protect websocket disconnected")

                    packet = packet_task.result()
                    event = parse_smartdetect_event(packet)
                    if event is not None and self._remember(event):
                        yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning(
                    "Protect websocket unavailable; reconnecting in %.1fs: %s",
                    backoff,
                    exc.__class__.__name__,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_max_s)
            finally:
                unsubscribe()
                unsubscribe_state()
                await self.client.close()

    def _remember(self, event: TriggerEvent) -> bool:
        key = _dedup_key(event)
        if key in self._seen:
            return False
        self._seen.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > MAX_DEDUP_KEYS:
            self._seen.discard(self._seen_order.popleft())
        return True


def _noop() -> None:
    return None


def parse_smartdetect_event(
    ws_packet: Any,
    *,
    received_at: datetime | None = None,
) -> TriggerEvent | None:
    """Parse one uiprotect WS packet into a sanitized animal trigger event."""

    action = _packet_action(ws_packet)
    if action == "remove":
        return None

    payload = _packet_payload(ws_packet)
    if payload is None:
        return None

    smart_types = _smart_detect_types(payload)
    if "animal" not in smart_types:
        return None

    camera_id = _first_value(payload, "camera_id", "cameraId", "camera", "device_id", "device")
    if camera_id is None:
        return None

    detection_ts = _coerce_datetime(_first_value(payload, "start", "timestamp", "time"))
    if detection_ts is None:
        return None

    event_id = _first_value(payload, "id", "event_id", "eventId") or _packet_update_id(ws_packet)
    score = _score(payload)
    bbox = _bbox(payload)
    notification_ts = _ensure_aware_utc(received_at or datetime.now(timezone.utc))
    raw = {
        "protect_event_id": str(event_id) if event_id is not None else None,
        "action": action,
        "smart_detect_types": sorted(smart_types),
    }
    if score is not None:
        raw["score"] = score
    raw = {key: value for key, value in raw.items() if value is not None}

    return TriggerEvent(
        camera_id=str(camera_id),
        reason=TriggerReason.PROTECT_ANIMAL,
        detection_ts=detection_ts,
        notification_ts=notification_ts,
        score=score,
        bbox=bbox,
        raw=raw,
    )


def _dedup_key(event: TriggerEvent) -> str:
    raw = event.raw or {}
    protect_event_id = raw.get("protect_event_id")
    if protect_event_id:
        return f"protect:{protect_event_id}:animal"
    return f"camera:{event.camera_id}:{event.detection_ts.isoformat()}:animal"


def _packet_action(packet: Any) -> str | None:
    action = _get_value(packet, "action")
    return _string_value(action).lower() if action is not None else None


def _packet_update_id(packet: Any) -> str | None:
    value = _get_value(packet, "new_update_id") or _get_value(packet, "newUpdateId")
    return str(value) if value is not None else None


def _packet_payload(packet: Any) -> Any | None:
    for key in ("new_obj", "newObj", "new", "payload", "data"):
        value = _get_value(packet, key)
        if value is not None:
            return value
    if isinstance(packet, Mapping):
        return packet
    return _get_value(packet, "old_obj") or _get_value(packet, "oldObj")


def _smart_detect_types(payload: Any) -> set[str]:
    raw = _first_value(payload, "smart_detect_types", "smartDetectTypes", "smartDetectTypesV2")
    if raw is None:
        raw = []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    types = {_string_value(item).lower() for item in raw if item is not None}

    for item in _smart_items(payload):
        object_type = _first_value(item, "object_type", "objectType", "type", "name")
        if object_type is not None:
            types.add(_string_value(object_type).lower())
    return types


def _score(payload: Any) -> float | None:
    for value in (
        _first_value(payload, "score", "confidence"),
        _thumbnail_value(payload, "confidence"),
        _smart_item_value(payload, "confidence"),
    ):
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score > 1.0:
            score /= 100.0
        return score
    return None


def _bbox(payload: Any) -> BBox | None:
    for value in (
        _first_value(payload, "coord", "coords", "bbox", "boundingBox"),
        _thumbnail_value(payload, "coord"),
        _smart_item_value(payload, "coord"),
    ):
        bbox = _coerce_bbox(value)
        if bbox is not None:
            return bbox
    return None


def _thumbnail_value(payload: Any, key: str) -> Any | None:
    metadata = _get_value(payload, "metadata")
    thumbnails = _get_value(metadata, "detected_thumbnails") or _get_value(metadata, "detectedThumbnails")
    if thumbnails is None:
        return None
    for thumbnail in thumbnails:
        thumb_type = _first_value(thumbnail, "type", "object_type", "objectType", "name")
        if thumb_type is not None and _string_value(thumb_type).lower() != "animal":
            continue
        value = _get_value(thumbnail, key)
        if value is not None:
            return value
    return None


def _smart_item_value(payload: Any, key: str) -> Any | None:
    for item in _smart_items(payload):
        object_type = _first_value(item, "object_type", "objectType", "type", "name")
        if object_type is not None and _string_value(object_type).lower() != "animal":
            continue
        value = _get_value(item, key)
        if value is not None:
            return value
    return None


def _smart_items(payload: Any) -> list[Any]:
    candidates = [
        _get_value(payload, "payload"),
        _get_value(payload, "smart_detect_items"),
        _get_value(payload, "smartDetectItems"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
    return []


def _coerce_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if isinstance(value, BBox):
        return value
    if isinstance(value, Mapping):
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return BBox(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
        if {"left", "top", "right", "bottom"}.issubset(value):
            return BBox(
                float(value["left"]),
                float(value["top"]),
                float(value["right"]),
                float(value["bottom"]),
            )
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return BBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware_utc(value)
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if value > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, timezone.utc)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return _ensure_aware_utc(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _first_value(payload: Any, *keys: str) -> Any | None:
    for key in keys:
        value = _get_value(payload, key)
        if value is not None:
            return value
    return None


def _get_value(obj: Any, key: str) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    if is_dataclass(obj) and not isinstance(obj, type) and hasattr(obj, key):
        return getattr(obj, key)
    return getattr(obj, key, None)


def _string_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
