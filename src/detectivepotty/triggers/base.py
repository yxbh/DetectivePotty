"""Trigger event contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from detectivepotty.events import TriggerReason
from detectivepotty.geometry import BBox


@dataclass(slots=True)
class TriggerEvent:
    camera_id: str
    reason: TriggerReason
    detection_ts: datetime
    notification_ts: datetime
    score: float | None = None
    bbox: BBox | None = None
    raw: Mapping[str, Any] | None = None


class Trigger(ABC):
    """Async stream of trigger events.

    Implementations should yield deduplicated events in chronological order.
    Protect-backed triggers may reconnect internally; YOLO triggers may be fed by
    a warm ``VideoSource``. ``bbox`` is always original-resolution when present.
    """

    @abstractmethod
    def events(self) -> AsyncIterator[TriggerEvent]:
        raise NotImplementedError
