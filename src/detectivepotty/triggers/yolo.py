"""YOLO fallback/corroboration trigger."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from detectivepotty.detect.yolo import DogDetector
from detectivepotty.events import Detection, TriggerReason
from detectivepotty.sources.base import Frame, VideoSource
from detectivepotty.triggers.base import Trigger, TriggerEvent


class YoloTrigger(Trigger):
    """Sample a warm ``VideoSource`` and yield dog-appearance trigger events."""

    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        detector: DogDetector,
        sample_rate_frames: int = 1,
        min_trigger_interval_s: float = 0.0,
    ) -> None:
        if sample_rate_frames <= 0:
            raise ValueError("sample_rate_frames must be positive")
        if min_trigger_interval_s < 0.0:
            raise ValueError("min_trigger_interval_s must be non-negative")
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.sample_rate_frames = sample_rate_frames
        self.min_trigger_interval_s = min_trigger_interval_s
        self._last_trigger_mono: float | None = None

    async def events(self) -> AsyncIterator[TriggerEvent]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.source.open)
        try:
            while True:
                frame = await loop.run_in_executor(None, self.source.read)
                if frame is None:
                    break
                if frame.frame_idx % self.sample_rate_frames != 0:
                    continue
                detections = await loop.run_in_executor(
                    None,
                    self._detect_frame,
                    frame,
                )
                if not detections:
                    continue
                if not self._should_emit(frame.mono_ts):
                    continue
                top = detections[0]
                self._last_trigger_mono = frame.mono_ts
                yield TriggerEvent(
                    camera_id=self.camera_id,
                    reason=TriggerReason.YOLO,
                    detection_ts=top.wall_ts,
                    notification_ts=datetime.now(timezone.utc),
                    score=top.confidence,
                    bbox=top.bbox,
                    raw=self._raw_payload(frame, detections),
                )
        finally:
            await loop.run_in_executor(None, self.source.close)

    def _detect_frame(self, frame: Frame) -> list[Detection]:
        return self.detector.detect(
            frame.bgr,
            frame_idx=frame.frame_idx,
            mono_ts=frame.mono_ts,
            wall_ts=frame.wall_ts,
        )

    def _should_emit(self, mono_ts: float) -> bool:
        if self._last_trigger_mono is None:
            return True
        elapsed = mono_ts - self._last_trigger_mono
        return elapsed >= self.min_trigger_interval_s

    @staticmethod
    def _raw_payload(frame: Frame, detections: list[Detection]) -> dict[str, Any]:
        return {
            "frame_idx": frame.frame_idx,
            "source_id": frame.source_id,
            "detections": [detection.to_dict() for detection in detections],
        }
