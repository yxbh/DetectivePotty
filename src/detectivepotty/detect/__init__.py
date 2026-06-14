"""Detection implementations and shared detector protocols."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from detectivepotty.detect.yolo import FrameMeta
from detectivepotty.events import Detection


class FrameDetector(Protocol):
    """Single-frame detector surface shared by pipeline, harvest, and export."""

    def detect(
        self,
        frame_bgr_original: Any,
        frame_idx: int = 0,
        mono_ts: float | None = None,
        wall_ts: datetime | None = None,
    ) -> list[Detection]: ...


class BatchDetector(Protocol):
    """Batched detector surface used by dense scans and optional fast paths."""

    def detect_batch(
        self,
        frames: Sequence[Any],
        metas: Sequence[FrameMeta] | None = None,
    ) -> list[list[Any]]: ...


class BatchedFrameDetector(FrameDetector, BatchDetector, Protocol):
    """Detector that supports both single-frame and batched inference."""
