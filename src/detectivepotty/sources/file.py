"""File-backed video source implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Callable, Self

import cv2

from detectivepotty.sources.base import Frame, VideoSource


class FileSource(VideoSource):
    """Decode a local video file into ``Frame`` objects.

    File frames use a synthetic stable UTC wall-clock timeline: ``open()`` records
    a base UTC wall time, then each emitted frame is timestamped as
    ``base_wall_ts + frame_idx / fps`` when the file FPS is known.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        target_fps: float | None = None,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.path = Path(path)
        self.source_id = str(self.path)
        self.target_fps = target_fps
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._fps: float | None = None
        self._resolution: tuple[int, int] | None = None
        self._base_wall_ts: datetime | None = None
        self._frame_idx = 0
        self._decoded_idx = 0
        self._next_emit_s = 0.0

    def open(self) -> Self:
        if self._capture is not None:
            return self

        capture = self._capture_factory(str(self.path))
        if not self._is_capture_opened(capture):
            self._release_capture(capture)
            raise RuntimeError(f"failed to open video file: {self.path}")

        self._capture = capture
        self._fps = self._positive_capture_value(cv2.CAP_PROP_FPS)
        width = self._positive_capture_value(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._positive_capture_value(cv2.CAP_PROP_FRAME_HEIGHT)
        self._resolution = (
            (int(width), int(height)) if width is not None and height is not None else None
        )
        self._base_wall_ts = datetime.now(timezone.utc)
        self._frame_idx = 0
        self._decoded_idx = 0
        self._next_emit_s = 0.0
        return self

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("FileSource must be opened before read()")

        while True:
            ok, bgr = self._capture.read()
            if not ok or bgr is None:
                return None

            decoded_idx = self._decoded_idx
            self._decoded_idx += 1
            if not self._should_emit(decoded_idx):
                continue

            frame_idx = self._frame_idx
            self._frame_idx += 1
            mono_ts = time.monotonic()
            wall_ts = self._wall_ts_for_frame(frame_idx)
            return Frame(
                bgr=bgr,
                frame_idx=frame_idx,
                mono_ts=mono_ts,
                wall_ts=wall_ts,
                source_id=self.source_id,
            )

    def close(self) -> None:
        if self._capture is not None:
            self._release_capture(self._capture)
            self._capture = None

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution

    @property
    def is_live(self) -> bool:
        return False

    def _wall_ts_for_frame(self, frame_idx: int) -> datetime:
        if self._base_wall_ts is None:
            return datetime.now(timezone.utc)
        timeline_fps = self.target_fps or self._fps
        if timeline_fps is None:
            return datetime.now(timezone.utc)
        return self._base_wall_ts + timedelta(seconds=frame_idx / timeline_fps)

    def _should_emit(self, decoded_idx: int) -> bool:
        if self.target_fps is None or self._fps is None or self.target_fps >= self._fps:
            return True

        decoded_s = decoded_idx / self._fps
        if decoded_s + 1e-9 < self._next_emit_s:
            return False
        self._next_emit_s += 1.0 / self.target_fps
        return True

    def _positive_capture_value(self, prop: int) -> float | None:
        if self._capture is None:
            return None
        value = self._capture.get(prop)
        return float(value) if value and value > 0 else None

    @staticmethod
    def _is_capture_opened(capture: Any) -> bool:
        is_opened = getattr(capture, "isOpened", None)
        return bool(is_opened()) if callable(is_opened) else True

    @staticmethod
    def _release_capture(capture: Any) -> None:
        release = getattr(capture, "release", None)
        if callable(release):
            release()
