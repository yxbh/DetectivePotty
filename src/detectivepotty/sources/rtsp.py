"""RTSP/RTSPS live video source implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import threading
import time
from typing import Any, Callable, Self

import cv2

from detectivepotty.sources.base import Frame, VideoSource, sanitize_source_id


class RTSPSource(VideoSource):
    """Continuously decode an RTSP/RTSPS stream and expose only the latest frame."""

    def __init__(
        self,
        url: str,
        *,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
        stale_timeout_s: float = 5.0,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 30.0,
        read_retry_s: float = 0.02,
        join_timeout_s: float = 2.0,
    ) -> None:
        if stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")
        if reconnect_initial_s <= 0 or reconnect_max_s <= 0:
            raise ValueError("reconnect backoff values must be positive")
        if read_retry_s <= 0:
            raise ValueError("read_retry_s must be positive")

        self._url = url
        self.source_id = sanitize_source_id(url)
        self._capture_factory = capture_factory
        self._stale_timeout_s = stale_timeout_s
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._read_retry_s = read_retry_s
        self._join_timeout_s = join_timeout_s

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any | None = None
        self._latest_frame: Frame | None = None
        self._latest_mono_ts: float | None = None
        self._frame_idx = 0
        self._fps: float | None = None
        self._resolution: tuple[int, int] | None = None
        self._closed = True

    def open(self) -> Self:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._latest_frame = None
            self._latest_mono_ts = None
            self._frame_idx = 0
            self._closed = False
            self._thread = threading.Thread(
                target=self._reader_loop,
                name=f"rtsp-source-{self.source_id}",
                daemon=True,
            )
            self._thread.start()
        return self

    def read(self) -> Frame | None:
        with self._condition:
            if self._closed:
                return None
            while self._latest_frame is None and not self._closed:
                self._condition.wait(timeout=self._read_retry_s)
            if self._closed:
                return None
            return self._latest_frame

    def close(self) -> None:
        thread: threading.Thread | None
        capture: Any | None
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            capture = self._capture
            thread = self._thread
            self._condition.notify_all()

        if capture is not None:
            self._release_capture(capture)
        if thread is not None:
            thread.join(timeout=self._join_timeout_s)

        with self._condition:
            self._capture = None
            self._thread = None
            self._condition.notify_all()

    @property
    def fps(self) -> float | None:
        with self._condition:
            return self._fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        with self._condition:
            return self._resolution

    @property
    def is_live(self) -> bool:
        return True

    def _reader_loop(self) -> None:
        backoff_s = self._reconnect_initial_s
        while not self._stop_event.is_set():
            capture = self._create_capture()
            if not self._is_capture_opened(capture):
                self._release_capture(capture)
                backoff_s = self._sleep_backoff(backoff_s)
                continue

            self._set_capture(capture)
            self._refresh_capture_metadata(capture)
            last_frame_mono = time.monotonic()
            backoff_s = self._reconnect_initial_s

            while not self._stop_event.is_set():
                ok, bgr = self._decode_next(capture)
                now = time.monotonic()
                if ok and bgr is not None:
                    last_frame_mono = now
                    self._publish_frame(bgr, now)
                    continue

                if now - last_frame_mono >= self._stale_timeout_s:
                    break
                self._stop_event.wait(self._read_retry_s)

            self._release_capture(capture)
            self._clear_capture(capture)
            if not self._stop_event.is_set():
                backoff_s = self._sleep_backoff(backoff_s)

        with self._condition:
            self._condition.notify_all()

    def _create_capture(self) -> Any:
        return self._capture_factory(self._url)

    def _decode_next(self, capture: Any) -> tuple[bool, Any | None]:
        return capture.read()

    def _publish_frame(self, bgr: Any, mono_ts: float) -> None:
        frame = Frame(
            bgr=bgr,
            frame_idx=self._frame_idx,
            mono_ts=mono_ts,
            wall_ts=datetime.now(timezone.utc),
            source_id=self.source_id,
        )
        with self._condition:
            self._frame_idx += 1
            self._latest_frame = frame
            self._latest_mono_ts = mono_ts
            self._condition.notify_all()

    def _refresh_capture_metadata(self, capture: Any) -> None:
        fps = self._capture_value(capture, cv2.CAP_PROP_FPS)
        width = self._capture_value(capture, cv2.CAP_PROP_FRAME_WIDTH)
        height = self._capture_value(capture, cv2.CAP_PROP_FRAME_HEIGHT)
        resolution = (
            (int(width), int(height)) if width is not None and height is not None else None
        )
        with self._condition:
            self._fps = fps
            self._resolution = resolution

    def _set_capture(self, capture: Any) -> None:
        with self._condition:
            self._capture = capture

    def _clear_capture(self, capture: Any) -> None:
        with self._condition:
            if self._capture is capture:
                self._capture = None

    def _sleep_backoff(self, backoff_s: float) -> float:
        self._stop_event.wait(backoff_s)
        return min(backoff_s * 2, self._reconnect_max_s)

    @staticmethod
    def _capture_value(capture: Any, prop: int) -> float | None:
        getter = getattr(capture, "get", None)
        if not callable(getter):
            return None
        value = getter(prop)
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None

    @staticmethod
    def _is_capture_opened(capture: Any) -> bool:
        is_opened = getattr(capture, "isOpened", None)
        return bool(is_opened()) if callable(is_opened) else True

    @staticmethod
    def _release_capture(capture: Any) -> None:
        release = getattr(capture, "release", None)
        if callable(release):
            release()
