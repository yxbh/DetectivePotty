"""Thread-safe rolling pre-roll frame buffer."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
from typing import Deque

from detectivepotty.sources.base import Frame, VideoSource


class RollingBuffer:
    """Keep recent per-camera frames for pre-roll retrieval.

    Appends evict by monotonic timestamp relative to the newest appended frame;
    wall-clock queries use ``Frame.wall_ts`` because external triggers carry
    wall-clock detection timestamps. Returned frames are in append order.
    """

    def __init__(self, window_s: float, *, max_frames: int | None = None) -> None:
        if window_s < 0:
            raise ValueError("window_s must be non-negative")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.window_s = window_s
        self.max_frames = max_frames
        self._frames: Deque[Frame] = deque()
        self._lock = threading.RLock()

    def append(self, frame: Frame) -> None:
        with self._lock:
            self._frames.append(frame)
            self._evict_locked(frame.mono_ts)

    def get_last(self, seconds: float) -> list[Frame]:
        """Return frames from the last ``seconds`` by monotonic timestamp."""

        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        with self._lock:
            if not self._frames:
                return []
            cutoff = self._frames[-1].mono_ts - seconds
            return [frame for frame in self._frames if frame.mono_ts >= cutoff]

    def get_window(self, start_ts: datetime, end_ts: datetime) -> list[Frame]:
        """Return frames whose UTC wall timestamps are in ``[start_ts, end_ts]``."""

        start_utc = self._to_utc(start_ts, "start_ts")
        end_utc = self._to_utc(end_ts, "end_ts")
        if start_utc > end_utc:
            return []
        with self._lock:
            return [
                frame
                for frame in self._frames
                if start_utc <= frame.wall_ts <= end_utc
            ]

    def snapshot(self) -> list[Frame]:
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def _evict_locked(self, newest_mono_ts: float) -> None:
        cutoff = newest_mono_ts - self.window_s
        while self._frames and self._frames[0].mono_ts < cutoff:
            self._frames.popleft()
        if self.max_frames is not None:
            while len(self._frames) > self.max_frames:
                self._frames.popleft()

    @staticmethod
    def _to_utc(value: datetime, name: str) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        return value.astimezone(timezone.utc)


class BufferedSourceWorker:
    """Pump a ``VideoSource`` into a ``RollingBuffer`` on a daemon thread."""

    def __init__(
        self,
        source: VideoSource,
        buffer: RollingBuffer,
        *,
        poll_s: float = 0.01,
        name: str | None = None,
    ) -> None:
        if poll_s < 0:
            raise ValueError("poll_s must be non-negative")
        self.source = source
        self.buffer = buffer
        self.poll_s = poll_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._name = name or "buffered-source-worker"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    def _run(self) -> None:
        last_key: tuple[str, int] | None = None
        try:
            self.source.open()
            while not self._stop_event.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                key = (frame.source_id, frame.frame_idx)
                if key != last_key:
                    self.buffer.append(frame)
                    last_key = key
                if self.poll_s:
                    self._stop_event.wait(self.poll_s)
        finally:
            self.source.close()
