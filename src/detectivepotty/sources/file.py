"""File-backed video source implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import time
from typing import Any, Callable, Self

import cv2

from detectivepotty.sources.base import Frame, VideoSource
from detectivepotty.sources.pyav_capture import open_capture

# UniFi Protect exports embed the real recording time in the filename, e.g.
# ``Backyard Grass 6-6-2026, 19.46.40 GMT+10 - 6-6-2026, 19.47.05 GMT+10.mp4``.
# The first timestamp is the recording start. Dates are US ``M-D-YYYY``.
_FILENAME_TS_RE = re.compile(
    r"(\d{1,2})-(\d{1,2})-(\d{4}),\s*"
    r"(\d{1,2})\.(\d{2})\.(\d{2})\s*"
    r"GMT([+-]\d{1,2})(?::?(\d{2}))?"
)

# UniFi Protect / NVR exports and our chunk downloader name files with an
# ISO-8601 *basic* UTC stamp, e.g. ``<cameraId>_20260606T230000Z.mp4`` (8 date
# digits, ``T``, 6 time digits, optional ``Z`` or numeric offset). The GMT
# app-export regex above does not match this, which previously dropped the
# anchor to the file's mtime (the download time, not the recording time).
_FILENAME_ISO_TS_RE = re.compile(
    r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z|[+-]\d{2}:?\d{2})?"
)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_gmt_export_ts(name: str) -> datetime | None:
    match = _FILENAME_TS_RE.search(name)
    if match is None:
        return None
    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
    gmt_hours = int(match.group(7))
    gmt_minutes = int(match.group(8)) if match.group(8) else 0
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    signed_minutes = gmt_hours * 60 + (gmt_minutes if gmt_hours >= 0 else -gmt_minutes)
    try:
        tz = timezone(timedelta(minutes=signed_minutes))
        local = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def _parse_iso_basic_ts(name: str) -> datetime | None:
    match = _FILENAME_ISO_TS_RE.search(name)
    if match is None:
        return None
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    token = match.group(7)
    if token and token not in ("Z", "z"):
        body = token[1:].replace(":", "")
        offset = timedelta(hours=int(body[:2]), minutes=int(body[2:4]))
        tz = timezone(offset if token[0] == "+" else -offset)
    else:
        # No suffix or ``Z`` -> UTC. Protect's basic stamps are always UTC.
        tz = timezone.utc
    try:
        local = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def parse_filename_start_ts(name: str) -> datetime | None:
    """Parse the real recording-start UTC time from an export/recording filename.

    Recognizes two namings: the UniFi app-export ``M-D-YYYY, H.MM.SS GMT±H``
    form and the Protect/NVR ISO-8601 *basic* ``YYYYMMDDTHHMMSS[Z]`` form (used
    by the chunk downloader and historical exports). Returns ``None`` when the
    filename has no recognizable, in-range timestamp so the caller can fall back
    to a different anchor.
    """

    return _parse_gmt_export_ts(name) or _parse_iso_basic_ts(name)


def derive_base_wall_ts(
    path: str | Path,
    *,
    override: datetime | None = None,
) -> tuple[datetime, str]:
    """Derive a deterministic UTC anchor for a file's synthetic timeline.

    Priority (first that succeeds wins), with the chosen ``time_basis`` returned
    alongside the timestamp so callers can convey confidence to the UI:

    0. ``override`` -> ``"config"`` (an explicit recording-start the operator set)
    1. the recording time parsed from the filename -> ``"filename"``
    2. the file's modification time -> ``"file_mtime"`` (stable across reruns of
       an untouched file; lower confidence — not necessarily the recording time)
    3. ``datetime.now`` -> ``"runtime_now"`` (last resort; the only
       non-deterministic basis, used when the path cannot be stat-ed)
    """

    if override is not None:
        return _ensure_utc(override), "config"
    candidate = Path(path)
    parsed = parse_filename_start_ts(candidate.name)
    if parsed is not None:
        return parsed, "filename"
    try:
        mtime = candidate.stat().st_mtime
    except OSError:
        return datetime.now(timezone.utc), "runtime_now"
    return datetime.fromtimestamp(mtime, tz=timezone.utc), "file_mtime"


class FileSource(VideoSource):
    """Decode a local video file into ``Frame`` objects.

    File frames use a deterministic, real-recording-time UTC timeline: ``open()``
    derives a base UTC wall time from the source (filename recording time -> file
    mtime -> now; see :func:`derive_base_wall_ts`), then each emitted frame is
    timestamped as ``base_wall_ts + frame_idx / fps`` when the file FPS is known.
    Because the anchor is derived deterministically from the source, re-running
    detection over the same clip reproduces identical frame timestamps — which is
    what lets reruns be deduplicated instead of piling up.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        target_fps: float | None = None,
        base_wall_ts: datetime | None = None,
        capture_factory: Callable[[str], Any] = open_capture,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.path = Path(path)
        self.source_id = str(self.path)
        self.target_fps = target_fps
        self._explicit_base = base_wall_ts
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._fps: float | None = None
        self._resolution: tuple[int, int] | None = None
        self._base_wall_ts: datetime | None = None
        self._time_basis: str | None = None
        self._first_source_time_s: float | None = None
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
        if self._explicit_base is not None:
            self._base_wall_ts = _ensure_utc(self._explicit_base)
            self._time_basis = "explicit"
        else:
            self._base_wall_ts, self._time_basis = derive_base_wall_ts(self.path)
        self._frame_idx = 0
        self._decoded_idx = 0
        self._first_source_time_s = None
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
            source_time_s = self._source_time_s(decoded_idx, frame_idx)
            wall_ts = self._wall_ts_for_time(frame_idx, source_time_s)
            return Frame(
                bgr=bgr,
                frame_idx=frame_idx,
                mono_ts=mono_ts,
                wall_ts=wall_ts,
                source_id=self.source_id,
                source_time_s=source_time_s,
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
    def base_wall_ts(self) -> datetime | None:
        """The deterministic UTC anchor chosen for this source's timeline."""

        return self._base_wall_ts

    @property
    def time_basis(self) -> str | None:
        """How ``base_wall_ts`` was derived (``filename``/``file_mtime``/...)."""

        return self._time_basis

    @property
    def is_live(self) -> bool:
        return False

    def _wall_ts_for_time(self, frame_idx: int, source_time_s: float | None) -> datetime:
        if self._base_wall_ts is None:
            return datetime.now(timezone.utc)
        if source_time_s is not None:
            return self._base_wall_ts + timedelta(seconds=source_time_s)
        timeline_fps = self.target_fps or self._fps
        if timeline_fps is None:
            return datetime.now(timezone.utc)
        return self._base_wall_ts + timedelta(seconds=frame_idx / timeline_fps)

    def _source_time_s(self, decoded_idx: int, emitted_idx: int) -> float | None:
        if self._capture is not None:
            getter = getattr(self._capture, "get", None)
            if callable(getter):
                try:
                    value_ms = float(getter(cv2.CAP_PROP_POS_MSEC))
                except Exception:  # pragma: no cover - defensive capture shim
                    value_ms = 0.0
                if value_ms > 0 or decoded_idx == 0:
                    raw_s = max(0.0, value_ms / 1000.0)
                    if self._first_source_time_s is None:
                        self._first_source_time_s = raw_s
                    return max(0.0, raw_s - self._first_source_time_s)
        timeline_fps = self.target_fps or self._fps
        if timeline_fps is None:
            return None
        return emitted_idx / timeline_fps

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
