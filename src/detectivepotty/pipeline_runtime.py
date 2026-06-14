"""Shared helper types and pure utilities for pipeline orchestration."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Protocol
from urllib.parse import urlsplit

from detectivepotty.classify.base import PottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.detect.yolo import FrameMeta
from detectivepotty.events import Detection, Track
from detectivepotty.potty_event import PottyCandidate, PottyEventDetector
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.sources.base import Frame, VideoSource

# Assumed worst-case decode rate used to bound per-camera live buffer memory.
# 60fps keeps pre-roll intact for common high-fps cameras while still bounding
# long-running live buffers when the source cannot report its decode rate up front.
_LIVE_ASSUMED_MAX_FPS = 60.0


class Detector(Protocol):
    def detect(
        self,
        frame_bgr: Any,
        *,
        frame_idx: int,
        mono_ts: float,
        wall_ts: datetime,
    ) -> list[Detection]: ...


DetectorFactory = Callable[..., Detector]
ClassifierFactory = Callable[..., PottyClassifier]
FileSourceFactory = Callable[[CameraConfig], VideoSource]
RTSPSourceFactory = Callable[[str], VideoSource]
StateMachineFactory = Callable[[CameraConfig], PottyEventDetector]
RecorderFactory = Callable[[Config, Any | None], EventRecorder]


def is_live_kind(kind: str) -> bool:
    """Live cameras stream forever; file cameras are finite."""

    return kind in ("protect", "rtsp")


def is_valid_rtsp_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in ("rtsp", "rtsps") and bool(parts.hostname)


def redact_url(message: str, url: str) -> str:
    """Keep a resolved RTSP URL (which embeds credentials) out of log text."""

    return message.replace(url, "<rtsp-url>") if url else message


def detect_frames_batched(
    detector: Detector,
    frames: Sequence[Frame],
    *,
    lock: Any,
) -> list[list[Detection]]:
    """Detect over ``frames`` in one batched forward, holding ``lock`` once.

    Falls back to a per-frame loop when the detector predates ``detect_batch``
    (e.g. a test fake), keeping the result identical — detections are per-image-
    independent, so the batched and per-frame results match frame-for-frame.
    Returns one detection list per input frame, in order.
    """

    if not frames:
        return []
    batch = getattr(detector, "detect_batch", None)
    with lock:
        if batch is not None:
            metas = [
                FrameMeta(frame_idx=f.frame_idx, mono_ts=f.mono_ts, wall_ts=f.wall_ts)
                for f in frames
            ]
            return batch([f.bgr for f in frames], metas)
        return [
            detector.detect(
                f.bgr,
                frame_idx=f.frame_idx,
                mono_ts=f.mono_ts,
                wall_ts=f.wall_ts,
            )
            for f in frames
        ]


@dataclass(slots=True)
class PendingCandidate:
    candidate: PottyCandidate
    protect_meta: dict[str, Any] | None = None


class FrameHistory:
    def __init__(self, window_s: float, *, max_frames: int | None = None) -> None:
        self.window_s = max(0.0, window_s)
        self.max_frames = max_frames
        self._frames: deque[Frame] = deque()

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)
        cutoff = frame.mono_ts - self.window_s
        while self._frames and self._frames[0].mono_ts < cutoff:
            self._frames.popleft()
        if self.max_frames is not None:
            while len(self._frames) > self.max_frames:
                self._frames.popleft()

    def snapshot(self) -> list[Frame]:
        return list(self._frames)

    def by_wall(self, start: datetime, end: datetime) -> list[Frame]:
        return [frame for frame in self._frames if start <= frame.wall_ts <= end]

    def by_mono(self, start: float, end: float) -> list[Frame]:
        return [frame for frame in self._frames if start <= frame.mono_ts <= end]


def call_camera_factory(
    factory: Callable[..., Any],
    camera_config: CameraConfig,
    config: Config,
) -> Any:
    try:
        return factory(camera_config, config)
    except TypeError as original_exc:
        try:
            return factory(camera_config)
        except TypeError:
            raise original_exc


def buffer_window_s(camera_config: CameraConfig) -> float:
    pre_event_window_s = max(
        camera_config.stationary_threshold_s,
        camera_config.dwell_trigger_s,
    )
    return max(
        1.0,
        camera_config.pre_roll_s
        + pre_event_window_s
        + camera_config.event_duration_s
        + camera_config.post_roll_s
        + 2.0,
    )


def history_max_frames(source_fps: float, window_s: float) -> int:
    return max(1, math.ceil(source_fps * window_s) + 2)


def live_buffer_max_frames(window_s: float) -> int:
    # Bound per-camera warm-buffer memory; without a cap N concurrent live
    # cameras decoding high-fps streams could exhaust memory.
    return max(1, math.ceil(window_s * _LIVE_ASSUMED_MAX_FPS) + 2)


def source_fps(source: VideoSource, camera_config: CameraConfig) -> float:
    fps = source.fps or camera_config.sample_rate_fps
    return fps if fps > 0 else camera_config.sample_rate_fps


def sample_every(source_fps: float, sample_rate_fps: float) -> int:
    if source_fps <= 0 or sample_rate_fps <= 0:
        return 1
    return max(1, round(source_fps / sample_rate_fps))


def retimestamp_file_frame(frame: Frame, base_mono: float, fps: float) -> Frame:
    # File decoding is faster than real time; use the file timeline for state durations.
    synthetic_mono = base_mono + frame.frame_idx / fps
    return Frame(
        bgr=frame.bgr,
        frame_idx=frame.frame_idx,
        mono_ts=synthetic_mono,
        wall_ts=frame.wall_ts,
        source_id=frame.source_id,
    )


def primary_track(candidate: PottyCandidate) -> Track | None:
    for track in candidate.tracks:
        if track.track_id == candidate.primary_track_id:
            return track
    return candidate.tracks[0] if candidate.tracks else None


def candidate_mono_bounds(
    candidate: PottyCandidate,
    camera_config: CameraConfig,
) -> tuple[float, float] | None:
    mono_values = [detection.mono_ts for detection in candidate_detections(candidate)]
    if not mono_values:
        return None
    start = min(mono_values) - camera_config.pre_roll_s
    end = max(mono_values) + camera_config.post_roll_s + camera_config.event_duration_s
    return start, end


def candidate_detections(candidate: PottyCandidate) -> Iterable[Detection]:
    yield from candidate.detections
    for track in candidate.tracks:
        yield from track.detections


def dedupe_frames(frames: Sequence[Frame]) -> list[Frame]:
    seen: set[tuple[str, int]] = set()
    unique: list[Frame] = []
    for frame in frames:
        key = (frame.source_id, frame.frame_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
    return unique
