"""Frame-index timeline helpers for CFR and PTS-backed clips."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

TIME_BASIS_CLIP_FRAMES = "clip_frames"
TIME_BASIS_CLIP_PTS = "clip_pts"


@dataclass(frozen=True, slots=True)
class FrameTimeline:
    """Map stable frame indices to presentation seconds.

    ``frame_idx`` remains the identity used for labels, detections, crops, and
    filenames. This helper only owns the time mapping. Without ``frame_times_s``
    it preserves the existing CFR behavior exactly.
    """

    fps: float
    frame_count: int
    frame_times_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.frame_count < 0:
            raise ValueError("frame_count must be >= 0")
        if self.fps < 0:
            raise ValueError("fps must be >= 0")
        if self.frame_times_s is not None:
            normalized = _normalize_frame_times(self.frame_times_s, self.frame_count)
            object.__setattr__(self, "frame_times_s", normalized)
            object.__setattr__(self, "frame_count", len(normalized))

    @classmethod
    def cfr(cls, fps: float, frame_count: int) -> "FrameTimeline":
        return cls(fps=fps, frame_count=frame_count)

    @classmethod
    def from_frame_times(
        cls,
        frame_times_s: Sequence[float],
        *,
        fps: float,
    ) -> "FrameTimeline":
        return cls(fps=fps, frame_count=len(frame_times_s), frame_times_s=tuple(frame_times_s))

    @property
    def has_pts(self) -> bool:
        return self.frame_times_s is not None

    @property
    def duration_s(self) -> float:
        if self.frame_count <= 0:
            return 0.0
        if self.frame_times_s is not None:
            if self.frame_count == 1:
                return self.frame_times_s[0] + (1.0 / self._effective_fps)
            return self.frame_times_s[-1] + max(
                0.0,
                self.frame_times_s[-1] - self.frame_times_s[-2],
            )
        return self.frame_count / self._effective_fps

    @property
    def time_basis(self) -> str:
        return TIME_BASIS_CLIP_PTS if self.has_pts else TIME_BASIS_CLIP_FRAMES

    def frame_to_seconds(self, frame_idx: int) -> float:
        frame_idx = self._clamp_frame(frame_idx)
        if self.frame_times_s is not None:
            return self.frame_times_s[frame_idx]
        return frame_idx / self._effective_fps

    def seconds_to_frame_floor(self, seconds: float) -> int:
        if self.frame_count <= 0:
            return 0
        seconds = max(0.0, seconds)
        if self.frame_times_s is None:
            return self._clamp_frame(int(seconds * self._effective_fps))
        idx = bisect_right(self.frame_times_s, seconds) - 1
        return self._clamp_frame(idx)

    def seconds_to_frame_ceil(self, seconds: float) -> int:
        if self.frame_count <= 0:
            return 0
        seconds = max(0.0, seconds)
        if self.frame_times_s is None:
            return self._clamp_frame(int(_ceil_seconds(seconds * self._effective_fps)))
        return self._clamp_frame(bisect_left(self.frame_times_s, seconds))

    def seconds_to_frame_nearest(self, seconds: float) -> int:
        if self.frame_count <= 0:
            return 0
        seconds = max(0.0, seconds)
        if self.frame_times_s is None:
            return self._clamp_frame(round(seconds * self._effective_fps))
        right = bisect_left(self.frame_times_s, seconds)
        if right <= 0:
            return 0
        if right >= self.frame_count:
            return self.frame_count - 1
        left = right - 1
        if abs(self.frame_times_s[left] - seconds) <= abs(self.frame_times_s[right] - seconds):
            return left
        return right

    def sample_frames_by_time(
        self,
        start_frame: int,
        end_frame: int,
        *,
        stride_s: float,
        max_frames: int,
    ) -> list[int]:
        if end_frame < start_frame:
            return []
        start_frame = self._clamp_frame(start_frame)
        end_frame = self._clamp_frame(end_frame)
        if end_frame < start_frame:
            return []
        if stride_s <= 0:
            stride_s = 1.0 / self._effective_fps

        frames: list[int] = [start_frame]
        if self.frame_times_s is None:
            step = max(1, round(stride_s * self._effective_fps))
            frames = list(range(start_frame, end_frame + 1, step))
        else:
            end_s = self.frame_to_seconds(end_frame)
            target = self.frame_to_seconds(start_frame) + stride_s
            while target <= end_s + 1e-9:
                idx = self.seconds_to_frame_ceil(target)
                if start_frame <= idx <= end_frame and idx != frames[-1]:
                    frames.append(idx)
                target += stride_s

        if not frames:
            frames = [start_frame]
        if max_frames > 0 and len(frames) > max_frames:
            frames = _thin_unique(frames, max_frames)
        return frames

    @property
    def _effective_fps(self) -> float:
        return self.fps if self.fps > 0 else 30.0

    def _clamp_frame(self, frame_idx: int) -> int:
        if self.frame_count <= 0:
            return 0
        return max(0, min(self.frame_count - 1, int(frame_idx)))


def timeline_from_metadata(meta: dict[str, Any]) -> FrameTimeline:
    fps = float(meta.get("fps") or 0.0)
    frame_times = meta.get("frame_times_s")
    if isinstance(frame_times, list) and frame_times:
        return FrameTimeline.from_frame_times([float(v) for v in frame_times], fps=fps)
    return FrameTimeline.cfr(fps=fps, frame_count=int(meta.get("frame_count") or 0))


def clip_frame_times(
    source_frame_times_s: Sequence[float] | None,
    start_frame: int,
    end_frame: int,
) -> list[float] | None:
    if source_frame_times_s is None:
        return None
    if start_frame < 0 or end_frame < start_frame or end_frame >= len(source_frame_times_s):
        return None
    window = [float(v) for v in source_frame_times_s[start_frame : end_frame + 1]]
    if not window:
        return None
    base = window[0]
    return [round(value - base, 9) for value in window]


def maybe_pts_times(frame_times_s: Sequence[float], fps: float) -> tuple[float, ...] | None:
    """Return normalized PTS times only when they differ from CFR enough to matter."""

    if not frame_times_s:
        return None
    normalized = _normalize_frame_times(frame_times_s, len(frame_times_s))
    if fps <= 0:
        return normalized
    for idx, value in enumerate(normalized):
        if abs(value - (idx / fps)) > 1e-4:
            return normalized
    return None


def _normalize_frame_times(
    frame_times_s: Sequence[float],
    frame_count: int,
) -> tuple[float, ...]:
    if len(frame_times_s) != frame_count:
        raise ValueError("frame_times_s length must match frame_count")
    if not frame_times_s:
        return tuple()

    raw = [float(v) for v in frame_times_s]
    if any(not isfinite(value) for value in raw):
        raise ValueError("frame_times_s values must be finite")
    base = raw[0]
    normalized = tuple(round(value - base, 9) for value in raw)
    previous = normalized[0]
    if previous < -1e-9:
        raise ValueError("frame_times_s must normalize to non-negative times")
    for value in normalized[1:]:
        if value + 1e-9 < previous:
            raise ValueError("frame_times_s must be monotonic")
        previous = value
    return normalized


def _ceil_seconds(value: float) -> int:
    as_int = int(value)
    return as_int if abs(value - as_int) < 1e-9 else as_int + 1


def _thin_unique(frames: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    last = len(frames) - 1
    selected = {
        frames[round(i * last / (max_frames - 1))]
        for i in range(max_frames)
    }
    return sorted(selected)
