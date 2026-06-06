"""OpenCV MP4 writing for recorded event windows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from detectivepotty.sources.base import Frame

DEFAULT_FPS = 5.0


def write_frames_to_mp4(
    frames: Sequence[Frame],
    dest: str | Path,
    *,
    fps: float | None = None,
) -> Path:
    """Write original-resolution BGR frames to an MP4 and return its path."""

    if not frames:
        raise ValueError("frames must not be empty")

    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    width, height = _frame_size(frames[0])
    writer = _open_video_writer(target, width, height, _choose_fps(frames, fps))
    try:
        for frame in frames:
            image = _normalize_bgr(frame.bgr)
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()

    if not target.exists() or target.stat().st_size == 0:
        raise OSError(f"failed to write clip: {target}")
    return target


def _open_video_writer(
    target: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open video writer: {target}")
    return writer


def _choose_fps(frames: Sequence[Frame], fps: float | None) -> float:
    if fps is not None and fps > 0:
        return float(fps)

    deltas = [
        later.mono_ts - earlier.mono_ts
        for earlier, later in zip(frames, frames[1:])
        if later.mono_ts > earlier.mono_ts
    ]
    if not deltas:
        return DEFAULT_FPS
    cadence = median(deltas)
    if cadence <= 0:
        return DEFAULT_FPS
    return max(0.1, min(240.0, 1.0 / cadence))


def _frame_size(frame: Frame) -> tuple[int, int]:
    image = _normalize_bgr(frame.bgr)
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    return width, height


def _normalize_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frames must be BGR images")
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


write_clip = write_frames_to_mp4
