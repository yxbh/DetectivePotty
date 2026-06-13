"""Browser-friendly MP4 (H.264) clip writing.

OpenCV's default ``mp4v`` fourcc produces MPEG-4 Part 2 video, which modern
browsers refuse to play in a ``<video>`` element (and which Playwright's bundled
Chromium cannot decode at all). Every clip we write and then play back in the
review / labeling portals must therefore be **H.264** in an MP4 container.

This module centralises that encoding so both the historical-footage *harvest*
writer and the live-event *recorder* produce identical, web-playable clips.

Encoder selection (first that works wins):

1. **ffmpeg / libx264** piped raw BGR frames — the gold standard. Always emits
   H.264 + ``yuv420p`` with ``+faststart`` (moov atom at the front) so the
   browser can begin playback and seek immediately over HTTP range requests.
   ffmpeg is already a hard dependency of the RTSP stack, so this path is the
   norm on a real deployment.
2. **OpenCV ``avc1``** — H.264 via OpenCV's own FFmpeg backend, used when the
   ffmpeg binary is not on ``PATH``. Availability is build-dependent.
3. **OpenCV ``mp4v``** — last-resort fallback so harvesting/recording never
   hard-fails; logged as a warning because the output is *not* web-playable.

All writers share the minimal ``write(frame)`` / ``release()`` surface of the
``ClipWriter`` protocol used across the codebase.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["open_h264_writer", "ffmpeg_binary"]


@lru_cache(maxsize=1)
def ffmpeg_binary() -> str | None:
    """Return the path to an ``ffmpeg`` executable, or ``None`` if unavailable."""

    return shutil.which("ffmpeg")


def _even(value: int) -> int:
    """libx264 + yuv420p requires even dimensions; round *down* to keep origin."""

    return value - (value % 2)


def _as_contiguous_bgr_u8(frame: np.ndarray) -> np.ndarray:
    image = frame
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    return image


class _FfmpegH264Writer:
    """Stream raw BGR frames into ``ffmpeg`` and encode H.264 + faststart."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        width, height = size
        self._width = _even(int(width))
        self._height = _even(int(height))
        if self._width <= 0 or self._height <= 0:
            raise ValueError(f"invalid clip size: {size}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        rate = fps if fps and fps > 0 else 30.0
        ffmpeg = ffmpeg_binary()
        assert ffmpeg is not None  # only constructed when available
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            f"{rate:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._proc.stdin is None:
            raise OSError("clip writer is closed")
        image = _as_contiguous_bgr_u8(frame)
        if image.shape[1] != self._width or image.shape[0] != self._height:
            image = np.ascontiguousarray(image[: self._height, : self._width])
        try:
            self._proc.stdin.write(image.tobytes())
        except BrokenPipeError as exc:
            raise OSError(f"ffmpeg writer failed: {self._drain_stderr()}") from exc

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self._drain_stderr()
        code = self._proc.wait()
        if code != 0:
            raise OSError(f"ffmpeg encode failed (exit {code}): {stderr}")

    def _drain_stderr(self) -> str:
        if self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return ""


class _Cv2Writer:
    """OpenCV-backed writer for the ``avc1`` / ``mp4v`` fallback paths."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], fourcc: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = fps if fps and fps > 0 else 30.0
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), rate, size
        )
        if not self._writer.isOpened():
            raise OSError(f"could not open clip writer ({fourcc}): {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def open_h264_writer(path: Path, fps: float, size: tuple[int, int]):
    """Open a web-playable clip writer for ``path`` at ``size`` (width, height).

    Returns an object exposing ``write(frame)`` / ``release()``. Prefers
    ffmpeg/libx264, then OpenCV ``avc1``, then OpenCV ``mp4v`` as a last resort.
    """

    path = Path(path)
    if ffmpeg_binary() is not None:
        try:
            return _FfmpegH264Writer(path, fps, size)
        except (OSError, ValueError) as exc:
            logger.warning("ffmpeg H.264 writer unavailable (%s); falling back", exc)
    try:
        return _Cv2Writer(path, fps, size, "avc1")
    except OSError as exc:
        logger.warning(
            "OpenCV avc1 (H.264) writer unavailable (%s); falling back to mp4v — "
            "clips will NOT be browser-playable",
            exc,
        )
    return _Cv2Writer(path, fps, size, "mp4v")
