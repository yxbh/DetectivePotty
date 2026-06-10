"""Interactive YOLO detection preview for tuning ``detection_conf_threshold``.

This module powers the ``detectivepotty tune-detect`` command. It plays a video (a
local file or a live RTSP/Protect stream), runs the dog detector at a low confidence
*floor* so borderline boxes are visible, and draws every detection. A live trackbar
sets the confidence threshold: boxes at or above it render solid green, boxes below it
render dim red, so the cutoff that keeps the dog but drops noise is easy to find.

The pure helpers (:func:`split_by_threshold`, :func:`render_preview_frame`,
:func:`fit_for_display`) are GUI-free and unit-tested. The interactive loop
(:func:`run_interactive_preview`) is the only part that opens a window and is therefore
exercised by hand on a machine with a display, never in the headless test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from detectivepotty.detect.yolo import DogDetector
    from detectivepotty.events import Detection
    from detectivepotty.sources.base import VideoSource

# BGR colors. Above-threshold boxes are bright green; below-threshold boxes are a
# muted red so they read as "would be filtered out at the current threshold".
_ABOVE_COLOR = (40, 200, 40)
_BELOW_COLOR = (60, 60, 200)
_HUD_COLOR = (240, 240, 240)
_HUD_BG = (0, 0, 0)


def split_by_threshold(
    detections: Sequence[Detection], threshold: float
) -> tuple[list[Detection], list[Detection]]:
    """Partition detections into ``(above, below)`` the confidence threshold.

    A detection is "above" when ``confidence >= threshold`` — matching the detector's
    own ``confidence >= detection_conf_threshold`` keep rule — so the green boxes are
    exactly what would survive filtering at that threshold in production.
    """

    above: list[Detection] = []
    below: list[Detection] = []
    for detection in detections:
        if detection.confidence >= threshold:
            above.append(detection)
        else:
            below.append(detection)
    return above, below


def render_preview_frame(
    frame_bgr: np.ndarray,
    above: Sequence[Detection],
    below: Sequence[Detection],
    hud_lines: Sequence[str] = (),
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with detection boxes and a HUD drawn on it.

    Boxes in ``below`` are drawn first (thin, dim red) so the ``above`` boxes (thick,
    bright green) sit on top when they overlap. The input frame is not mutated.
    """

    if frame_bgr.ndim < 2:
        raise ValueError("frame_bgr must be an image array")
    canvas = frame_bgr.copy()
    height, width = canvas.shape[:2]

    for detection in below:
        _draw_box(canvas, detection, width, height, _BELOW_COLOR, thickness=2)
    for detection in above:
        _draw_box(canvas, detection, width, height, _ABOVE_COLOR, thickness=3)

    _draw_hud(canvas, hud_lines)
    return canvas


def fit_for_display(
    frame_bgr: np.ndarray, max_w: int = 1280, max_h: int = 720
) -> np.ndarray:
    """Downscale a frame to fit ``max_w`` x ``max_h`` for display, preserving aspect.

    Detection always runs on the full-resolution frame; this only shrinks the *display*
    so a 2688-wide clip fits on screen. Frames already within bounds are returned as-is.
    """

    height, width = frame_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return frame_bgr
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return frame_bgr
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def _draw_box(
    canvas: np.ndarray,
    detection: Detection,
    width: int,
    height: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = detection.bbox.clip_to(width, height).to_int_tuple()
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    label = f"{detection.class_name} {detection.confidence:.2f}"
    cv2.putText(
        canvas,
        label,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_hud(canvas: np.ndarray, hud_lines: Sequence[str]) -> None:
    if not hud_lines:
        return
    line_h = 26
    pad = 8
    box_h = pad * 2 + line_h * len(hud_lines)
    box_w = 8 + max(len(line) for line in hud_lines) * 11
    box_w = min(box_w, canvas.shape[1])
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), _HUD_BG, thickness=-1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, dst=canvas)
    for i, line in enumerate(hud_lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(
            canvas,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _HUD_COLOR,
            1,
            cv2.LINE_AA,
        )


class FrameProvider(Protocol):
    """Minimal frame-source contract the interactive loop drives."""

    is_live: bool
    fps: float
    position: int
    total_frames: int | None

    def read_next(self) -> np.ndarray | None: ...
    def step_back(self) -> np.ndarray | None: ...
    def restart(self) -> None: ...
    def close(self) -> None: ...


class FileFrameProvider:
    """Seekable file provider backed by ``cv2.VideoCapture`` (supports step/restart).

    Stays on ``cv2`` rather than the PyAV decode backend used by the harvest/export
    seams because it seeks/steps by frame index, which ``PyAvCapture`` does not
    support (it is sequential-only).
    """

    is_live = False

    def __init__(self, path: str | Path) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open video file: {path}")
        self._capture = capture
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames: int | None = total if total > 0 else None
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.position = -1

    def read_next(self) -> np.ndarray | None:
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            return None
        self.position += 1
        return bgr

    def _seek(self, index: int) -> None:
        index = max(0, index)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.position = index - 1

    def step_back(self) -> np.ndarray | None:
        self._seek(max(0, self.position - 1))
        return self.read_next()

    def restart(self) -> None:
        self._seek(0)

    def close(self) -> None:
        self._capture.release()


class LiveFrameProvider:
    """Play-only provider for live ``VideoSource`` streams (RTSP/Protect)."""

    is_live = True
    total_frames = None

    def __init__(self, source: VideoSource) -> None:
        self._source = source.open()
        self.fps = source.fps or 15.0
        self.position = -1

    def read_next(self) -> np.ndarray | None:
        frame = self._source.read()
        if frame is None:
            return None
        self.position += 1
        return frame.bgr

    def step_back(self) -> np.ndarray | None:
        return None

    def restart(self) -> None:
        return None

    def close(self) -> None:
        self._source.close()


def run_interactive_preview(
    provider: FrameProvider,
    detector: DogDetector,
    *,
    initial_conf: float,
    every_n: int = 1,
    window_name: str = "DetectivePotty tune-detect",
    max_w: int = 1280,
    max_h: int = 720,
) -> float:
    """Drive the interactive tuning window and return the threshold chosen at exit.

    Detection runs at the detector's (low) floor so all borderline boxes are returned;
    the trackbar applies the live threshold for the green/red split. Detections are
    cached per frame, so moving the slider re-colors instantly without re-inference.
    """

    every_n = max(1, every_n)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    slider = "conf x100"
    cv2.createTrackbar(slider, window_name, _clamp_pct(initial_conf), 100, lambda _v: None)

    current_bgr: np.ndarray | None = None
    current_dets: list[Detection] = []
    playing = True
    advances = 0
    threshold = initial_conf

    try:
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            threshold = cv2.getTrackbarPos(slider, window_name) / 100.0

            if playing or current_bgr is None:
                bgr = provider.read_next()
                if bgr is None:
                    if provider.is_live:
                        break
                    provider.restart()
                    bgr = provider.read_next()
                    if bgr is None:
                        break
                if advances % every_n == 0:
                    current_dets = _detect(detector, bgr, provider.position)
                current_bgr = bgr
                advances += 1

            above, below = split_by_threshold(current_dets, threshold)
            hud = _hud_lines(provider, threshold, above, below, detector, playing)
            annotated = render_preview_frame(current_bgr, above, below, hud)
            cv2.imshow(window_name, fit_for_display(annotated, max_w, max_h))

            delay = _frame_delay_ms(provider.fps) if playing else 50
            key = cv2.waitKey(delay) & 0xFF
            action = _handle_key(key, provider, playing)
            if action == "quit":
                break
            if action == "toggle":
                playing = not playing
            elif action in {"step_forward", "step_back", "restart"}:
                playing = False
                stepped = _apply_step(action, provider)
                if stepped is not None:
                    current_bgr = stepped
                    current_dets = _detect(detector, stepped, provider.position)
    finally:
        provider.close()
        cv2.destroyWindow(window_name)

    return threshold


def _detect(detector: DogDetector, bgr: np.ndarray, frame_idx: int) -> list[Detection]:
    return detector.detect(
        bgr,
        frame_idx=max(0, frame_idx),
        mono_ts=time.monotonic(),
        wall_ts=datetime.now(timezone.utc),
    )


def _apply_step(action: str, provider: FrameProvider) -> np.ndarray | None:
    if action == "step_forward":
        return provider.read_next()
    if action == "step_back":
        return provider.step_back()
    if action == "restart":
        provider.restart()
        return provider.read_next()
    return None


def _handle_key(key: int, provider: FrameProvider, playing: bool) -> str | None:
    if key in (ord("q"), 27):  # q or Esc
        return "quit"
    if key == ord(" "):
        return "toggle"
    if key in (ord("n"), 83):  # n or right-arrow
        return "step_forward"
    if not provider.is_live and key in (ord("p"), 81):  # p or left-arrow
        return "step_back"
    if not provider.is_live and key == ord("r"):
        return "restart"
    return None


def _hud_lines(
    provider: FrameProvider,
    threshold: float,
    above: Sequence[Detection],
    below: Sequence[Detection],
    detector: DogDetector,
    playing: bool,
) -> list[str]:
    pos = max(0, provider.position)
    where = "LIVE" if provider.is_live else (
        f"frame {pos}/{provider.total_frames}" if provider.total_frames else f"frame {pos}"
    )
    latency = ""
    info: Any = getattr(detector, "last_inference", None)
    if info is not None:
        latency = f"  {info.latency_ms:.0f}ms"
    keys = (
        "space pause  q quit"
        if provider.is_live
        else "space play/pause  n/p step  r restart  q quit"
    )
    return [
        f"threshold: {threshold:.2f}   ({'playing' if playing else 'paused'})",
        f"{where}   green>=thr: {len(above)}   below: {len(below)}",
        f"device {detector.device}{latency}",
        keys,
    ]


def _frame_delay_ms(fps: float) -> int:
    if fps <= 0:
        return 33
    return max(1, int(1000.0 / fps))


def _clamp_pct(conf: float) -> int:
    return max(0, min(100, round(conf * 100)))
