"""Exhaustive dense-YOLO ground truth for the harvest bake-off.

The reference every cheap strategy is scored against is the *exhaustive* pass: run
the detector on **every frame** of the acquired window and mark each second a
"dog-second" if any (or enough) of its frames contained a dog. This pass doubles as
the "blind scrub" strategy (recall 1.0, zero compute saved) and the ground truth.

Backend = the tuner's batched ``detect_batch`` path. On this machine a batched
CoreML export (``export_coreml(batch=32, dynamic=True)``) runs ~5.2 ms/frame — the
fastest measured backend — so a 48h@30fps pass is a single overnight run. The
detector is injected (a ``DetectorLike``) so the unit tests exercise this with a
fake and never touch a model, GPU, or video file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

from .timeline import SecondTimeline


class DetectorLike(Protocol):
    """Minimal surface this module needs from ``DogDetector`` (inject a fake in tests)."""

    def detect_batch(
        self, frames: Sequence[np.ndarray], metas: Sequence[object] | None = ...
    ) -> list[list[object]]: ...


@dataclass(frozen=True)
class FrameDetection:
    """Whether ``frame_idx`` (at ``fps``) held at least one dog, + its best score."""

    frame_idx: int
    has_dog: bool
    max_confidence: float


@dataclass
class GroundTruth:
    """Per-frame dog presence over a window + helpers to derive dog-seconds."""

    fps: float
    detections: list[FrameDetection]

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    @property
    def duration_s(self) -> int:
        if self.fps <= 0:
            return 0
        return int(self.frame_count / self.fps + 0.999999)

    @property
    def dog_frame_count(self) -> int:
        return sum(1 for d in self.detections if d.has_dog)

    def dog_seconds(self, min_dog_frames: int = 1) -> SecondTimeline:
        """Seconds containing at least ``min_dog_frames`` dog frames.

        ``min_dog_frames=1`` (default) is the most permissive / highest-recall
        definition of "a dog was here this second" — appropriate for ground truth,
        where missing a real dog-second is worse than tolerating a single-frame
        false positive from the detector.
        """

        if self.fps <= 0:
            return SecondTimeline.empty(0)
        counts: dict[int, int] = {}
        for d in self.detections:
            if d.has_dog:
                sec = int(d.frame_idx / self.fps)
                counts[sec] = counts.get(sec, 0) + 1
        selected = (sec for sec, n in counts.items() if n >= min_dog_frames)
        return SecondTimeline.from_iterable(selected, self.duration_s)


def _result_has_dog(detections: list[object]) -> tuple[bool, float]:
    """Reduce one frame's detection list to (has_dog, max_conf).

    ``DogDetector`` already filters to dogs and returns ``Detection`` objects with a
    ``.confidence``; we stay duck-typed so a fake in tests can return simple stand-ins.
    """

    best = 0.0
    found = False
    for det in detections:
        found = True
        conf = getattr(det, "confidence", None)
        if conf is not None and conf > best:
            best = float(conf)
    return found, best


def detect_frames(
    frames: Iterable[np.ndarray],
    detector: DetectorLike,
    *,
    fps: float,
    batch_size: int = 32,
    start_frame: int = 0,
) -> GroundTruth:
    """Run ``detector.detect_batch`` over ``frames`` in ``batch_size`` chunks.

    IO-agnostic: ``frames`` is any iterable of BGR arrays (a ``cv2`` capture loop in
    the CLI, a list in tests), so this is fully exercisable offline. Frame indices
    are assigned sequentially from ``start_frame``.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out: list[FrameDetection] = []
    idx = start_frame
    batch: list[np.ndarray] = []

    def _flush() -> None:
        nonlocal idx
        if not batch:
            return
        results = detector.detect_batch(batch, None)
        for frame_dets in results:
            has_dog, conf = _result_has_dog(list(frame_dets))
            out.append(FrameDetection(frame_idx=idx, has_dog=has_dog, max_confidence=conf))
            idx += 1
        batch.clear()

    for frame in frames:
        batch.append(frame)
        if len(batch) >= batch_size:
            _flush()
    _flush()
    return GroundTruth(fps=fps, detections=out)


def iter_video_frames(video_path: str, *, every_n: int = 1) -> Iterator[np.ndarray]:
    """Yield BGR frames from ``video_path`` (CLI use; not for tests).

    Decodes through the project's default capture backend
    (:func:`detectivepotty.sources.pyav_capture.open_capture` — PyAV multithreaded
    software decode with an OpenCV fallback), which is ~2.5–3x faster than OpenCV's
    single-threaded ``VideoCapture`` on this footage and yields the same frames in
    the same order. ``every_n>1`` subsamples (stride) — the exhaustive ground truth
    uses ``every_n=1`` (every frame); cheaper coarse passes can stride.
    """

    from detectivepotty.sources.pyav_capture import open_capture

    cap = open_capture(video_path)
    is_opened = getattr(cap, "isOpened", None)
    if callable(is_opened) and not is_opened():
        _release_capture(cap)
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i % every_n == 0:
                yield frame
            i += 1
    finally:
        _release_capture(cap)


def video_fps(video_path: str) -> float:
    import cv2

    from detectivepotty.sources.pyav_capture import open_capture

    cap = open_capture(video_path)
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        _release_capture(cap)


def _release_capture(cap: object) -> None:
    release = getattr(cap, "release", None)
    if callable(release):
        release()
