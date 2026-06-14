"""Decode, detect, and track sampled frames for historical harvesting."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from detectivepotty.detect.yolo import FrameMeta
from detectivepotty.events import Detection
from detectivepotty.harvest_spans import FrameSample
from detectivepotty.sources.prefetch import prefetch
from detectivepotty.tracking import Tracker

# Sampled frames are detected in one batched forward when the detector exposes
# ``detect_batch`` (real ``DogDetector``); the live pipeline proves CoreML/MPS
# true-batches here. Measured on the production CoreML export, batch 32 runs the
# scan's inference ~3.4x faster than single-frame ``detect`` (the dominant cost
# of a decode-overlapped scan). ``1`` reproduces the legacy single-frame path,
# as does any detector lacking ``detect_batch`` (e.g. test fakes).
DEFAULT_DETECT_BATCH_SIZE = 32


class DetectorLike(Protocol):
    """Minimal detector surface the harvester needs (matches ``DogDetector``).

    The scan uses :meth:`detect_batch` when present (batched forward, much faster
    on accelerated backends) and otherwise falls back to per-frame :meth:`detect`,
    so a fake exposing only ``detect`` still works unchanged.
    """

    def detect(
        self, frame_bgr_original: np.ndarray, frame_idx: int = ...
    ) -> list[Detection]: ...


def scan_for_dogs(
    input_path: Path,
    *,
    detector: DetectorLike,
    sample_every: int,
    iou_threshold: float,
    max_age_frames: int,
    center_dist_gate: float = 0.0,
    detect_batch_size: int = DEFAULT_DETECT_BATCH_SIZE,
    capture_factory: Callable[[str], Any],
) -> tuple[float, int, dict[str, list[FrameSample]]]:
    capture = capture_factory(str(input_path))
    if not _capture_opened(capture):
        _release(capture)
        raise RuntimeError(f"failed to open video file: {input_path}")

    fps = _capture_value(capture, cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker(
        iou_threshold=iou_threshold,
        max_age_frames=max_age_frames,
        center_dist_gate=center_dist_gate,
    )
    presence: dict[str, list[FrameSample]] = {}

    def _decode_frames():
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame

    # Feed one sampled frame's detections through the tracker and record presence.
    # Detections are per-image-independent, so replaying a batch's results in
    # frame order here yields the same tracks (and thus the same spans) as the
    # per-frame path -- batching only changes how the forward pass is issued.
    def _ingest(frame_idx: int, detections: list[Detection]) -> None:
        tracks = tracker.update(list(detections))
        time_s = frame_idx / fps
        for track in tracks:
            latest = latest_detection_at(track, frame_idx)
            if latest is None:
                continue
            presence.setdefault(track.track_id, []).append(
                FrameSample(
                    frame_idx=frame_idx,
                    time_s=time_s,
                    bbox=latest.bbox,
                    confidence=latest.confidence,
                    class_name=latest.class_name,
                )
            )

    detect_batch = getattr(detector, "detect_batch", None)
    batch_size = max(1, detect_batch_size)
    use_batch = detect_batch is not None and batch_size > 1

    pending_frames: list[np.ndarray] = []
    pending_idx: list[int] = []

    def _flush_batch() -> None:
        if not pending_frames:
            return
        metas = [FrameMeta(frame_idx=i) for i in pending_idx]
        results = detect_batch(pending_frames, metas)
        for idx, dets in zip(pending_idx, results):
            _ingest(idx, list(dets))
        pending_frames.clear()
        pending_idx.clear()

    decoded_idx = 0
    try:
        # Pipeline decode against detect+track: a background thread reads ahead
        # while the (GIL-releasing) detector runs, turning a decode-bound scan into
        # an inference-bound one. Tracking stays sequential on this thread, and
        # sampled frames are detected in batches of ``batch_size`` (when the
        # detector supports it) so the accelerator runs one forward per batch.
        for frame in prefetch(_decode_frames()):
            if decoded_idx % sample_every == 0:
                if use_batch:
                    pending_frames.append(frame)
                    pending_idx.append(decoded_idx)
                    if len(pending_frames) >= batch_size:
                        _flush_batch()
                else:
                    detections = detector.detect(frame, frame_idx=decoded_idx)
                    _ingest(decoded_idx, list(detections))
            decoded_idx += 1
        if use_batch:
            _flush_batch()
    finally:
        _release(capture)

    return fps, decoded_idx, presence


def latest_detection_at(track: Any, frame_idx: int) -> Detection | None:
    best: Detection | None = None
    for det in track.detections:
        if det.frame_idx == frame_idx:
            if best is None or det.confidence > best.confidence:
                best = det
    return best


_latest_detection_at = latest_detection_at


def _capture_opened(capture: Any) -> bool:
    is_opened = getattr(capture, "isOpened", None)
    return bool(is_opened()) if callable(is_opened) else True


def _capture_value(capture: Any, prop: int) -> float | None:
    value = capture.get(prop)
    return float(value) if value and value > 0 else None


def _release(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()
