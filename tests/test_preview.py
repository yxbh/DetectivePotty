"""Headless tests for the GUI-free preview helpers (no ``cv2.imshow``)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.preview import (
    fit_for_display,
    render_preview_frame,
    split_by_threshold,
)

BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _det(confidence: float, bbox: BBox = BBox(10, 10, 60, 60)) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_name="dog",
        frame_idx=0,
        mono_ts=0.0,
        wall_ts=BASE_TS,
    )


def test_split_by_threshold_partitions_at_boundary() -> None:
    dets = [_det(0.10), _det(0.25), _det(0.40)]
    above, below = split_by_threshold(dets, 0.25)
    assert [d.confidence for d in above] == [0.25, 0.40]
    assert [d.confidence for d in below] == [0.10]


def test_split_by_threshold_empty() -> None:
    above, below = split_by_threshold([], 0.5)
    assert above == []
    assert below == []


def test_render_preview_frame_draws_and_preserves_size() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    above = [_det(0.9, BBox(20, 20, 100, 100))]
    below = [_det(0.1, BBox(30, 30, 90, 90))]
    out = render_preview_frame(frame, above, below, ["threshold: 0.25"])
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    # Input is untouched; output has drawn (non-zero) pixels.
    assert not np.any(frame)
    assert np.any(out)


def test_render_preview_frame_handles_no_detections() -> None:
    frame = np.full((90, 120, 3), 30, dtype=np.uint8)
    out = render_preview_frame(frame, [], [])
    assert out.shape == frame.shape


def test_fit_for_display_scales_down_large_frames() -> None:
    big = np.zeros((1512, 2688, 3), dtype=np.uint8)
    scaled = fit_for_display(big, max_w=1280, max_h=720)
    h, w = scaled.shape[:2]
    assert w <= 1280 and h <= 720
    # Aspect ratio preserved.
    assert abs((w / h) - (2688 / 1512)) < 0.01


def test_fit_for_display_leaves_small_frames_unchanged() -> None:
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    out = fit_for_display(small, max_w=1280, max_h=720)
    assert out.shape == small.shape
