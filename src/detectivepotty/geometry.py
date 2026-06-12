"""Geometry helpers for original-resolution dog detection boxes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class BBox:
    """Pixel-space bounding box in a stated reference resolution."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def scaled(self, sx: float, sy: float) -> "BBox":
        return BBox(
            x1=self.x1 * sx,
            y1=self.y1 * sy,
            x2=self.x2 * sx,
            y2=self.y2 * sy,
        )

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        """Return crop-safe integer coordinates using floor/ceil."""

        return (
            math.floor(self.x1),
            math.floor(self.y1),
            math.ceil(self.x2),
            math.ceil(self.y2),
        )

    def clip_to(self, w: int | float, h: int | float) -> "BBox":
        if w < 0 or h < 0:
            raise ValueError("Frame dimensions must be non-negative")

        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return BBox(
            x1=min(max(x1, 0.0), float(w)),
            y1=min(max(y1, 0.0), float(h)),
            x2=min(max(x2, 0.0), float(w)),
            y2=min(max(y2, 0.0), float(h)),
        )

    def union(self, other: "BBox") -> "BBox":
        """Return the smallest box enclosing both ``self`` and ``other``."""

        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other: "BBox") -> float:
        """Return intersection-over-union with ``other`` (0.0 when disjoint)."""

        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0.0:
            return 0.0
        denom = self.area + other.area - intersection
        if denom <= 0.0:
            return 0.0
        return intersection / denom

    def expand(
        self,
        margin_frac: float,
        frame_w: int | float,
        frame_h: int | float,
    ) -> "BBox":
        if margin_frac < 0:
            raise ValueError("margin_frac must be non-negative")

        dx = self.width * margin_frac
        dy = self.height * margin_frac
        return BBox(
            x1=self.x1 - dx,
            y1=self.y1 - dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        ).clip_to(frame_w, frame_h)


def map_bbox_to_original(
    bbox_in_inference_space: BBox,
    inference_wh: Tuple[int, int],
    original_wh: Tuple[int, int],
) -> BBox:
    """Map a downscaled-inference bbox back to original-resolution pixels."""

    inference_w, inference_h = inference_wh
    original_w, original_h = original_wh
    if inference_w <= 0 or inference_h <= 0:
        raise ValueError("inference_wh must contain positive dimensions")
    if original_w <= 0 or original_h <= 0:
        raise ValueError("original_wh must contain positive dimensions")

    sx = original_w / inference_w
    sy = original_h / inference_h
    return bbox_in_inference_space.scaled(sx, sy).clip_to(original_w, original_h)


def crop_from_frame(
    frame_bgr_original_res: np.ndarray,
    bbox_original_res: BBox,
    margin_frac: float = 0.25,
) -> np.ndarray:
    """Crop a dog-centered region from the original-resolution BGR frame."""

    if frame_bgr_original_res.ndim < 2:
        raise ValueError("frame_bgr_original_res must be an image array")

    frame_h, frame_w = frame_bgr_original_res.shape[:2]
    crop_box = bbox_original_res.expand(margin_frac, frame_w, frame_h)
    x1, y1, x2, y2 = crop_box.to_int_tuple()
    x1 = min(max(x1, 0), frame_w)
    y1 = min(max(y1, 0), frame_h)
    x2 = min(max(x2, 0), frame_w)
    y2 = min(max(y2, 0), frame_h)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr_original_res[0:0, 0:0].copy()
    return frame_bgr_original_res[y1:y2, x1:x2].copy()
