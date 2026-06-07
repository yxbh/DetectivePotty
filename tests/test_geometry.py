from __future__ import annotations

import numpy as np

from detectivepotty.geometry import BBox, crop_from_frame, map_bbox_to_original


def test_bbox_union_encloses_both_boxes() -> None:
    left = BBox(10, 20, 40, 60)
    right = BBox(30, 5, 50, 45)

    merged = left.union(right)

    assert merged == BBox(10, 5, 50, 60)
    # Union is commutative and idempotent.
    assert right.union(left) == merged
    assert left.union(left) == left


def test_map_bbox_to_original_scales_from_inference_to_original() -> None:
    bbox = BBox(100, 50, 300, 200)

    mapped = map_bbox_to_original(
        bbox,
        inference_wh=(1280, 720),
        original_wh=(2688, 1512),
    )

    assert mapped == BBox(210, 105, 630, 420)


def test_map_bbox_to_original_clips_to_original_frame() -> None:
    bbox = BBox(-10, 10, 1300, 730)

    mapped = map_bbox_to_original(
        bbox,
        inference_wh=(1280, 720),
        original_wh=(2688, 1512),
    )

    assert mapped == BBox(0, 21, 2688, 1512)


def test_bbox_expand_and_clip() -> None:
    bbox = BBox(10, 20, 30, 60)

    assert bbox.expand(0.5, frame_w=100, frame_h=100) == BBox(0, 0, 40, 80)
    assert BBox(-5, 10, 110, 120).clip_to(100, 100) == BBox(0, 10, 100, 100)


def test_crop_from_frame_uses_original_resolution() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = BBox(50, 25, 100, 50)

    crop = crop_from_frame(frame, bbox, margin_frac=0.5)

    assert crop.shape == (51, 100, 3)
