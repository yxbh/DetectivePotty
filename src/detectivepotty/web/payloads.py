"""Shared JSON payload shapers for web API contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox


_UNSET = object()


def bbox_coordinates_payload(bbox: BBox) -> dict[str, float]:
    return {
        "x1": float(bbox.x1),
        "y1": float(bbox.y1),
        "x2": float(bbox.x2),
        "y2": float(bbox.y2),
    }


def detection_payload(det: Detection, *, track_id: Any = _UNSET) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **bbox_coordinates_payload(det.bbox),
        "confidence": float(det.confidence),
        "class_name": det.class_name,
    }
    if track_id is not _UNSET:
        payload["track_id"] = track_id
    return payload


def scene_object_payload(class_name: str, confidence: float, bbox: BBox) -> dict[str, Any]:
    return {
        "class_name": class_name,
        "confidence": confidence,
        **bbox_coordinates_payload(bbox),
    }


def metadata_bbox_payload(box: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x1": float(box.get("x1", 0.0)),
        "y1": float(box.get("y1", 0.0)),
        "x2": float(box.get("x2", 0.0)),
        "y2": float(box.get("y2", 0.0)),
    }


def recorded_track_box_payload(
    det: Mapping[str, Any],
    *,
    clip_frame_idx: int,
) -> dict[str, Any]:
    box = det.get("bbox") or {}
    return {
        "clip_frame_idx": clip_frame_idx,
        "bbox": metadata_bbox_payload(box),
        "confidence": float(det.get("confidence", 0.0)),
        "class_name": str(det.get("class_name") or "dog"),
    }
