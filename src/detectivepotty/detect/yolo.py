"""Ultralytics YOLO dog detector with detect-small/crop-big mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Iterable

import numpy as np

from detectivepotty.device import resolve_device
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox

DOG_CLASS_NAME = "dog"

logger = logging.getLogger(__name__)

# Process-wide cache of loaded ultralytics models, keyed by (requested model name,
# resolved device). Multiple ``DogDetector`` instances (e.g. one per camera) share a
# single loaded model: the pipeline serializes all inference behind a global lock, so
# concurrent use is never an issue, and this avoids N-fold model load time and memory.
_MODEL_CACHE: dict[tuple[str, str], tuple[object, str]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Drop all cached shared models (test hygiene / explicit reload)."""

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


@dataclass(frozen=True, slots=True)
class InferenceInfo:
    original_wh: tuple[int, int]
    inference_wh: tuple[int, int]
    original_to_inference_scale: tuple[float, float]
    inference_to_original_scale: tuple[float, float]
    latency_ms: float
    device: str


class DogDetector:
    """YOLO wrapper that returns dog boxes in original-resolution pixels."""

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        long_edge: int = 640,
        conf_threshold: float = 0.25,
        device: str = "auto",
        use_shared_model: bool = True,
    ) -> None:
        if long_edge <= 0:
            raise ValueError("long_edge must be positive")
        self.long_edge = long_edge
        self.conf_threshold = conf_threshold
        self.device = resolve_device(device)
        self.model_name = model_name
        self._use_shared_model = use_shared_model
        self._candidates: tuple[str, ...] = (model_name, "yolov8n.pt")
        self.model = self._acquire_model(self.device)
        self.last_inference: InferenceInfo | None = None

    def _acquire_model(self, device: str):
        """Return a model for ``device``, sharing one cached instance when enabled."""

        if not self._use_shared_model:
            return self._load_model(self._candidates)
        key = (self._candidates[0], device)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                model, resolved_name = cached
                self.model_name = resolved_name
                return model
            model = self._load_model(self._candidates)
            _MODEL_CACHE[key] = (model, self.model_name)
            return model

    def detect(
        self,
        frame_bgr_original: np.ndarray,
        frame_idx: int = 0,
        mono_ts: float | None = None,
        wall_ts: datetime | None = None,
    ) -> list[Detection]:
        if frame_bgr_original.ndim < 2:
            raise ValueError("frame_bgr_original must be an image array")

        mono_ts = time.monotonic() if mono_ts is None else mono_ts
        wall_ts = datetime.now(timezone.utc) if wall_ts is None else wall_ts
        original_h, original_w = frame_bgr_original.shape[:2]

        started = time.perf_counter()
        results = self._predict(frame_bgr_original)
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Ultralytics letterboxes the frame to ``imgsz`` (``self.long_edge``)
        # internally and returns boxes already in original-image coordinates, so
        # no manual rescale is needed. ``inference_wh`` is the effective network
        # resolution (long edge = imgsz, aspect preserved) recorded for telemetry.
        scale = self.long_edge / max(original_w, original_h)
        inference_w = max(1, round(original_w * scale))
        inference_h = max(1, round(original_h * scale))
        self.last_inference = InferenceInfo(
            original_wh=(original_w, original_h),
            inference_wh=(inference_w, inference_h),
            original_to_inference_scale=(
                inference_w / original_w,
                inference_h / original_h,
            ),
            inference_to_original_scale=(
                original_w / inference_w,
                original_h / inference_h,
            ),
            latency_ms=latency_ms,
            device=self.device,
        )

        detections: list[Detection] = []
        for xyxy, confidence, class_name in self._iter_boxes(results):
            if class_name.lower() != DOG_CLASS_NAME:
                continue
            if confidence < self.conf_threshold:
                continue
            original_bbox = BBox(*xyxy).clip_to(original_w, original_h)
            detections.append(
                Detection(
                    bbox=original_bbox,
                    confidence=confidence,
                    class_name=class_name,
                    frame_idx=frame_idx,
                    mono_ts=mono_ts,
                    wall_ts=wall_ts,
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _predict(self, inference_frame: np.ndarray):
        try:
            return self.model.predict(
                inference_frame,
                imgsz=self.long_edge,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception:
            if self.device == "cpu":
                raise
            logger.warning(
                "YOLO inference failed on device %s; falling back to CPU.",
                self.device,
            )
            self.device = "cpu"
            # Swap to a CPU-keyed model so we never move a shared accelerator model
            # between devices (which would thrash other cameras using it).
            self.model = self._acquire_model("cpu")
            return self.model.predict(
                inference_frame,
                imgsz=self.long_edge,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

    def _iter_boxes(self, results: Iterable[object]):
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            xyxys = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            for xyxy, confidence, class_id in zip(xyxys, confidences, class_ids):
                if isinstance(names, dict):
                    class_name = str(names.get(int(class_id), class_id))
                else:
                    class_name = str(names[int(class_id)])
                yield tuple(float(value) for value in xyxy), float(confidence), class_name

    def _load_model(self, candidates: Iterable[str]):
        from ultralytics import YOLO

        errors: list[Exception] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                model = YOLO(candidate)
                self.model_name = candidate
                return model
            except Exception as exc:  # pragma: no cover - depends on network/cache.
                errors.append(exc)
        raise RuntimeError(f"Unable to load YOLO model: {errors!r}")
