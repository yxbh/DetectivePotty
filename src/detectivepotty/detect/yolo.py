"""Ultralytics YOLO dog detector with detect-small/crop-big mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Iterable

import cv2
import numpy as np

from detectivepotty.events import Detection
from detectivepotty.geometry import BBox, map_bbox_to_original

DOG_CLASS_NAME = "dog"


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
        model_name: str = "yolo11n.pt",
        long_edge: int = 1280,
        conf_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        if long_edge <= 0:
            raise ValueError("long_edge must be positive")
        self.long_edge = long_edge
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)
        self.model_name = model_name
        self.model = self._load_model((model_name, "yolov8n.pt"))
        self.last_inference: InferenceInfo | None = None

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
        inference_frame = self._resize_for_inference(frame_bgr_original)
        inference_h, inference_w = inference_frame.shape[:2]

        started = time.perf_counter()
        results = self._predict(inference_frame)
        latency_ms = (time.perf_counter() - started) * 1000.0
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
            inference_bbox = BBox(*xyxy)
            original_bbox = map_bbox_to_original(
                inference_bbox,
                (inference_w, inference_h),
                (original_w, original_h),
            )
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

    def _resize_for_inference(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        long_edge = max(width, height)
        if long_edge <= self.long_edge:
            return frame_bgr
        scale = self.long_edge / long_edge
        resized_wh = (round(width * scale), round(height * scale))
        return cv2.resize(frame_bgr, resized_wh, interpolation=cv2.INTER_AREA)

    def _predict(self, inference_frame: np.ndarray):
        try:
            return self.model.predict(
                inference_frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception:
            if self.device != "mps":
                raise
            self.device = "cpu"
            return self.model.predict(
                inference_frame,
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

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested not in {"auto", "mps", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cpu")
        if requested == "cpu":
            return "cpu"
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        if requested == "mps":
            return "cpu"
        return "cpu"
