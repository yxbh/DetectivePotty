"""Ultralytics YOLO dog detector with detect-small/crop-big mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Iterable, Sequence

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


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Per-frame identity/timestamps for batched detection.

    Mirrors the optional args of :meth:`DogDetector.detect` so a batch entry
    carries the same context a single ``detect`` call would.
    """

    frame_idx: int = 0
    mono_ts: float | None = None
    wall_ts: datetime | None = None


@dataclass
class BatchStats:
    """Cumulative batch-inference telemetry for interpreting throughput.

    Lets the operator see whether batches are actually forming on the
    accelerator (vs silently falling back to per-frame) and the effective
    batch size achieved.
    """

    calls: int = 0
    frames: int = 0
    batched_calls: int = 0
    per_frame_fallback_calls: int = 0
    total_latency_ms: float = 0.0
    max_effective_batch: int = 0

    @property
    def mean_batch_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def mean_effective_batch(self) -> float:
        return self.frames / self.calls if self.calls else 0.0


class DogDetector:
    """YOLO wrapper that returns dog boxes in original-resolution pixels."""

    def __init__(
        self,
        model_name: str = "models/yolo11m.pt",
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
        # Set once a batched predict raises on this accelerator (e.g. a
        # fixed-batch=1 CoreML package): we then submit frames one-at-a-time on
        # the SAME accelerator rather than forcing CPU.
        self._batch_unsupported = False
        self.batch_stats = BatchStats()

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

        self.last_inference = self._inference_info(original_w, original_h, latency_ms)
        result = results[0] if len(results) else None
        return self._result_to_detections(
            result, original_w, original_h, frame_idx, mono_ts, wall_ts
        )

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        metas: Sequence[FrameMeta] | None = None,
    ) -> list[list[Detection]]:
        """Run one batched forward over ``frames``; return per-frame detections.

        Detections are per-image-independent (same weights, per-image NMS), so the
        result for ``frames[i]`` is identical to calling :meth:`detect` on it. The
        fallback ladder is: batched on the accelerator → per-frame on the SAME
        accelerator (handles a fixed-batch=1 CoreML package) → CPU only if a single
        frame also fails (delegated to :meth:`_predict`). We never force CPU just
        because batching is unsupported.
        """

        frames = list(frames)
        if not frames:
            return []
        if metas is None:
            metas = [FrameMeta(frame_idx=i) for i in range(len(frames))]
        if len(metas) != len(frames):
            raise ValueError("metas length must match frames length")
        for frame in frames:
            if frame.ndim < 2:
                raise ValueError("each frame must be an image array")

        resolved: list[tuple[int, float, datetime]] = []
        for meta in metas:
            mono = time.monotonic() if meta.mono_ts is None else meta.mono_ts
            wall = datetime.now(timezone.utc) if meta.wall_ts is None else meta.wall_ts
            resolved.append((meta.frame_idx, mono, wall))

        started = time.perf_counter()
        results, batched = self._predict_batch(frames)
        latency_ms = (time.perf_counter() - started) * 1000.0

        self._record_batch_stats(len(frames), latency_ms, batched)

        # Telemetry reflects the last frame's geometry (uniform within a camera);
        # latency is the whole-batch time.
        last_h, last_w = frames[-1].shape[:2]
        self.last_inference = self._inference_info(last_w, last_h, latency_ms)

        outputs: list[list[Detection]] = []
        for frame, result, (frame_idx, mono, wall) in zip(frames, results, resolved):
            original_h, original_w = frame.shape[:2]
            outputs.append(
                self._result_to_detections(
                    result, original_w, original_h, frame_idx, mono, wall
                )
            )
        return outputs

    def _inference_info(
        self, original_w: int, original_h: int, latency_ms: float
    ) -> InferenceInfo:
        # Ultralytics letterboxes the frame to ``imgsz`` (``self.long_edge``)
        # internally and returns boxes already in original-image coordinates, so
        # no manual rescale is needed. ``inference_wh`` is the effective network
        # resolution (long edge = imgsz, aspect preserved) recorded for telemetry.
        scale = self.long_edge / max(original_w, original_h)
        inference_w = max(1, round(original_w * scale))
        inference_h = max(1, round(original_h * scale))
        return InferenceInfo(
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

    def _result_to_detections(
        self,
        result: object | None,
        original_w: int,
        original_h: int,
        frame_idx: int,
        mono_ts: float,
        wall_ts: datetime,
    ) -> list[Detection]:
        detections: list[Detection] = []
        if result is None:
            return detections
        for xyxy, confidence, class_name in self._iter_boxes([result]):
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

    def _record_batch_stats(
        self, n_frames: int, latency_ms: float, batched: bool
    ) -> None:
        stats = self.batch_stats
        stats.calls += 1
        stats.frames += n_frames
        stats.total_latency_ms += latency_ms
        if batched:
            stats.batched_calls += 1
            stats.max_effective_batch = max(stats.max_effective_batch, n_frames)
        else:
            stats.per_frame_fallback_calls += 1

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

    def _predict_batch(self, frames: Sequence[np.ndarray]) -> tuple[list, bool]:
        """Return (results, batched). ``batched`` is False when we submitted the
        frames one-at-a-time (fixed-batch accelerator) so telemetry stays honest.

        A single-element batch is still issued as one ``predict`` call (cheap, and
        lets a one-frame tail reuse the batched path)."""

        if not self._batch_unsupported:
            try:
                results = self.model.predict(
                    list(frames),
                    imgsz=self.long_edge,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False,
                )
                return list(results), True
            except Exception:
                logger.warning(
                    "Batched YOLO inference failed on device %s; falling back to "
                    "per-frame on the same accelerator.",
                    self.device,
                )
                self._batch_unsupported = True
        return self._predict_per_frame(frames), False

    def _predict_per_frame(self, frames: Sequence[np.ndarray]) -> list:
        results: list = []
        for frame in frames:
            # ``_predict`` itself handles the accelerator -> CPU fallback if a
            # single frame fails, so CPU is only reached when per-frame also fails.
            results.extend(self._predict(frame))
        return results

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
