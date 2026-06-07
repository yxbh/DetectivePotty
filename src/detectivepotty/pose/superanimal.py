"""SuperAnimal-Quadruped top-down pose backend (DeepLabCut 3.x).

We already have a dog bounding box from the detector, so this runs TOP-DOWN: the
detector box is expanded by ``crop_margin_frac`` (to rescue boxes the detector drew
too small — a real failure mode on small/night dogs), the frame is cropped to that
expanded region, and the pose model runs on the crop. Keypoints come back in crop
pixels and are mapped to original-frame pixels.

``deeplabcut`` is a very heavy dependency, so it is imported lazily inside
``_build_dlc_infer_fn`` only — importing this module (and running the unit tests)
never requires it. The actual model call is isolated behind an injectable
``infer_fn`` seam so the crop/coordinate/provenance/gating logic is testable without
the model, and so an ONNX path can be swapped in later.

Validated against real deeplabcut 3.0 in ``files/pose_spike/inmemory_runner.py``:
a persistent in-memory ``PoseInferenceRunner`` runs ~52 ms/crop warm on MPS (vs
~1284 ms for the file-based API), expects RGB input, returns
``preds[0]["bodyparts"]`` shaped ``(n_individuals, 39, 3)`` and its
``cfg["metadata"]["bodyparts"]`` order matches :data:`QUADRUPED_KEYPOINTS`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from detectivepotty.config import PoseConfig
from detectivepotty.device import resolve_device
from detectivepotty.geometry import BBox
from detectivepotty.pose.base import PoseEstimator
from detectivepotty.pose.keypoints import (
    QUADRUPED_KEYPOINTS,
    QUADRUPED_SCHEMA,
    Keypoint,
    PoseKeypoints,
)
from detectivepotty.pose.telemetry import (
    OUTCOME_BAD_SHAPE,
    OUTCOME_BUILD_FAILED,
    OUTCOME_INFER_ERROR,
    OUTCOME_INFER_NONE,
    OUTCOME_NO_FINITE_KEYPOINTS,
    OUTCOME_SKIP_TINY_CROP,
    OUTCOME_SUCCESS,
    PoseTelemetry,
    PoseTelemetrySnapshot,
)

logger = logging.getLogger(__name__)

# (rgb_crop, bbox_xyxy_in_crop_coords) -> (n_keypoints, 3) array of (x, y, conf)
# in crop pixel coordinates. The bbox tells a top-down model which region to pose;
# we currently pass the whole (already expanded) crop.
InferFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

SUPERANIMAL_NAME = "superanimal_quadruped"
DEFAULT_DETECTOR_NAME = "fasterrcnn_resnet50_fpn_v2"

# Below this crop edge length (px) a pose is meaningless; skip the model entirely.
_MIN_CROP_EDGE_PX = 8


def resolve_pose_device(requested: str) -> str:
    """Resolve a pose device, mirroring the detector's policy for consistency.

    Delegates to :func:`detectivepotty.device.resolve_device`: ``auto`` prefers
    CUDA, then MPS, then CPU; an explicit accelerator that is unavailable falls
    back to CPU.
    """

    return resolve_device(requested)


def resolve_bodypart_order(cfg_bodyparts: Sequence[str]) -> tuple[str, ...]:
    """Return the keypoint name order matching the model's output rows.

    Accepts the model's metadata order when it is the known schema (optionally
    reordered), but refuses an incompatible schema: the semantic alias layer keys
    off specific raw DeepLabCut names, so a renamed/missing/extra set must fail
    loudly rather than silently mislabel keypoints.
    """

    parts = tuple(cfg_bodyparts)
    if parts == QUADRUPED_KEYPOINTS:
        return QUADRUPED_KEYPOINTS
    if set(parts) == set(QUADRUPED_KEYPOINTS):
        return parts
    raise RuntimeError(
        "SuperAnimal bodypart schema is incompatible with the expected "
        f"quadruped schema (got {len(parts)} parts: {parts!r})"
    )


def normalize_pose_array(raw: object, n_bodyparts: int) -> np.ndarray | None:
    """Coerce a raw model output to a single ``(n_bodyparts, 3)`` float array.

    Returns ``None`` for empty/no-individual outputs or any unexpected shape so a
    malformed prediction degrades to "no pose" instead of a partial mis-mapping.
    """

    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return None
        arr = arr[0]
    if arr.shape != (n_bodyparts, 3):
        return None
    return arr


class SuperAnimalPoseEstimator(PoseEstimator):
    """Top-down DeepLabCut SuperAnimal-Quadruped pose estimator."""

    def __init__(
        self,
        config: PoseConfig,
        *,
        infer_fn: InferFn | None = None,
        superanimal_name: str = SUPERANIMAL_NAME,
        detector_name: str = DEFAULT_DETECTOR_NAME,
    ) -> None:
        self._crop_margin = config.crop_margin_frac
        self._model_name = config.model_name
        self._min_keypoint_conf = config.min_keypoint_conf
        self._device = resolve_pose_device(config.device)
        self._superanimal_name = superanimal_name
        self._detector_name = detector_name

        self._bodyparts: tuple[str, ...] = QUADRUPED_KEYPOINTS
        self._infer_fn = infer_fn
        self._lock = threading.Lock()
        self._build_failed = False
        self._infer_error_count = 0
        self._telemetry = PoseTelemetry(conf_threshold=self._min_keypoint_conf)

    def estimate(
        self,
        frame_bgr_original: np.ndarray,
        bbox: BBox,
        frame_idx: int = 0,
        mono_ts: float | None = None,
        wall_ts: datetime | None = None,
        source_id: str | None = None,
    ) -> PoseKeypoints | None:
        if frame_bgr_original.ndim < 2:
            raise ValueError("frame_bgr_original must be an image array")

        mono_ts = time.monotonic() if mono_ts is None else mono_ts
        wall_ts = datetime.now(timezone.utc) if wall_ts is None else wall_ts
        frame_h, frame_w = frame_bgr_original.shape[:2]

        crop_box = bbox.expand(self._crop_margin, frame_w, frame_h)
        x1, y1, x2, y2 = crop_box.to_int_tuple()
        if x2 - x1 < _MIN_CROP_EDGE_PX or y2 - y1 < _MIN_CROP_EDGE_PX:
            self._telemetry.record(OUTCOME_SKIP_TINY_CROP)
            return None
        # Provenance records the integer box actually sliced, not the float box.
        actual_crop = BBox(float(x1), float(y1), float(x2), float(y2))

        crop_bgr = frame_bgr_original[y1:y2, x1:x2]
        crop_h, crop_w = crop_bgr.shape[:2]
        rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        target_bbox = np.array([[0.0, 0.0, float(crop_w), float(crop_h)]], dtype=float)

        raw, latency_ms, fail_outcome = self._infer(rgb_crop, target_bbox)
        if raw is None:
            # Build/inference failure; ``latency_ms`` is None unless the model ran.
            self._telemetry.record(fail_outcome, latency_ms=latency_ms)
            return None

        arr = normalize_pose_array(raw, len(self._bodyparts))
        if arr is None:
            self._telemetry.record(OUTCOME_BAD_SHAPE, latency_ms=latency_ms)
            return None

        points: dict[str, Keypoint] = {}
        for name, (kx, ky, kc) in zip(self._bodyparts, arr):
            if not (np.isfinite(kx) and np.isfinite(ky) and np.isfinite(kc)):
                continue
            points[name] = Keypoint(float(kx) + x1, float(ky) + y1, float(kc))
        if not points:
            self._telemetry.record(OUTCOME_NO_FINITE_KEYPOINTS, latency_ms=latency_ms)
            return None

        confidences = [kp.confidence for kp in points.values()]
        mean_conf = sum(confidences) / len(confidences)
        frac_ge = sum(c >= self._min_keypoint_conf for c in confidences) / len(confidences)
        self._telemetry.record(
            OUTCOME_SUCCESS,
            latency_ms=latency_ms,
            mean_conf=mean_conf,
            frac_ge=frac_ge,
            n_keypoints=len(points),
        )

        return PoseKeypoints(
            points=points,
            frame_idx=frame_idx,
            mono_ts=mono_ts,
            wall_ts=wall_ts,
            source_id=source_id,
            backend="superanimal",
            model_name=self._model_name,
            device=self._device,
            image_size=(frame_w, frame_h),
            crop_bbox=actual_crop,
            crop_margin_frac=self._crop_margin,
            keypoint_schema=QUADRUPED_SCHEMA,
            latency_ms=latency_ms,
        )

    def telemetry_snapshot(self) -> PoseTelemetrySnapshot:
        """Return a point-in-time snapshot of pose perf/health telemetry."""

        return self._telemetry.snapshot()

    def log_telemetry(self, level: int = logging.INFO) -> None:
        """Log the current telemetry snapshot (handy for spikes/periodic dumps)."""

        logger.log(level, "pose telemetry: %s", self._telemetry.snapshot().to_dict())

    def _infer(
        self, rgb_crop: np.ndarray, target_bbox: np.ndarray
    ) -> tuple[np.ndarray | None, float | None, str | None]:
        """Run the model under a lock, lazily building the runner on first use.

        Returns ``(raw_or_none, run_ms_or_none, fail_outcome_or_none)``. ``run_ms``
        times ONLY the model call (not lock wait or the one-time build) and is
        populated whenever the model actually ran (even if it returned nothing).
        Build/setup failures disable the backend permanently (one error log) so a
        misconfiguration does not retry the heavy load every frame; per-frame
        inference errors are caught and rate-limited so one bad frame never kills a
        camera thread.
        """

        with self._lock:
            if self._infer_fn is None:
                if self._build_failed:
                    return None, None, OUTCOME_BUILD_FAILED
                build_started = time.perf_counter()
                try:
                    self._infer_fn = self._build_dlc_infer_fn()
                except Exception:
                    build_ms = (time.perf_counter() - build_started) * 1000.0
                    self._build_failed = True
                    self._telemetry.record_cold_start(build_ms, ok=False)
                    logger.error(
                        "SuperAnimal pose backend failed to initialize; "
                        "pose is disabled for this run",
                        exc_info=True,
                    )
                    return None, None, OUTCOME_BUILD_FAILED
                build_ms = (time.perf_counter() - build_started) * 1000.0
                self._telemetry.record_cold_start(build_ms, ok=True)
            run_started = time.perf_counter()
            try:
                raw = self._infer_fn(rgb_crop, target_bbox)
            except Exception:
                self._infer_error_count += 1
                if self._infer_error_count == 1 or self._infer_error_count % 100 == 0:
                    logger.warning(
                        "SuperAnimal pose inference failed (count=%d)",
                        self._infer_error_count,
                        exc_info=True,
                    )
                return None, None, OUTCOME_INFER_ERROR
            run_ms = (time.perf_counter() - run_started) * 1000.0
            if raw is None:
                return None, run_ms, OUTCOME_INFER_NONE
            return raw, run_ms, None

    def _build_dlc_infer_fn(self) -> InferFn:
        try:
            from deeplabcut.pose_estimation_pytorch import modelzoo
            from deeplabcut.pose_estimation_pytorch.apis import (
                get_pose_inference_runner,
            )
        except ImportError as exc:  # pragma: no cover - exercised only with the dep.
            raise ImportError(
                "The SuperAnimal pose backend requires deeplabcut. Install it in a "
                "separate environment (it is intentionally not a core dependency): "
                "pip install 'deeplabcut[pytorch]>=3.0'"
            ) from exc

        snapshot_path = modelzoo.get_super_animal_snapshot_path(
            dataset=self._superanimal_name, model_name=self._model_name
        )
        model_cfg = modelzoo.load_super_animal_config(
            super_animal=self._superanimal_name,
            model_name=self._model_name,
            detector_name=self._detector_name,
        )
        self._bodyparts = resolve_bodypart_order(model_cfg["metadata"]["bodyparts"])
        runner = get_pose_inference_runner(
            model_config=model_cfg,
            snapshot_path=snapshot_path,
            batch_size=1,
            max_individuals=1,
            device=self._device,
        )

        def infer(rgb_crop: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
            preds = runner.inference([(rgb_crop, {"bboxes": target_bbox})])
            return np.asarray(preds[0]["bodyparts"], dtype=float)

        return infer
