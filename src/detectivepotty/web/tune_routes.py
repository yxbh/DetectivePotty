"""Tune API routes for the local review web app."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from detectivepotty.config import Config
from detectivepotty.web.schemas import (
    ExportCoremlRequest,
    TunePoseRangeRequest,
    TunePoseRequest,
)
from detectivepotty.web.payloads import scene_object_payload
from detectivepotty.web.media import no_store_video_response
from detectivepotty.web.tune_tracking import (
    TUNE_DETECTION_FLOOR,
    TUNE_TRACKERS,
    ULTRALYTICS_TRACKERS as _ULTRALYTICS_TRACKERS,
    TuneUltralyticsTrackerParams,
    ultralytics_boxes as _ultralytics_boxes,
    ultralytics_dog_class_indices as _ultralytics_dog_class_indices,
    ultralytics_tracker_yaml as _ultralytics_tracker_yaml,
)


logger = logging.getLogger(__name__)

# Upper bound on total pose crops the batched pose pass (`POST /api/tune/pose-range`)
# will run for one request, regardless of how many frames/boxes the client sends.
TUNE_POSE_MAX_CROPS = 64

# Upper bound on the number of source frames one "Track range" request will decode.
TUNE_TRACK_MAX_FRAMES = 6000


@dataclass(frozen=True, slots=True)
class _TrackRangeRequest:
    file_path: Path
    model_name: str
    bounded_count: int
    ultra_params: TuneUltralyticsTrackerParams


def _coreml_batch_map(models: list[str]) -> dict[str, int]:
    """Map each ``.mlpackage`` in ``models`` to its baked max batch size.

    Lets the model picker label CoreML options with the batch they run (e.g.
    ``yolo11m (CoreML ×16)``) so it is obvious which exports are batched. Reads
    each package's spec once (memoised by mtime); ``.pt`` weights are omitted.
    """

    from detectivepotty.detect import coreml_export

    return {
        m: coreml_export.coreml_max_batch(m)
        for m in models
        if m.endswith(".mlpackage")
    }


def register_tune_routes(
    app: FastAPI,
    config: Config,
    *,
    ultralytics_tracking_available: Callable[[], bool],
) -> None:
    """Register Tune detection, tracking, export, and pose routes on ``app``."""

    def _get_tune_detector(model_name: str | None = None):
        """Lazily build (and cache) a tuner YOLO detector for ``model_name``.

        ``model_name`` defaults to the configured model. It must be in the
        discovered allow-list (``app.state.tune_models``) or ``ValueError`` is
        raised (the endpoint maps that to 400), so an arbitrary model string
        can't trigger a download or filesystem read. Returns ``(detector, name)``.
        """

        name = model_name or app.state.tune_default_model
        if name not in app.state.tune_models:
            raise ValueError(f"unknown model: {name}")
        with app.state.tune_detector_lock:
            cached = app.state.tune_detectors.get(name)
            if cached is None:
                from detectivepotty.detect.yolo import DogDetector

                cached = DogDetector(
                    model_name=name,
                    long_edge=config.global_settings.inference_long_edge_px,
                    conf_threshold=TUNE_DETECTION_FLOOR,
                    device=config.global_settings.device,
                    alias_classes=config.global_settings.dog_alias_classes,
                    alias_nms_iou=config.global_settings.dog_alias_nms_iou,
                )
                app.state.tune_detectors[name] = cached
                while len(app.state.tune_detectors) > app.state.tune_detector_cache_size:
                    app.state.tune_detectors.popitem(last=False)
            else:
                app.state.tune_detectors.move_to_end(name)
            return cached, name

    def _get_tune_pose():
        """Resolve (estimator | None, available) once and cache it on app.state."""

        if app.state.tune_pose_resolved is not None:
            return app.state.tune_pose_resolved
        with app.state.tune_pose_lock:
            if app.state.tune_pose_resolved is None:
                from detectivepotty.web.tune import build_tune_pose_estimator

                app.state.tune_pose_resolved = build_tune_pose_estimator(config)
            return app.state.tune_pose_resolved

    @app.get("/api/tune/files")
    def tune_files(path: str = "") -> dict:
        from detectivepotty.web.tune import list_tune_dir

        try:
            return list_tune_dir(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc

    def _resolve_tune_model(model: str | None) -> str:
        """Default + allow-list a requested model, or raise 400."""

        name = model or app.state.tune_default_model
        if name not in app.state.tune_models:
            raise HTTPException(status_code=400, detail="unknown model")
        return name

    def _track_range_request(
        path: str,
        count: int,
        model: str | None,
        tracker: str,
        ultra_conf: float,
        track_high_thresh: float | None,
        track_low_thresh: float | None,
        new_track_thresh: float | None,
        track_buffer: int | None,
        match_thresh: float | None,
        proximity_thresh: float | None,
        appearance_thresh: float | None,
    ) -> _TrackRangeRequest:
        if tracker not in TUNE_TRACKERS:
            raise HTTPException(status_code=400, detail=f"unknown tracker: {tracker}")

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc

        return _TrackRangeRequest(
            file_path=file_path,
            model_name=_resolve_tune_model(model),
            bounded_count=min(count, TUNE_TRACK_MAX_FRAMES),
            ultra_params=TuneUltralyticsTrackerParams(
                conf=ultra_conf,
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                proximity_thresh=proximity_thresh,
                appearance_thresh=appearance_thresh,
            ),
        )

    def _ensure_ultralytics_tracking(model_name: str) -> None:
        if not model_name.endswith(".pt"):
            raise HTTPException(
                status_code=400,
                detail="Ultralytics tracking requires a .pt model",
            )
        if not ultralytics_tracking_available():
            raise HTTPException(
                status_code=400,
                detail="Ultralytics tracking unavailable (install `lap`)",
            )

    def _detect_payload(
        file_path: Path,
        index: int,
        model_name: str,
        want_pose: bool,
        want_image: bool,
    ) -> dict:
        """Decode a frame, run detection (+optional pose), and shape the payload.

        Shared by ``/api/tune/frame`` (``want_image=True`` — server-rendered JPEG,
        kept for back-compat) and ``/api/tune/detect`` (``want_image=False`` —
        boxes only, the cheap payload the client buffers for the async overlay).
        Runs inside ``run_in_threadpool``; all model inference is serialized by
        ``tune_infer_lock`` because torch/MPS isn't reliably concurrent.
        """

        from detectivepotty.web import tune as tune_mod

        frame, idx, total, fps, width, height = tune_mod.read_frame(file_path, index)
        detector, model_used = _get_tune_detector(model_name)
        # Only resolve/build the pose estimator when pose is actually requested.
        # Building the real SuperAnimal backend is slow, so doing it on the
        # boxes-only buffer path would stall the first detection (YOLO priority).
        if want_pose:
            estimator, pose_available = _get_tune_pose()
        else:
            resolved = app.state.tune_pose_resolved
            estimator = None
            pose_available = resolved[1] if resolved is not None else True
        with app.state.tune_infer_lock:
            detections = detector.detect(
                frame,
                frame_idx=idx,
                mono_ts=time.monotonic(),
                wall_ts=datetime.now(timezone.utc),
            )
            pose_list: list = []
            if want_pose and estimator is not None:
                try:
                    pose_list = tune_mod.pose_payload(
                        estimator, frame, detections, frame_idx=idx
                    )
                except Exception:
                    logger.warning("pose inference failed for tune detect", exc_info=True)
                    pose_list = []
        payload = {
            "path": str(file_path),
            "index": idx,
            "total_frames": total or None,
            "fps": fps,
            "width": width,
            "height": height,
            "model": model_used,
            "detection_floor": TUNE_DETECTION_FLOOR,
            "detections": tune_mod.detections_payload(detections),
            "pose": pose_list,
            "pose_available": pose_available,
        }
        if want_image:
            payload["image"] = tune_mod.encode_jpeg_dataurl(frame)
        return payload

    def _scene_payload(
        file_path: Path,
        index: int,
        model_name: str,
        top_n: int,
    ) -> dict:
        """Top-N all-class detections for one frame (diagnostic, boxes but no image).

        Mirrors ``_detect_payload`` but skips the dog-class filter, so the reviewer
        can see whether a frame with no dog box is empty, a sub-threshold dog, or an
        animal classed as something else (cat/person/...). Each object carries its
        original-frame box (``x1,y1,x2,y2``) so the client can overlay it. Read-only —
        runs the same detector at the confidence floor under ``tune_infer_lock`` and
        never touches the harvest/detection pipeline.
        """

        from detectivepotty.web import tune as tune_mod

        frame, idx, total, fps, width, height = tune_mod.read_frame(file_path, index)
        detector, model_used = _get_tune_detector(model_name)
        with app.state.tune_infer_lock:
            objects = detector.detect_scene_objects(frame, top_n=top_n)
        return {
            "path": str(file_path),
            "index": idx,
            "total_frames": total or None,
            "fps": fps,
            "width": width,
            "height": height,
            "model": model_used,
            "detection_floor": TUNE_DETECTION_FLOOR,
            "objects": [
                scene_object_payload(class_name, confidence, bbox)
                for class_name, confidence, bbox in objects
            ],
        }

    def _detect_range_payload(
        file_path: Path,
        start: int,
        count: int,
        model_name: str,
    ) -> dict:
        """Detections for a contiguous run of frames in one batched forward.

        The tuner's background filler walks frames in order, so a window decodes
        sequentially (cheap) and runs through ``detect_batch`` as a single GPU
        forward under ``tune_infer_lock`` — the batched analogue of the per-frame
        ``/api/tune/detect``. Returns ``{model, frames: [<per-frame payload>...]}``
        with each entry shaped exactly like the boxes-only ``_detect_payload`` so
        the client can buffer them identically. Pose is not run here; it stays on
        the decoupled ``/api/tune/pose`` lane.
        """

        from detectivepotty.detect.yolo import FrameMeta
        from detectivepotty.web import tune as tune_mod

        frames, total, fps, width, height = tune_mod.read_frames(file_path, start, count)
        detector, model_used = _get_tune_detector(model_name)
        bgr_list = [frame for _idx, frame in frames]
        wall = datetime.now(timezone.utc)
        mono = time.monotonic()
        metas = [
            FrameMeta(frame_idx=idx, mono_ts=mono, wall_ts=wall) for idx, _frame in frames
        ]
        batch = getattr(detector, "detect_batch", None)
        with app.state.tune_infer_lock:
            if batch is not None:
                results = batch(bgr_list, metas)
            else:
                # A detector predating ``detect_batch`` (e.g. a test fake): loop
                # ``detect``; results are identical (per-image-independent boxes).
                results = [
                    detector.detect(
                        frame, frame_idx=idx, mono_ts=mono, wall_ts=wall
                    )
                    for idx, frame in frames
                ]
        resolved = app.state.tune_pose_resolved
        pose_available = resolved[1] if resolved is not None else True
        out_frames = [
            {
                "path": str(file_path),
                "index": idx,
                "total_frames": total or None,
                "fps": fps,
                "width": width,
                "height": height,
                "model": model_used,
                "detection_floor": TUNE_DETECTION_FLOOR,
                "detections": tune_mod.detections_payload(detections),
                "pose": [],
                "pose_available": pose_available,
            }
            for (idx, _frame), detections in zip(frames, results)
        ]
        return {"model": model_used, "frames": out_frames}

    def _iter_track_range(
        file_path: Path,
        start: int,
        count: int,
        model_name: str,
        *,
        sample_every: int,
        iou_threshold: float,
        max_age_frames: int,
        center_dist_gate: float,
    ):
        """Streaming generator core of the ``ours`` Track-range backend.

        Decodes ``[start, start+count)`` in ``tune_detection_batch_size`` chunks,
        batch-detects the **sampled** frames (``frame_idx % sample_every == 0`` in
        source numbering, matching the harvest scan), and replays each chunk's
        detections through a single persistent harvest ``Tracker`` **in ascending
        frame order** — so a track's id is consistent across the whole pass. Yields
        ``{"type":"frames","frames":[...]}`` per non-empty chunk (forward-fill /
        progress), then a final ``{"type":"done", ...stats}``. Because chunks and the
        frames within them are already ascending, draining this reproduces
        :func:`tune.track_detections`'s output byte-for-byte (see ``_track_range_payload``).
        Inference is serialized per chunk by ``tune_infer_lock`` (released between
        chunks so other detect requests can interleave). Works with any model
        including CoreML. ``count`` is pre-clamped by the caller.
        """

        from detectivepotty.detect.yolo import FrameMeta
        from detectivepotty.tracking import Tracker
        from detectivepotty.web import tune as tune_mod

        detector, model_used = _get_tune_detector(model_name)
        batch = getattr(detector, "detect_batch", None)
        decode_cap = max(1, app.state.config.global_settings.tune_detection_batch_size)
        stride = max(1, sample_every)

        tracker = Tracker(
            iou_threshold=iou_threshold,
            max_age_frames=max_age_frames,
            center_dist_gate=center_dist_gate,
        )
        out_frames: list[dict] = []
        fps = 0.0
        total = 0
        cursor = start
        end = start + count
        wall = datetime.now(timezone.utc)
        mono = time.monotonic()
        while cursor < end:
            chunk_count = min(decode_cap, end - cursor)
            try:
                frames, total, fps, _w, _h = tune_mod.read_frames(
                    file_path, cursor, chunk_count
                )
            except IndexError:
                break  # past EOF: track what we have
            sampled = [(idx, frame) for idx, frame in frames if idx % stride == 0]
            chunk_out: list[dict] = []
            if sampled:
                bgr_list = [frame for _idx, frame in sampled]
                metas = [
                    FrameMeta(frame_idx=idx, mono_ts=mono, wall_ts=wall)
                    for idx, _frame in sampled
                ]
                with app.state.tune_infer_lock:
                    if batch is not None:
                        results = batch(bgr_list, metas)
                    else:
                        results = [
                            detector.detect(
                                frame, frame_idx=idx, mono_ts=mono, wall_ts=wall
                            )
                            for idx, frame in sampled
                        ]
                for (idx, _frame), dets in zip(sampled, results):
                    record = tune_mod.track_step(tracker, idx, list(dets))
                    out_frames.append(record)
                    chunk_out.append(record)
            if chunk_out:
                yield {"type": "frames", "frames": chunk_out}
            # Advance by the frames actually decoded (handles short EOF reads).
            last_decoded = frames[-1][0]
            if last_decoded < cursor:  # pragma: no cover - defensive
                break
            cursor = last_decoded + 1

        stats = tune_mod.summarize_tracked_frames(
            out_frames,
            fps=fps or 30.0,
            total_frames=total or None,
            sample_every=stride,
            tracker="ours",
            iou_threshold=iou_threshold,
            max_age_frames=max_age_frames,
            center_dist_gate=center_dist_gate,
        )
        yield {
            "type": "done",
            "model": model_used,
            "start": start,
            "count": count,
            "fps": fps,
            "total_frames": total or None,
            "detection_floor": TUNE_DETECTION_FLOOR,
            "stats": stats,
        }

    def _track_range_payload(
        file_path: Path,
        start: int,
        count: int,
        model_name: str,
        *,
        sample_every: int,
        iou_threshold: float,
        max_age_frames: int,
        center_dist_gate: float,
    ) -> dict:
        """Non-streaming ``ours`` Track-range payload (drains :func:`_iter_track_range`).

        Kept byte-identical to the pre-streaming behaviour by collecting every
        ``frames`` record and merging the final ``done`` record — so the existing
        ``/api/tune/track-range`` contract + tests are unchanged.
        """

        frames: list[dict] = []
        done: dict = {}
        for rec in _iter_track_range(
            file_path,
            start,
            count,
            model_name,
            sample_every=sample_every,
            iou_threshold=iou_threshold,
            max_age_frames=max_age_frames,
            center_dist_gate=center_dist_gate,
        ):
            if rec["type"] == "frames":
                frames.extend(rec["frames"])
            elif rec["type"] == "done":
                done = rec
        return {
            "model": done.get("model", model_name),
            "start": start,
            "count": count,
            "fps": done.get("fps", 0.0),
            "total_frames": done.get("total_frames"),
            "detection_floor": TUNE_DETECTION_FLOOR,
            "frames": frames,
            "stats": done.get("stats", {}),
        }

    def _iter_track_range_ultralytics(
        file_path: Path,
        start: int,
        count: int,
        model_name: str,
        *,
        tracker: str,
        sample_every: int,
        ultra_params: TuneUltralyticsTrackerParams,
    ):
        """Streaming generator core of the ``.pt``-only Ultralytics tracker backend.

        Streams the **sampled** frames (harvest scan stride) sequentially through
        ``model.track(persist=True, tracker=<yaml>)`` so the built-in motion/appearance
        association assigns persistent IDs, yielding ``{"type":"frames","frames":[...]}``
        per decode chunk (forward-fill / progress) then a final ``{"type":"done",
        ...stats}`` via :func:`tune.summarize_tracked_frames`. ``botsort_reid`` is
        BoT-SORT with ``with_reid: True`` (appearance ReID from the detector's own
        features). A **fresh** ``YOLO`` is built per call so ``persist`` tracker state
        never leaks across requests; the request still feeds sampled frames in-order,
        but ``tune_infer_lock`` is released before yielding each chunk so a slow
        client cannot block unrelated inference. Draining this reproduces
        ``_track_range_ultralytics_payload``. ``count`` is pre-clamped by the caller.
        """

        from detectivepotty.device import resolve_device
        from detectivepotty.web import tune as tune_mod

        # Allow-list + `.pt` guard (also enforced by the endpoint, kept here so the
        # generator is safe if ever driven directly).
        if model_name not in app.state.tune_models:
            raise ValueError(f"unknown model: {model_name}")
        if not model_name.endswith(".pt"):
            raise ValueError("Ultralytics tracking requires a .pt model")
        if not ultralytics_tracking_available():
            raise ValueError("Ultralytics tracking unavailable (install `lap`)")

        from ultralytics import YOLO

        device = resolve_device(app.state.config.global_settings.device)
        long_edge = app.state.config.global_settings.inference_long_edge_px
        decode_cap = max(1, app.state.config.global_settings.tune_detection_batch_size)
        stride = max(1, sample_every)

        # Resolve the tracker yaml. bytetrack/botsort ship with Ultralytics; when
        # the UI overrides any thresholds (or asks for ReID) we copy the bundled
        # yaml to a temp file and patch only those keys for this request.
        from ultralytics.utils import ROOT as _ULTRA_ROOT

        trackers_dir = Path(_ULTRA_ROOT) / "cfg" / "trackers"
        tracker_yaml, tmp_dir = _ultralytics_tracker_yaml(
            trackers_dir, tracker, ultra_params
        )

        out_frames: list[dict] = []
        fps = 0.0
        total = 0
        try:
            model = YOLO(model_name)
            dog_idxs = _ultralytics_dog_class_indices(
                model, app.state.config.global_settings.dog_alias_classes
            )
            cursor = start
            end = start + count
            while cursor < end:
                chunk_count = min(decode_cap, end - cursor)
                try:
                    frames, total, fps, _w, _h = tune_mod.read_frames(
                        file_path, cursor, chunk_count
                    )
                except IndexError:
                    break  # past EOF: track what we have
                chunk_out: list[dict] = []
                with app.state.tune_infer_lock:
                    for idx, frame in frames:
                        if idx % stride != 0:
                            continue
                        result = model.track(
                            frame,
                            persist=True,
                            tracker=tracker_yaml,
                            classes=dog_idxs,
                            imgsz=long_edge,
                            device=device,
                            conf=ultra_params.conf,
                            verbose=False,
                        )[0]
                        record = {
                            "index": idx,
                            "detections": _ultralytics_boxes(result),
                        }
                        out_frames.append(record)
                        chunk_out.append(record)
                if chunk_out:
                    yield {"type": "frames", "frames": chunk_out}
                last_decoded = frames[-1][0]
                if last_decoded < cursor:  # pragma: no cover - defensive
                    break
                cursor = last_decoded + 1
        finally:
            if tmp_dir is not None:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

        stats = tune_mod.summarize_tracked_frames(
            out_frames,
            fps=fps or 30.0,
            total_frames=total or None,
            sample_every=stride,
            tracker=tracker,
        )
        stats["ultralytics"] = ultra_params.payload(tracker)
        yield {
            "type": "done",
            "model": model_name,
            "start": start,
            "count": count,
            "fps": fps,
            "total_frames": total or None,
            "detection_floor": ultra_params.conf,
            "stats": stats,
        }

    def _track_range_ultralytics_payload(
        file_path: Path,
        start: int,
        count: int,
        model_name: str,
        *,
        tracker: str,
        sample_every: int,
        ultra_params: TuneUltralyticsTrackerParams,
    ) -> dict:
        """Non-streaming Ultralytics Track-range payload (drains the generator)."""

        frames: list[dict] = []
        done: dict = {}
        for rec in _iter_track_range_ultralytics(
            file_path,
            start,
            count,
            model_name,
            tracker=tracker,
            sample_every=sample_every,
            ultra_params=ultra_params,
        ):
            if rec["type"] == "frames":
                frames.extend(rec["frames"])
            elif rec["type"] == "done":
                done = rec
        return {
            "model": done.get("model", model_name),
            "start": start,
            "count": count,
            "fps": done.get("fps", 0.0),
            "total_frames": done.get("total_frames"),
            "detection_floor": done.get("detection_floor", TUNE_DETECTION_FLOOR),
            "frames": frames,
            "stats": done.get("stats", {}),
        }

    def _pose_payload(
        file_path: Path,
        index: int,
        boxes: list[list[float]],
    ) -> dict:
        """Run pose for ``boxes`` on one frame — the decoupled pose pass.

        Drives ``POST /api/tune/pose``. The frame is decoded *outside*
        ``tune_infer_lock`` (CPU/IO shouldn't block the GPU); only inference is
        serialized. Inference failure downgrades pose app-wide (same contract as
        ``_detect_payload``) so the UI stops promising pose instead of retrying
        the heavy path every frame.
        """

        from detectivepotty.web import tune as tune_mod

        estimator, pose_available = _get_tune_pose()
        if estimator is None:
            return {"index": index, "pose": [], "pose_available": False}

        frame, idx, _total, _fps, _w, _h = tune_mod.read_frame(file_path, index)
        pose_list: list = []
        with app.state.tune_infer_lock:
            try:
                pose_list = tune_mod.pose_payload_for_boxes(
                    estimator, frame, boxes, frame_idx=idx
                )
            except Exception:
                logger.warning("pose inference failed for tune pose", exc_info=True)
                pose_list = []
        return {"index": idx, "pose": pose_list, "pose_available": pose_available}

    def _pose_range_payload(
        file_path: Path,
        frames_in: list[tuple[int, list[list[float]]]],
    ) -> dict:
        """Batched pose over a run of frames — one ``estimate_batch`` GPU forward.

        Drives ``POST /api/tune/pose-range``. Each frame is decoded *outside*
        ``tune_infer_lock`` (CPU/IO shouldn't block the GPU); a frame that can't be
        decoded is carried as ``None`` so it still appears in the response (with
        empty pose) and the client can mark it terminal instead of retrying it
        forever. Only the single combined ``estimate_batch`` is serialized by the
        lock. Inference failure downgrades pose app-wide (same contract as
        ``_pose_payload``). Returns ``{frames: [{index, pose, pose_available}...]}``
        with one entry per requested frame, in request order.
        """

        from detectivepotty.web import tune as tune_mod

        estimator, pose_available = _get_tune_pose()
        if estimator is None:
            return {
                "frames": [
                    {"index": index, "pose": [], "pose_available": False}
                    for index, _boxes in frames_in
                ]
            }

        items: list[tuple[int, object, list[list[float]]]] = []
        for index, boxes in frames_in:
            try:
                frame, _idx, _total, _fps, _w, _h = tune_mod.read_frame(file_path, index)
            except IndexError:
                # Un-decodable (EOF / corrupt): keep the index but pose nothing, so
                # the client marks it terminal rather than re-requesting forever.
                frame = None
            items.append((index, frame, boxes))

        with app.state.tune_infer_lock:
            try:
                results = tune_mod.pose_payload_for_frames(estimator, items)
            except Exception:
                logger.warning("pose inference failed for tune pose range", exc_info=True)
                results = [(index, []) for index, _frame, _boxes in items]
        return {
            "frames": [
                {"index": index, "pose": pose_list, "pose_available": pose_available}
                for index, pose_list in results
            ]
        }

    @app.get("/api/tune/models")
    def tune_models() -> dict:
        """The model picker's allow-list: discovered weights + the configured one."""

        return {
            "models": list(app.state.tune_models),
            "default": app.state.tune_default_model,
            "coreml_batch": _coreml_batch_map(app.state.tune_models),
        }

    @app.post("/api/tune/export-coreml")
    async def tune_export_coreml(req: ExportCoremlRequest) -> dict:
        """Export a discovered ``.pt`` model to a GPU-safe CoreML ``.mlpackage``.

        Drives the tuner's "Export to CoreML (GPU)" button. The source must be a
        ``.pt`` already in the allow-list (``.mlpackage`` / unknown → 400). The
        export is heavy (~20-60s) and macOS-only, so it runs in a threadpool
        serialized by ``tune_export_lock``. On success the new ``.mlpackage`` is
        added to the allow-list and returned for immediate selection.
        """

        name = req.model
        if name not in app.state.tune_models:
            raise HTTPException(status_code=400, detail="unknown model")
        if not name.endswith(".pt"):
            raise HTTPException(status_code=400, detail="model is not a .pt")

        from detectivepotty.detect import coreml_export

        def _run() -> str:
            with app.state.tune_export_lock:
                return str(
                    coreml_export.export_coreml(
                        name,
                        imgsz=config.global_settings.inference_long_edge_px,
                        batch=config.global_settings.tune_detection_batch_size,
                    )
                )

        try:
            result = await run_in_threadpool(_run)
        except Exception as exc:  # noqa: BLE001 - surface any export failure as 500
            logger.exception("CoreML export failed for %s", name)
            raise HTTPException(status_code=500, detail="coreml export failed") from exc

        if result not in app.state.tune_models:
            app.state.tune_models = [*app.state.tune_models, result]
        return {
            "model": result,
            "models": list(app.state.tune_models),
            "default": app.state.tune_default_model,
            "coreml_batch": _coreml_batch_map(app.state.tune_models),
        }

    @app.get("/api/tune/meta")
    def tune_meta(path: str) -> dict:
        """Clip fps/frame-count/dimensions for index<->time mapping (no inference)."""

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        try:
            total, fps, width, height, duration = tune_mod.read_meta(file_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="clip not available") from exc
        return {
            "path": str(file_path),
            "total_frames": total or None,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
        }

    @app.get("/api/tune/clip")
    def tune_clip(path: str) -> FileResponse:
        """Stream the raw clip for the ``<video>`` element (Range-seekable)."""

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        # FileResponse honours the Range header (206 partial content) so the
        # <video> element can seek; no-store avoids caching a large local clip
        # across selections.
        return no_store_video_response(file_path)

    @app.get("/api/tune/frame")
    async def tune_frame(
        path: str,
        index: Annotated[int, Query(ge=0)] = 0,
        pose: Annotated[int, Query(ge=0, le=1)] = 0,
        model: str | None = None,
    ) -> dict:
        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)

        try:
            return await run_in_threadpool(
                _detect_payload,
                file_path,
                index,
                model_name,
                bool(pose),
                True,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/detect")
    async def tune_detect(
        path: str,
        index: Annotated[int, Query(ge=0)] = 0,
        pose: Annotated[int, Query(ge=0, le=1)] = 0,
        model: str | None = None,
    ) -> dict:
        """Detections (+optional pose) for one frame — no image. The buffer source."""

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)

        try:
            return await run_in_threadpool(
                _detect_payload,
                file_path,
                index,
                model_name,
                bool(pose),
                False,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/scene")
    async def tune_scene(
        path: str,
        index: Annotated[int, Query(ge=0)] = 0,
        top_n: Annotated[int, Query(ge=1, le=20)] = 8,
        model: str | None = None,
    ) -> dict:
        """Top-N all-class detections for one frame (diagnostic, no dog filter)."""

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)

        try:
            return await run_in_threadpool(
                _scene_payload,
                file_path,
                index,
                model_name,
                top_n,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/detect_range", include_in_schema=False)
    @app.get("/api/tune/detect-range")
    async def tune_detect_range(
        path: str,
        start: Annotated[int, Query(ge=0)] = 0,
        count: Annotated[int, Query(ge=1)] = 1,
        model: str | None = None,
    ) -> dict:
        """Batched detections for a contiguous ``[start, start+count)`` window.

        The filler's backfill source: one sequential decode + one ``detect_batch``
        forward instead of ``count`` single-frame round-trips, which is what lifts
        GPU utilization off the batch-1 floor. ``count`` is capped by
        ``tune_detection_batch_size`` so a client can't request an unbounded run.
        """

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)
        cap = max(1, app.state.config.global_settings.tune_detection_batch_size)
        bounded = min(count, cap)

        try:
            return await run_in_threadpool(
                _detect_range_payload,
                file_path,
                start,
                bounded,
                model_name,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/track_range", include_in_schema=False)
    @app.get("/api/tune/track-range")
    async def tune_track_range(
        path: str,
        start: Annotated[int, Query(ge=0)] = 0,
        count: Annotated[int, Query(ge=1)] = 1,
        model: str | None = None,
        tracker: str = "ours",
        sample_every: Annotated[int, Query(ge=1, le=60)] = 5,
        iou_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.3,
        max_age_frames: Annotated[int, Query(ge=0, le=300)] = 15,
        center_dist_gate: Annotated[float, Query(ge=0.0, le=20.0)] = 1.5,
        ultra_conf: Annotated[float, Query(ge=0.0, le=1.0)] = TUNE_DETECTION_FLOOR,
        track_high_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        track_low_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        new_track_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        track_buffer: Annotated[int | None, Query(ge=0, le=10000)] = None,
        match_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        proximity_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        appearance_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    ) -> dict:
        """Track a ``[start, start+count)`` range with the chosen tracker backend.

        ``tracker=off``/``ours`` use the harvest IoU ``Tracker`` replay (decode +
        detect the sampled frames, replay through ``Tracker`` with the supplied
        knobs) — works with every model including CoreML ``.mlpackage``.
        ``tracker=bytetrack``/``botsort``/``botsort_reid`` use Ultralytics native
        tracking (``.pt``-only, ``botsort_reid`` adds appearance ReID). Either way
        returns persistent per-frame track-ID boxes + de-fragmentation stats
        (distinct tracks, spans, presence windows, spans-per-window). ``count`` is
        capped by ``TUNE_TRACK_MAX_FRAMES``.
        """

        request = _track_range_request(
            path,
            count,
            model,
            tracker,
            ultra_conf,
            track_high_thresh,
            track_low_thresh,
            new_track_thresh,
            track_buffer,
            match_thresh,
            proximity_thresh,
            appearance_thresh,
        )

        if tracker in _ULTRALYTICS_TRACKERS:
            _ensure_ultralytics_tracking(request.model_name)
            try:
                return await run_in_threadpool(
                    _track_range_ultralytics_payload,
                    request.file_path,
                    start,
                    request.bounded_count,
                    request.model_name,
                    tracker=tracker,
                    sample_every=sample_every,
                    ultra_params=request.ultra_params,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except (FileNotFoundError, IndexError) as exc:
                raise HTTPException(
                    status_code=404, detail="frame not available"
                ) from exc

        try:
            return await run_in_threadpool(
                _track_range_payload,
                request.file_path,
                start,
                request.bounded_count,
                request.model_name,
                sample_every=sample_every,
                iou_threshold=iou_threshold,
                max_age_frames=max_age_frames,
                center_dist_gate=center_dist_gate,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/track_range_stream", include_in_schema=False)
    @app.get("/api/tune/track-range-stream")
    async def tune_track_range_stream(
        path: str,
        start: Annotated[int, Query(ge=0)] = 0,
        count: Annotated[int, Query(ge=1)] = 1,
        model: str | None = None,
        tracker: str = "ours",
        sample_every: Annotated[int, Query(ge=1, le=60)] = 5,
        iou_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.3,
        max_age_frames: Annotated[int, Query(ge=0, le=300)] = 15,
        center_dist_gate: Annotated[float, Query(ge=0.0, le=20.0)] = 1.5,
        ultra_conf: Annotated[float, Query(ge=0.0, le=1.0)] = TUNE_DETECTION_FLOOR,
        track_high_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        track_low_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        new_track_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        track_buffer: Annotated[int | None, Query(ge=0, le=10000)] = None,
        match_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        proximity_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        appearance_thresh: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    ) -> StreamingResponse:
        """Stream a Track-range pass as newline-delimited JSON (forward-fill + progress).

        The streaming sibling of :func:`tune_track_range`: identical params/backends,
        but instead of buffering the whole pass it emits one
        ``{"type":"frames","frames":[...]}`` line per decode chunk as the in-order
        0→end pass computes (so the client can fill the timeline + show progress live),
        then a final ``{"type":"done", ...stats}`` line. All guards (unknown tracker,
        bad path, ultra ``.pt``/availability) run **before** streaming starts so they
        still surface as a 400; a mid-stream failure emits a final
        ``{"type":"error","detail":...}`` line. ``count`` is capped by
        ``TUNE_TRACK_MAX_FRAMES``. The sync generator runs in the threadpool, so its
        blocking decode/inference never blocks the event loop.
        """

        request = _track_range_request(
            path,
            count,
            model,
            tracker,
            ultra_conf,
            track_high_thresh,
            track_low_thresh,
            new_track_thresh,
            track_buffer,
            match_thresh,
            proximity_thresh,
            appearance_thresh,
        )

        if tracker in _ULTRALYTICS_TRACKERS:
            _ensure_ultralytics_tracking(request.model_name)
            gen = _iter_track_range_ultralytics(
                request.file_path,
                start,
                request.bounded_count,
                request.model_name,
                tracker=tracker,
                sample_every=sample_every,
                ultra_params=request.ultra_params,
            )
        else:
            gen = _iter_track_range(
                request.file_path,
                start,
                request.bounded_count,
                request.model_name,
                sample_every=sample_every,
                iou_threshold=iou_threshold,
                max_age_frames=max_age_frames,
                center_dist_gate=center_dist_gate,
            )

        async def _ndjson():
            try:
                async for rec in iterate_in_threadpool(gen):
                    yield json.dumps(rec) + "\n"
            except (ValueError, FileNotFoundError, IndexError) as exc:
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).exception("track_range_stream failed")
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            finally:
                close = getattr(gen, "close", None)
                if close is not None:
                    close()

        return StreamingResponse(_ndjson(), media_type="application/x-ndjson")

    @app.post("/api/tune/pose")
    async def tune_pose(req: TunePoseRequest) -> dict:
        """Pose keypoints for client-supplied boxes on one frame — no YOLO re-run.

        The decoupled, proactive pose pass: the tuner sends the detection boxes it
        already buffered and gets back keypoints, so pose precomputes behind YOLO
        without redoing detection. Returns ``{index, pose, pose_available}``;
        ``pose_available=False`` (with empty pose) when the pose backend isn't
        installed/working, which tells the client to stop the pose pass.
        """

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(req.path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc

        try:
            return await run_in_threadpool(
                _pose_payload,
                file_path,
                req.index,
                req.boxes,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.post("/api/tune/pose_range", include_in_schema=False)
    @app.post("/api/tune/pose-range")
    async def tune_pose_range(req: TunePoseRangeRequest) -> dict:
        """Batched pose for a run of frames' buffered boxes — one GPU forward.

        The pose analogue of ``/api/tune/detect-range``: instead of one pose
        request per frame (the batch-1 floor that measured ~9-14x slower on the
        SuperAnimal backend), the tuner sends a window of frames + their boxes and
        pose runs as a single batched ``estimate_batch``. Frames are capped by
        ``tune_detection_batch_size`` and total crops by ``TUNE_POSE_MAX_CROPS`` so
        one request can't monopolize the inference lock.
        """

        from detectivepotty.web import tune as tune_mod

        try:
            file_path = tune_mod.resolve_tune_file(req.path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc

        frame_cap = max(1, app.state.config.global_settings.tune_detection_batch_size)
        frames_in: list[tuple[int, list[list[float]]]] = []
        crops = 0
        for frame in req.frames[:frame_cap]:
            if crops >= TUNE_POSE_MAX_CROPS:
                break
            remaining = TUNE_POSE_MAX_CROPS - crops
            boxes = frame.boxes[:remaining]
            frames_in.append((frame.index, boxes))
            crops += len(boxes)
            if crops >= TUNE_POSE_MAX_CROPS:
                break

        try:
            return await run_in_threadpool(
                _pose_range_payload,
                file_path,
                frames_in,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc
