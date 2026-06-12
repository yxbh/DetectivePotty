"""FastAPI app for local DetectivePotty event review."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import yaml

from detectivepotty.config import Config, load_config
from detectivepotty.events import Label, LabelStatus
from detectivepotty.web.dataset_index import DatasetIndex, fixed_media_path, media_path
from starlette.datastructures import MutableHeaders


logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

# Env var the reload factory reads to locate the config. uvicorn's --reload
# re-imports the app in a worker subprocess that never re-runs the CLI, so the
# config path has to travel via the environment rather than a call argument.
CONFIG_ENV_VAR = "DETECTIVEPOTTY_CONFIG"

# How often the live-stream endpoint re-scans the dataset directory for new
# events. The `serve` process observes the filesystem because the `run`
# pipeline writes events from a separate process; 2s keeps perceived latency
# low without hammering the disk.
STREAM_POLL_SECONDS = 2.0

# Event media (clips, frames, crops) is content-stable per event_id, so let the
# browser cache it instead of re-downloading on every navigation. Starlette's
# FileResponse has no conditional-304 support, so a real max-age is required.
# Reruns can replace media under a reused event_id; the frontend appends the
# event's media_version (?v=) so a fresh token bypasses this cache.
MEDIA_CACHE_CONTROL = "private, max-age=3600"

_BUILD_MISSING_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>DetectivePotty Review</title>
  </head>
  <body style="font-family: system-ui, sans-serif; margin: 3rem; color: #ddd;
               background: #10141c;">
    <h1>DetectivePotty Review</h1>
    <p>The review portal frontend has not been built yet.</p>
    <p>Run the following, then reload:</p>
    <pre>cd src/detectivepotty/web/frontend
npm install
npm run build</pre>
  </body>
</html>
"""


# Detection floor for the in-browser tuner. The detector runs at this low
# confidence so borderline boxes are still returned; the client-side slider
# (whose minimum is this floor) decides green-kept vs red-dropped without any
# re-inference. Anything under the floor was never produced and cannot be
# recovered by lowering the slider — hence the slider can't go below it.
TUNE_DETECTION_FLOOR = 0.05

# Upper bound on total pose crops the batched pose pass (`POST /api/tune/pose_range`)
# will run for one request, regardless of how many frames/boxes the client sends.
# `estimate_batch` chunks by `classifier_batch_size` (the GPU forward size), so this
# only bounds how long one request can hold `tune_infer_lock` (an AbortController
# can't cancel an already-running server thread) — it stops a buggy/hostile client
# from monopolizing the GPU after a scrub/model change. The normal client sends
# ~8 frames x ~1 box, far under this.
TUNE_POSE_MAX_CROPS = 64

# Upper bound on the number of source frames one "Track range" request will decode.
# Tracking must decode the whole requested range sequentially (in `sample_every`
# stride) under the single clip reader, so this caps how long one request runs /
# how much it decodes. ~6000 frames ≈ 200s at 30fps — generous for eyeballing a
# dog visit, while keeping the synchronous request bounded. The client marks an
# in/out sub-range; the server clamps `count` to this.
TUNE_TRACK_MAX_FRAMES = 6000

# Suffix -> MIME for the tuner clip endpoint. Browsers play mp4/mov/webm
# natively; mkv/avi are served with a correct type even if a given browser
# can't decode them.
_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}

# Tune "Track range" tracker backends. ``off``/``ours`` use the harvest IoU
# ``Tracker`` replay (every model incl. CoreML); the three native values map to
# Ultralytics built-in trackers (``.pt``-only — Ultralytics tracking won't run on
# a CoreML package). ``botsort_reid`` is BoT-SORT with appearance ReID enabled.
TUNE_TRACKERS = ("off", "ours", "bytetrack", "botsort", "botsort_reid")
_ULTRALYTICS_TRACKERS = ("bytetrack", "botsort", "botsort_reid")


@dataclass(frozen=True, slots=True)
class TuneUltralyticsTrackerParams:
    """Per-run Ultralytics tracking knobs exposed by the Tune UI.

    ``conf`` is passed to ``YOLO.track``. The other fields are optional overrides
    for the bundled ByteTrack/BoT-SORT YAML; ``None`` means keep the YAML default.
    """

    conf: float = TUNE_DETECTION_FLOOR
    track_high_thresh: float | None = None
    track_low_thresh: float | None = None
    new_track_thresh: float | None = None
    track_buffer: int | None = None
    match_thresh: float | None = None
    proximity_thresh: float | None = None
    appearance_thresh: float | None = None

    def yaml_overrides(self, tracker: str) -> dict[str, float | int | bool]:
        overrides: dict[str, float | int | bool] = {}
        for key in (
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
        ):
            value = getattr(self, key)
            if value is not None:
                overrides[key] = value
        if tracker in ("botsort", "botsort_reid"):
            for key in ("proximity_thresh", "appearance_thresh"):
                value = getattr(self, key)
                if value is not None:
                    overrides[key] = value
        if tracker == "botsort_reid":
            overrides["with_reid"] = True
        return overrides

    def payload(self, tracker: str) -> dict[str, float | int | bool | None]:
        return {
            "conf": self.conf,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "proximity_thresh": (
                self.proximity_thresh if tracker in ("botsort", "botsort_reid") else None
            ),
            "appearance_thresh": (
                self.appearance_thresh if tracker in ("botsort", "botsort_reid") else None
            ),
            "with_reid": tracker == "botsort_reid",
        }


def _ultralytics_tracking_available() -> bool:
    """True when Ultralytics native tracking can run (its ``lap`` dep imports).

    Ultralytics' association step needs ``lap`` (linear assignment); it is a core
    dependency, but the endpoint feature-detects it so a stripped env degrades to a
    clear 400 instead of an opaque import error mid-request.
    """

    import importlib.util

    return importlib.util.find_spec("lap") is not None


def _ultralytics_dog_class_indices(
    model: object, alias_classes: Iterable[str] = ()
) -> list[int]:
    """Class indices accepted as dogs for ``model`` (``dog`` + any ``alias_classes``).

    Alias classes (e.g. ``sheep``/``cow`` — dog-confusable but yard-implausible) are
    folded in so the native Ultralytics tracker recovers the same boxes the detector
    does. Falls back to COCO ``16`` (``dog``) when no class names match.
    """

    accepted = {"dog", *(str(c).lower() for c in alias_classes)}
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        idxs = [int(i) for i, n in names.items() if str(n).lower() in accepted]
    else:  # pragma: no cover - list-style names are rare
        idxs = [i for i, n in enumerate(names) if str(n).lower() in accepted]
    return idxs or [16]


def _ultralytics_tracker_yaml(
    trackers_dir: Path,
    tracker: str,
    params: TuneUltralyticsTrackerParams,
) -> tuple[str, Path | None]:
    """Return ``(tracker_yaml, temp_dir)`` for one Ultralytics tracking request."""

    if tracker == "bytetrack":
        base_yaml = trackers_dir / "bytetrack.yaml"
    elif tracker in ("botsort", "botsort_reid"):
        base_yaml = trackers_dir / "botsort.yaml"
    else:  # pragma: no cover - guarded by endpoints
        raise ValueError(f"unknown ultralytics tracker: {tracker}")

    overrides = params.yaml_overrides(tracker)
    if not overrides:
        return str(base_yaml), None

    with base_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid tracker yaml: {base_yaml}")
    data.update(overrides)

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="dp_tracker_"))
    out = tmp_dir / f"{tracker}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(out), tmp_dir


def _ultralytics_boxes(result: object) -> list[dict]:
    """Map one Ultralytics tracking ``Results`` to Tune track-box dicts.

    Returns ``[{x1,y1,x2,y2,confidence,class_name,track_id}...]`` in original-image
    coordinates (Ultralytics already maps boxes back from the letterboxed input).
    Boxes the tracker hasn't assigned an ID yet (``boxes.id is None``) are skipped —
    only persistent tracks contribute to the overlay + de-fragmentation stats.
    """

    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    ids = boxes.id.tolist()
    out: list[dict] = []
    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        out.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(conf),
                "class_name": "dog",
                "track_id": str(int(tid)),
            }
        )
    return out


class LabelUpdate(BaseModel):
    label: Label
    label_status: LabelStatus
    note: str | None = Field(default=None, max_length=2000)
    dog: str | None = Field(default=None, max_length=200)


class ExportCoremlRequest(BaseModel):
    model: str = Field(max_length=500)


class TunePoseRequest(BaseModel):
    """Body for ``POST /api/tune/pose`` — the decoupled pose pass.

    ``boxes`` are the ``[x1, y1, x2, y2]`` detections the tuner already buffered,
    so pose runs without re-running YOLO. Bounded to keep a hostile/buggy client
    from scheduling unbounded inference work.
    """

    path: str = Field(max_length=4096)
    index: int = Field(ge=0)
    boxes: list[list[float]] = Field(default_factory=list, max_length=64)


class TunePoseRangeFrame(BaseModel):
    """One frame's buffered boxes within a batched pose request."""

    index: int = Field(ge=0)
    boxes: list[list[float]] = Field(default_factory=list, max_length=64)


class TunePoseRangeRequest(BaseModel):
    """Body for ``POST /api/tune/pose_range`` — the batched pose pass.

    Carries the buffered boxes for a run of frames so pose runs as **one batched
    GPU forward across the whole window** instead of one request per frame (the
    SuperAnimal backend measured ~9-14x faster batched than the batch-1 floor).
    Bounded (frame count + per-frame boxes) so a hostile/buggy client can't
    schedule unbounded work; the server further caps total crops
    (``TUNE_POSE_MAX_CROPS``).
    """

    path: str = Field(max_length=4096)
    frames: list[TunePoseRangeFrame] = Field(default_factory=list, max_length=64)


class _ApiNoStoreMiddleware:
    """Default ``Cache-Control: no-store`` for /api/ responses.

    Implemented as pure ASGI (not ``BaseHTTPMiddleware``) because the latter
    buffers the whole response body, which breaks the streaming
    ``GET /api/stream`` SSE endpoint. This only rewrites the response *start*
    headers, so streamed bodies flush incrementally. Endpoints that set their
    own Cache-Control (e.g. the SSE stream's ``no-cache``) are left untouched.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                if "cache-control" not in headers:
                    headers["cache-control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_wrapper)


async def _event_stream(dataset_index, is_disconnected, *, sleep=asyncio.sleep, poll_seconds=None):
    """Async generator yielding SSE frames for newly-recorded events.

    Pulled out of the route handler so the diff/seed/heartbeat logic is unit
    testable without spinning up an HTTP server (``TestClient`` cannot cleanly
    consume an infinite stream). ``is_disconnected`` is an async predicate
    (``request.is_disconnected`` in production) and ``sleep`` is injectable so
    tests can drive it without real delays.
    """
    if poll_seconds is None:
        poll_seconds = STREAM_POLL_SECONDS
    try:
        records = await run_in_threadpool(dataset_index.scan)
    except Exception:  # pragma: no cover - defensive
        records = []
    known: set[str] = {record.event_id for record in records}
    yield f"event: ready\ndata: {json.dumps({'count': len(known)})}\n\n"

    while True:
        if await is_disconnected():
            break
        await sleep(poll_seconds)
        fresh: list = []
        try:
            records = await run_in_threadpool(dataset_index.scan)
            fresh = [r for r in records if r.event_id not in known]
        except Exception:  # pragma: no cover - defensive
            fresh = []
        # Oldest-first so the client prepends newest last.
        for record in reversed(fresh):
            try:
                summary = await run_in_threadpool(dataset_index.summary, record)
            except Exception:  # pragma: no cover - defensive
                # Leave it unknown so it is retried on the next scan rather than
                # permanently dropped.
                continue
            known.add(record.event_id)
            payload = json.dumps(summary)
            yield f"id: {record.event_id}\nevent: new\ndata: {payload}\n\n"
        # Always heartbeat (even after a failed scan) so proxies and the browser
        # don't idle the connection shut.
        yield ": ping\n\n"


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


def create_app(
    config: Config,
    *,
    tune_detector: object | None = None,
    tune_pose_estimator: object | None = None,
) -> FastAPI:
    dataset_index = DatasetIndex(config.global_settings.dataset_dir)
    dogs = list(config.global_settings.dogs)
    app = FastAPI(title="DetectivePotty", version="0.1.0")
    app.state.dataset_index = dataset_index
    app.state.dogs = dogs
    # Tuner state. Detector/pose are built lazily on first /api/tune/frame so app
    # creation stays cheap and offline (tests inject fakes here instead). All
    # inference is serialized by ``tune_infer_lock`` — torch/MPS isn't reliably
    # safe for concurrent model execution, matching the pipeline's invariant.
    from detectivepotty.web.tune import collect_tune_models, collect_tune_roots

    app.state.config = config
    app.state.tune_roots = collect_tune_roots(config)
    # Root the range-labeling API discovers harvested clips under. Kept separate
    # from the tuner's browse roots (which include it) so /api/label only ever
    # exposes harvested clip dirs, never the wider dataset/data tree.
    app.state.harvest_root = config.global_settings.harvest_dir
    app.state.tune_default_model = config.global_settings.model_name
    # Per-model detector cache (model string -> DogDetector), built lazily under
    # ``tune_detector_lock``. An injected detector (tests) seeds the cache for the
    # default model and pins the allow-list to just that model, so no scanning or
    # real model build happens offline.
    if tune_detector is not None:
        app.state.tune_detectors = {app.state.tune_default_model: tune_detector}
        app.state.tune_models = [app.state.tune_default_model]
    else:
        app.state.tune_detectors = {}
        app.state.tune_models = collect_tune_models(config)
    app.state.tune_detector_lock = threading.Lock()
    app.state.tune_infer_lock = threading.Lock()
    app.state.tune_pose_lock = threading.Lock()
    # Serializes CoreML exports (heavy, macOS-only) triggered from the tuner UI so
    # only one runs at a time.
    app.state.tune_export_lock = threading.Lock()
    # Resolved as (estimator | None, available). Seeded when a fake is injected.
    app.state.tune_pose_resolved = (
        (tune_pose_estimator, True) if tune_pose_estimator is not None else None
    )
    app.add_middleware(_ApiNoStoreMiddleware)

    # The built Svelte app references hashed files under /assets. Mount it only
    # when the build exists so app creation still succeeds in CI / fresh
    # checkouts where dist/ (gitignored) has not been built yet.
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        built = FRONTEND_DIST / "index.html"
        if built.is_file():
            return FileResponse(built, headers={"Cache-Control": "no-store"})
        return HTMLResponse(_BUILD_MISSING_HTML)

    @app.get("/api/dogs")
    def list_dogs() -> dict:
        return {"dogs": dogs}

    @app.get("/api/events")
    def list_events(
        response: Response,
        camera: str | None = None,
        label_status: LabelStatus | None = None,
        date: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict]:
        records = dataset_index.scan()
        unfiltered_total = len(records)
        summaries = dataset_index.list_summaries(
            camera=camera,
            label_status=label_status,
            date=date,
            records=records,
        )
        total = len(summaries)
        page = summaries[offset : offset + limit]
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Unfiltered-Count"] = str(unfiltered_total)
        response.headers["X-Limit"] = str(limit)
        response.headers["X-Offset"] = str(offset)
        logger.info(
            "served events page=%d filtered=%d total=%d (camera=%s status=%s date=%s)",
            len(page),
            total,
            unfiltered_total,
            camera,
            label_status.value if label_status else None,
            date,
        )
        return page

    @app.get("/api/stream")
    async def stream_events(request: Request) -> StreamingResponse:
        """Server-Sent Events feed of newly-recorded potty events.

        The `run` pipeline writes events to disk from a separate process and
        always renames ``metadata.json`` into place last (after clip/frames/
        crops/overlays), so any event a scan observes is already complete. The
        generator seeds the set of known event_ids on connect (no backfill
        spam), then diffs each scan and pushes only genuinely-new event_ids.
        Reconnect gaps are reconciled client-side via a one-shot ``/api/events``
        poll on (re)connect, so re-seeding here cannot silently drop events.
        """
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            _event_stream(dataset_index, request.is_disconnected),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/api/events/{event_id}")
    def get_event(event_id: str) -> dict:
        record = _event_or_404(dataset_index, event_id)
        return dataset_index.detail(record)

    @app.get("/api/events/{event_id}/media/clip")
    def get_clip(event_id: str) -> FileResponse:
        record = _event_or_404(dataset_index, event_id)
        path = fixed_media_path(record, "clip.mp4", missing_ok=True)
        if path is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return FileResponse(
            path, media_type="video/mp4", headers={"Cache-Control": MEDIA_CACHE_CONTROL}
        )

    @app.get("/api/events/{event_id}/media/protect")
    def get_protect_recording(event_id: str) -> FileResponse:
        record = _event_or_404(dataset_index, event_id)
        path = fixed_media_path(record, "protect_recording.mp4", missing_ok=True)
        if path is None:
            raise HTTPException(status_code=404, detail="protect recording not found")
        return FileResponse(
            path, media_type="video/mp4", headers={"Cache-Control": MEDIA_CACHE_CONTROL}
        )

    @app.get("/api/events/{event_id}/frames/{name:path}")
    def get_frame(event_id: str, name: str) -> FileResponse:
        return _serve_image(dataset_index, event_id, "frames", name)

    @app.get("/api/events/{event_id}/crops/{name:path}")
    def get_crop(event_id: str, name: str) -> FileResponse:
        return _serve_image(dataset_index, event_id, "crops", name)

    @app.get("/api/events/{event_id}/crops_overlay/{name:path}")
    def get_crop_overlay(event_id: str, name: str) -> FileResponse:
        return _serve_image(dataset_index, event_id, "crops_overlay", name)

    @app.post("/api/events/{event_id}/label")
    def label_event(event_id: str, update: LabelUpdate) -> dict:
        record = _event_or_404(dataset_index, event_id)
        dog_kwargs: dict = {}
        if "dog" in update.model_fields_set:
            dog = update.dog.strip() if update.dog is not None else None
            dog = dog or None
            if dog is not None and dogs and dog not in dogs:
                raise HTTPException(status_code=422, detail="unknown dog")
            dog_kwargs["dog"] = dog
        try:
            return dataset_index.update_label(
                record,
                label=update.label,
                label_status=update.label_status,
                note=update.note,
                **dog_kwargs,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="label update failed") from exc

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
        cached = app.state.tune_detectors.get(name)
        if cached is not None:
            return cached, name
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
                    # find_spec said the dep exists but inference failed (missing
                    # model files, bad install, ...). Downgrade so the UI stops
                    # promising pose and we don't retry the heavy path every frame.
                    logger.warning("pose inference failed; disabling pose overlay")
                    app.state.tune_pose_resolved = (None, False)
                    pose_available = False
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
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": float(bbox.x1),
                    "y1": float(bbox.y1),
                    "x2": float(bbox.x2),
                    "y2": float(bbox.y2),
                }
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
        ``/api/tune/track_range`` contract + tests are unchanged.
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
        never leaks across requests; the lock is held for the whole pass because the
        online tracker's per-frame state must be updated atomically. Draining this
        reproduces ``_track_range_ultralytics_payload``. ``count`` is pre-clamped by the
        caller.
        """

        from detectivepotty.device import resolve_device
        from detectivepotty.web import tune as tune_mod

        # Allow-list + `.pt` guard (also enforced by the endpoint, kept here so the
        # generator is safe if ever driven directly).
        if model_name not in app.state.tune_models:
            raise ValueError(f"unknown model: {model_name}")
        if not model_name.endswith(".pt"):
            raise ValueError("Ultralytics tracking requires a .pt model")
        if not _ultralytics_tracking_available():
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
            with app.state.tune_infer_lock:
                while cursor < end:
                    chunk_count = min(decode_cap, end - cursor)
                    try:
                        frames, total, fps, _w, _h = tune_mod.read_frames(
                            file_path, cursor, chunk_count
                        )
                    except IndexError:
                        break  # past EOF: track what we have
                    chunk_out: list[dict] = []
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
                logger.warning("pose inference failed; disabling pose overlay")
                app.state.tune_pose_resolved = (None, False)
                pose_available = False
                pose_list = []
        return {"index": idx, "pose": pose_list, "pose_available": pose_available}

    def _pose_range_payload(
        file_path: Path,
        frames_in: list[tuple[int, list[list[float]]]],
    ) -> dict:
        """Batched pose over a run of frames — one ``estimate_batch`` GPU forward.

        Drives ``POST /api/tune/pose_range``. Each frame is decoded *outside*
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
                logger.warning("pose inference failed; disabling pose overlay")
                app.state.tune_pose_resolved = (None, False)
                pose_available = False
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
        return FileResponse(
            file_path,
            media_type=_VIDEO_MIME.get(file_path.suffix.lower(), "video/mp4"),
            headers={"Cache-Control": "no-store"},
        )

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

    @app.get("/api/tune/detect_range")
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

    @app.get("/api/tune/track_range")
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

        from detectivepotty.web import tune as tune_mod

        if tracker not in TUNE_TRACKERS:
            raise HTTPException(status_code=400, detail=f"unknown tracker: {tracker}")
        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)
        bounded = min(count, TUNE_TRACK_MAX_FRAMES)
        ultra_params = TuneUltralyticsTrackerParams(
            conf=ultra_conf,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
        )

        if tracker in _ULTRALYTICS_TRACKERS:
            if not model_name.endswith(".pt"):
                raise HTTPException(
                    status_code=400,
                    detail="Ultralytics tracking requires a .pt model",
                )
            if not _ultralytics_tracking_available():
                raise HTTPException(
                    status_code=400,
                    detail="Ultralytics tracking unavailable (install `lap`)",
                )
            try:
                return await run_in_threadpool(
                    _track_range_ultralytics_payload,
                    file_path,
                    start,
                    bounded,
                    model_name,
                    tracker=tracker,
                    sample_every=sample_every,
                    ultra_params=ultra_params,
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
                file_path,
                start,
                bounded,
                model_name,
                sample_every=sample_every,
                iou_threshold=iou_threshold,
                max_age_frames=max_age_frames,
                center_dist_gate=center_dist_gate,
            )
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail="frame not available") from exc

    @app.get("/api/tune/track_range_stream")
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

        from detectivepotty.web import tune as tune_mod

        if tracker not in TUNE_TRACKERS:
            raise HTTPException(status_code=400, detail=f"unknown tracker: {tracker}")
        try:
            file_path = tune_mod.resolve_tune_file(path, app.state.tune_roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid path") from exc
        model_name = _resolve_tune_model(model)
        bounded = min(count, TUNE_TRACK_MAX_FRAMES)
        ultra_params = TuneUltralyticsTrackerParams(
            conf=ultra_conf,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
        )

        if tracker in _ULTRALYTICS_TRACKERS:
            if not model_name.endswith(".pt"):
                raise HTTPException(
                    status_code=400,
                    detail="Ultralytics tracking requires a .pt model",
                )
            if not _ultralytics_tracking_available():
                raise HTTPException(
                    status_code=400,
                    detail="Ultralytics tracking unavailable (install `lap`)",
                )
            gen = _iter_track_range_ultralytics(
                file_path,
                start,
                bounded,
                model_name,
                tracker=tracker,
                sample_every=sample_every,
                ultra_params=ultra_params,
            )
        else:
            gen = _iter_track_range(
                file_path,
                start,
                bounded,
                model_name,
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

    @app.post("/api/tune/pose_range")
    async def tune_pose_range(req: TunePoseRangeRequest) -> dict:
        """Batched pose for a run of frames' buffered boxes — one GPU forward.

        The pose analogue of ``/api/tune/detect_range``: instead of one pose
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
            frames_in.append((frame.index, frame.boxes))
            crops += len(frame.boxes)
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

    @app.get("/api/label/clips")
    def label_clips() -> dict:
        """List harvested clips with their labeling progress (unlabeled first)."""

        from detectivepotty.web import labeling

        return {
            "clips": labeling.list_clips(app.state.harvest_root),
            "vocabulary": labeling.label_vocabulary(),
        }

    @app.get("/api/label/clips/{span_id}")
    def label_clip_detail(span_id: str) -> dict:
        """Geometry + detection tracks + existing labels for one harvested clip."""

        from detectivepotty.web import labeling

        try:
            clip_dir = labeling.clip_dir_for(app.state.harvest_root, span_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc
        return labeling.clip_detail(clip_dir, app.state.harvest_root)

    @app.put("/api/label/clips/{span_id}/labels")
    def label_clip_save(span_id: str, payload: dict = Body(...)) -> dict:
        """Validate + persist ``labels.json`` for one clip, return fresh detail."""

        from detectivepotty.web import labeling

        try:
            clip_dir = labeling.clip_dir_for(app.state.harvest_root, span_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc
        try:
            return labeling.save_clip_labels(clip_dir, payload, app.state.harvest_root)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/label/clips/{span_id}/video")
    def label_clip_video(span_id: str) -> FileResponse:
        """Stream a harvested ``clip.mp4`` (Range-seekable) for the labeler's video."""

        from detectivepotty.harvest import CLIP_NAME
        from detectivepotty.web import labeling

        try:
            clip_dir = labeling.clip_dir_for(app.state.harvest_root, span_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc
        clip_path = clip_dir / CLIP_NAME
        return FileResponse(
            clip_path,
            media_type=_VIDEO_MIME.get(clip_path.suffix.lower(), "video/mp4"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Response:
        """Serve the SPA shell for client-side routes (e.g. /tune, /live).

        Real ``/api/*`` routes and the ``/assets`` mount are registered earlier
        so they win; only unknown non-API paths fall through here so a browser
        refresh on a client route returns index.html instead of 404. API/asset
        paths that reach here are genuinely unknown -> 404 (never HTML).
        """

        if full_path in {"api", "assets"} or full_path.startswith(("api/", "assets/")):
            raise HTTPException(status_code=404, detail="not found")
        built = FRONTEND_DIST / "index.html"
        if built.is_file():
            return FileResponse(built, headers={"Cache-Control": "no-store"})
        return HTMLResponse(_BUILD_MISSING_HTML)

    return app


def create_app_from_env() -> FastAPI:
    """Build the app from ``$DETECTIVEPOTTY_CONFIG`` (default ``config.yaml``).

    uvicorn's ``--reload`` re-imports the app in a worker subprocess, so it needs
    an import string pointing at a zero-arg factory rather than a prebuilt app
    object. The config path is read from the environment because that subprocess
    never re-runs the CLI; it falls back to ``config.yaml`` relative to the
    process working directory.
    """

    config_path = os.environ.get(CONFIG_ENV_VAR, "config.yaml")
    return create_app(load_config(config_path))


def run_server(
    config: Config,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    reload: bool = False,
    config_path: str | Path | None = None,
) -> None:
    """Serve the review app.

    ``reload=True`` (dev only) re-launches via an import string + app factory so
    uvicorn can hot-reload on source changes; the prebuilt ``config`` is ignored
    in that mode because each reloaded worker rebuilds it from ``config_path``
    (passed through ``$DETECTIVEPOTTY_CONFIG``). Without reload it serves the
    already-built app object, which is the production path.
    """

    if reload:
        if config_path is not None:
            os.environ[CONFIG_ENV_VAR] = str(config_path)
        # Watch the whole package so edits to app.py or anything it imports
        # (config, dataset_index, events, ...) trigger a reload.
        reload_dir = str(Path(__file__).resolve().parents[1])
        uvicorn.run(
            "detectivepotty.web.app:create_app_from_env",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[reload_dir],
        )
        return
    uvicorn.run(create_app(config), host=host, port=port)


def _event_or_404(dataset_index: DatasetIndex, event_id: str):
    record = dataset_index.get_event(event_id)
    if record is None:
        raise HTTPException(status_code=404, detail="event not found")
    return record


def _serve_image(
    dataset_index: DatasetIndex,
    event_id: str,
    kind: str,
    name: str,
) -> FileResponse:
    record = _event_or_404(dataset_index, event_id)
    try:
        path = media_path(record, kind, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid media filename") from exc
    if path is None:
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(path, headers={"Cache-Control": MEDIA_CACHE_CONTROL})
