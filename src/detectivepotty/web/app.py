"""FastAPI app for local DetectivePotty event review."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

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


class LabelUpdate(BaseModel):
    label: Label
    label_status: LabelStatus
    note: str | None = Field(default=None, max_length=2000)
    dog: str | None = Field(default=None, max_length=200)


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


def create_app(config: Config) -> FastAPI:
    dataset_index = DatasetIndex(config.global_settings.dataset_dir)
    dogs = list(config.global_settings.dogs)
    app = FastAPI(title="DetectivePotty", version="0.1.0")
    app.state.dataset_index = dataset_index
    app.state.dogs = dogs
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
