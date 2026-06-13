"""Review-event API routes and SSE streaming."""

from __future__ import annotations

from typing import Annotated
import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from detectivepotty.events import LabelStatus
from detectivepotty.web.dataset_index import DatasetIndex, fixed_media_path, media_path
from detectivepotty.web.schemas import LabelUpdate


logger = logging.getLogger(__name__)

STREAM_POLL_SECONDS = 2.0
MEDIA_CACHE_CONTROL = "private, max-age=3600"


async def _event_stream(
    dataset_index,
    is_disconnected,
    *,
    sleep=asyncio.sleep,
    poll_seconds=None,
):
    """Async generator yielding SSE frames for newly-recorded events."""

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
        for record in reversed(fresh):
            try:
                summary = await run_in_threadpool(dataset_index.summary, record)
            except Exception:  # pragma: no cover - defensive
                continue
            known.add(record.event_id)
            payload = json.dumps(summary)
            yield f"id: {record.event_id}\nevent: new\ndata: {payload}\n\n"
        yield ": ping\n\n"


def register_event_routes(
    app: FastAPI,
    *,
    dataset_index: DatasetIndex,
    dogs: list[str],
) -> None:
    """Register dataset event review routes on ``app``."""

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
        """Server-Sent Events feed of newly-recorded potty events."""

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
