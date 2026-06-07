"""FastAPI app for local DetectivePotty event review."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from detectivepotty.config import Config
from detectivepotty.events import Label, LabelStatus
from detectivepotty.web.dataset_index import DatasetIndex, fixed_media_path, media_path


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class LabelUpdate(BaseModel):
    label: Label
    label_status: LabelStatus
    note: str | None = Field(default=None, max_length=2000)
    dog: str | None = Field(default=None, max_length=200)


def create_app(config: Config) -> FastAPI:
    dataset_index = DatasetIndex(config.global_settings.dataset_dir)
    dogs = list(config.global_settings.dogs)
    app = FastAPI(title="DetectivePotty", version="0.1.0")
    app.state.dataset_index = dataset_index
    app.state.dogs = dogs

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def _no_store_api(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

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
        summaries = dataset_index.list_summaries(
            camera=camera,
            label_status=label_status,
            date=date,
        )
        total = len(summaries)
        unfiltered_total = len(dataset_index.scan())
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
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/events/{event_id}/media/protect")
    def get_protect_recording(event_id: str) -> FileResponse:
        record = _event_or_404(dataset_index, event_id)
        path = fixed_media_path(record, "protect_recording.mp4", missing_ok=True)
        if path is None:
            raise HTTPException(status_code=404, detail="protect recording not found")
        return FileResponse(path, media_type="video/mp4")

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


def run_server(
    config: Config,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
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
    return FileResponse(path)
