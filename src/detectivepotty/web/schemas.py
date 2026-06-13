"""Request schemas shared by the web API routers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from detectivepotty.events import Label, LabelStatus


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
    """Body for ``POST /api/tune/pose-range`` — the batched pose pass.

    Carries the buffered boxes for a run of frames so pose runs as **one batched
    GPU forward across the whole window** instead of one request per frame (the
    SuperAnimal backend measured ~9-14x faster batched than the batch-1 floor).
    Bounded (frame count + per-frame boxes) so a hostile/buggy client can't
    schedule unbounded work; the server further caps total crops.
    """

    path: str = Field(max_length=4096)
    frames: list[TunePoseRangeFrame] = Field(default_factory=list, max_length=64)
