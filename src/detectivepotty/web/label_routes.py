"""Harvested clip labeling API routes."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse


def register_label_routes(
    app: FastAPI,
    *,
    video_mime: Mapping[str, str],
) -> None:
    """Register range-labeling routes on ``app``."""

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
            media_type=video_mime.get(clip_path.suffix.lower(), "video/mp4"),
            headers={"Cache-Control": "no-store"},
        )
