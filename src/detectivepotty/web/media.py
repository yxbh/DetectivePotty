"""Shared media response helpers for local web routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.responses import FileResponse

VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}

NO_STORE = "no-store"
PRIVATE_MEDIA_CACHE = "private, max-age=3600"


def video_media_type(path: str | Path) -> str:
    return VIDEO_MIME.get(Path(path).suffix.lower(), "video/mp4")


def no_store_file_response(path: str | Path, *, media_type: str | None = None) -> FileResponse:
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": NO_STORE},
    )


def no_store_video_response(path: str | Path) -> FileResponse:
    return no_store_file_response(path, media_type=video_media_type(path))


def cached_media_response(path: str | Path, *, media_type: str | None = None) -> FileResponse:
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": PRIVATE_MEDIA_CACHE},
    )


def cached_video_response(path: str | Path) -> FileResponse:
    return cached_media_response(path, media_type=video_media_type(path))
