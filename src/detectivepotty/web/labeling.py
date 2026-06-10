"""Backend for the range-labeling UX over harvested clips.

Discovers harvested clip dirs (``clip.mp4`` + ``metadata.json``), surfaces each
clip's labeling state (existing ``labels.json`` plus the span's detection track,
so the UI can show the bound dog box per frame), and persists ``labels.json``.

Everything is pure functions over a *harvest root* directory so the FastAPI layer
and the offline tests share one implementation. No model inference happens here —
the scrub/decode/detect surface is reused from ``/api/tune`` (the harvest root is
added to the tuner's allowed roots), and the boxes shown for binding come from the
harvest ``metadata.json`` the :mod:`detectivepotty.harvest` writer already
recorded. The exporter binds a labeled range to its dog by ``track_id``, so a
harvested clip's single span track is the box the labeler confirms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detectivepotty.harvest import CLIP_NAME, METADATA_NAME
from detectivepotty.labels import (
    LABELS_NAME,
    Behavior,
    ClipLabels,
    Dog,
    load_labels,
    save_labels,
)

UNKNOWN_DATE = "unknown-date"


def _read_metadata(clip_dir: Path) -> dict[str, Any]:
    with (clip_dir / METADATA_NAME).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clip_dir_for(root: str | Path, span_id: str) -> Path:
    """Resolve ``span_id`` to a harvested clip dir under ``root``, fail closed.

    ``span_id`` must be a single path component (the harvest writer names dirs
    that way); anything with a separator or ``..`` is rejected before touching
    the filesystem so the endpoint can't be turned into a traversal read. The
    resolved dir must still sit inside ``root`` and carry both ``clip.mp4`` and
    ``metadata.json`` to count as a harvested clip.
    """

    root_path = Path(root).resolve()
    if not span_id or span_id in {".", ".."} or "/" in span_id or "\\" in span_id:
        raise ValueError("invalid span id")
    candidate = (root_path / span_id).resolve()
    if candidate != root_path and not candidate.is_relative_to(root_path):
        raise ValueError("span id escapes harvest root")
    if not (candidate / METADATA_NAME).is_file() or not (candidate / CLIP_NAME).is_file():
        raise ValueError("not a harvested clip")
    return candidate


def discover_clip_dirs(root: str | Path) -> list[Path]:
    """Return harvested clip dirs (``metadata.json`` + ``clip.mp4``) under ``root``."""

    root_path = Path(root)
    if not root_path.is_dir():
        return []
    found: set[Path] = set()
    for meta_path in root_path.rglob(METADATA_NAME):
        clip_dir = meta_path.parent
        if (clip_dir / CLIP_NAME).is_file():
            found.add(clip_dir)
    return sorted(found)


def _date_of(meta: dict[str, Any]) -> str:
    start = str(meta.get("source_span_start_utc") or meta.get("source_start_utc") or "")
    return start[:10] if len(start) >= 10 else UNKNOWN_DATE


def _labels_for(clip_dir: Path) -> ClipLabels | None:
    if not (clip_dir / LABELS_NAME).is_file():
        return None
    try:
        return load_labels(clip_dir)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def summarize_clip(clip_dir: Path) -> dict[str, Any]:
    """One clip's listing row: identity, geometry, and labeling progress."""

    meta = _read_metadata(clip_dir)
    fps = float(meta.get("fps") or 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    detections = meta.get("detections") or []
    labels = _labels_for(clip_dir)
    ranges = list(labels.ranges) if labels else []
    trainable = [r for r in ranges if r.is_trainable]
    behaviors = sorted({r.behavior.value for r in ranges})
    dogs = sorted({r.dog.value for r in ranges})

    return {
        "span_id": str(meta.get("span_id") or clip_dir.name),
        "clip_path": str((clip_dir / CLIP_NAME).resolve()),
        "source_id": str(meta.get("source_id") or clip_dir.name),
        "date": _date_of(meta),
        "fps": fps,
        "frame_count": frame_count,
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "duration_s": round(frame_count / fps, 3) if fps > 0 else 0.0,
        "track_id": str(meta.get("track_id")) if meta.get("track_id") is not None else None,
        "n_detections": len(detections),
        "labeled": bool(trainable),
        "n_ranges": len(ranges),
        "n_trainable_ranges": len(trainable),
        "behaviors": behaviors,
        "dogs": dogs,
    }


def list_clips(root: str | Path) -> list[dict[str, Any]]:
    """All harvested clips under ``root``, newest day first, unlabeled surfaced first.

    Sort key puts unlabeled clips ahead of labeled ones (the labeler's queue),
    then newest date, then span id for stability.
    """

    rows = [summarize_clip(clip_dir) for clip_dir in discover_clip_dirs(root)]
    rows.sort(key=lambda r: (r["labeled"], _neg_date_key(r["date"]), r["span_id"]))
    return rows


def _neg_date_key(date: str) -> str:
    # Newest first without parsing: invert each digit of an ISO date so plain
    # ascending string sort yields descending dates. Non-dates sort last.
    if len(date) == 10 and date[4] == "-" and date[7] == "-":
        return "".join("9" if c == "-" else str(9 - int(c)) for c in date if c != "-")
    return "~~~~~~~~"


def clip_detail(clip_dir: Path) -> dict[str, Any]:
    """Full payload for the labeling screen: geometry, tracks, existing labels.

    ``tracks`` groups the recorded detection boxes by ``track_id`` (sorted by
    clip frame), giving the UI a per-frame box to draw and bind a range to. The
    labels block is the round-tripped ``labels.json`` (empty ranges if none yet).
    """

    meta = _read_metadata(clip_dir)
    summary = summarize_clip(clip_dir)
    tracks: dict[str, list[dict[str, Any]]] = {}
    for det in meta.get("detections", []):
        track_id = str(det.get("track_id"))
        box = det.get("bbox") or {}
        tracks.setdefault(track_id, []).append(
            {
                "clip_frame_idx": int(det.get("clip_frame_idx", 0)),
                "bbox": {
                    "x1": float(box.get("x1", 0.0)),
                    "y1": float(box.get("y1", 0.0)),
                    "x2": float(box.get("x2", 0.0)),
                    "y2": float(box.get("y2", 0.0)),
                },
                "confidence": float(det.get("confidence", 0.0)),
            }
        )
    for entries in tracks.values():
        entries.sort(key=lambda item: item["clip_frame_idx"])

    labels = _labels_for(clip_dir) or ClipLabels()
    return {
        **summary,
        "tracks": tracks,
        "labels": labels.to_dict(),
    }


def save_clip_labels(clip_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate + atomically persist ``labels.json``; return the new clip detail.

    Validation (enum membership, inverted ranges, non-negative frames) is owned
    by :class:`~detectivepotty.labels.ClipLabels` / ``LabelRange`` and surfaces as
    ``ValueError`` for the API to map to 400.
    """

    labels = ClipLabels.from_dict(payload)
    labels.clip = CLIP_NAME
    save_labels(labels, clip_dir)
    return clip_detail(clip_dir)


def label_vocabulary() -> dict[str, list[str]]:
    """The fixed enum choices the UI renders (behaviors + dog identities)."""

    return {
        "behaviors": [b.value for b in Behavior],
        "dogs": [d.value for d in Dog],
    }
