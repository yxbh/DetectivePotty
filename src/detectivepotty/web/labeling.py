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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from detectivepotty.harvest import CLIP_NAME, METADATA_NAME
from detectivepotty.harvest_unvr import CAMERAS_NAME
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


def _parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _parse_camera_id(source_id: str) -> str | None:
    """A UNVR ``source_id`` is ``<cameraId>@<UTCstamp>``; pull the camera id out."""

    if "@" in source_id:
        cid = source_id.split("@", 1)[0].strip()
        return cid or None
    return None


def load_camera_names(root: str | Path) -> dict[str, str]:
    """Read the ``cameras.json`` id→name sidecar at ``root`` (empty if absent)."""

    path = Path(root) / CAMERAS_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _resolve_camera_name(
    meta: dict[str, Any], camera_names: dict[str, str]
) -> str | None:
    """Friendly camera name: clip metadata first, then the id→name sidecar."""

    name = meta.get("camera_name")
    if name:
        return str(name)
    cid = _parse_camera_id(str(meta.get("source_id") or ""))
    if cid and cid in camera_names:
        return camera_names[cid]
    return None


def _clip_geom(
    meta: dict[str, Any],
) -> tuple[float, int, datetime | None, datetime | None]:
    """Return ``(fps, frame_count, span_start_abs, source_start_abs)``.

    ``span_start_abs`` is the wall-clock time of clip frame 0; ``source_start_abs``
    is the time of the parent source recording's frame 0 (what a detection's
    ``time_s`` is measured from). Either is derived from the other via ``start_s``
    when only one is present.
    """

    fps = float(meta.get("fps") or 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    start_s = float(meta.get("start_s") or 0.0)
    span_start = _parse_dt(meta.get("source_span_start_utc"))
    source_start = _parse_dt(meta.get("source_start_utc"))
    if span_start is None and source_start is not None:
        span_start = source_start + timedelta(seconds=start_s)
    if source_start is None and span_start is not None:
        source_start = span_start - timedelta(seconds=start_s)
    return fps, frame_count, span_start, source_start


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


def summarize_clip(
    clip_dir: Path, *, camera_names: dict[str, str] | None = None
) -> dict[str, Any]:
    """One clip's listing row: identity, geometry, and labeling progress."""

    meta = _read_metadata(clip_dir)
    if camera_names is None:
        camera_names = load_camera_names(clip_dir.parent)
    fps = float(meta.get("fps") or 0.0)
    frame_count = int(meta.get("frame_count") or 0)
    detections = meta.get("detections") or []
    source_id = str(meta.get("source_id") or clip_dir.name)
    labels = _labels_for(clip_dir)
    ranges = list(labels.ranges) if labels else []
    trainable = [r for r in ranges if r.is_trainable]
    behaviors = sorted({r.behavior.value for r in ranges})
    dogs = sorted({r.dog.value for r in ranges})
    span_start = str(
        meta.get("source_span_start_utc") or meta.get("source_start_utc") or ""
    )
    span_end = str(meta.get("source_span_end_utc") or "")

    return {
        "span_id": str(meta.get("span_id") or clip_dir.name),
        "clip_path": str((clip_dir / CLIP_NAME).resolve()),
        "source_id": source_id,
        "camera_id": _parse_camera_id(source_id),
        "camera_name": _resolve_camera_name(meta, camera_names),
        "date": _date_of(meta),
        "span_start_utc": span_start or None,
        "span_end_utc": span_end or None,
        "fps": fps,
        "frame_count": frame_count,
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "duration_s": round(frame_count / fps, 3) if fps > 0 else 0.0,
        "detect_conf": (
            float(meta["detect_conf"]) if meta.get("detect_conf") is not None else None
        ),
        "track_id": str(meta.get("track_id")) if meta.get("track_id") is not None else None,
        "n_detections": len(detections),
        "labeled": bool(trainable),
        "n_ranges": len(ranges),
        "n_trainable_ranges": len(trainable),
        "behaviors": behaviors,
        "dogs": dogs,
        # scene_* are filled in by list_clips once all rows are known.
        "scene_id": None,
        "scene_size": 1,
    }


def list_clips(root: str | Path) -> list[dict[str, Any]]:
    """All harvested clips under ``root``, newest first, unlabeled surfaced first.

    Sort key puts unlabeled clips ahead of labeled ones (the labeler's queue),
    then newest ``span_start_utc`` *timestamp* (not just the day), then span id
    for stability. Sorting on the parsed datetime — rather than a day bucket with
    a ``span_id`` string fallback — is what keeps same-day clips chronological
    (a string fallback ordered ``"11…"`` before ``"0903…"``). Clips that overlap
    in absolute time on the same camera are tagged with a shared ``scene_id``
    (multi-dog scenes) so the UI can group them.
    """

    camera_names = load_camera_names(root)
    rows = [
        summarize_clip(clip_dir, camera_names=camera_names)
        for clip_dir in discover_clip_dirs(root)
    ]
    _assign_scenes(rows)
    rows.sort(key=lambda r: (r["labeled"], _neg_ts_key(r), r["span_id"]))
    return rows


def _assign_scenes(rows: list[dict[str, Any]]) -> None:
    """Tag rows that overlap in time on the same camera with a shared scene id.

    A "scene" is a maximal run of clips on one camera whose
    ``[span_start_utc, span_end_utc]`` windows chain by overlap — i.e. the same
    real-world moment captured as one clip per tracked dog. Mutates ``rows`` in
    place, setting ``scene_id`` (``None`` when alone or unparseable) and
    ``scene_size``.
    """

    by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        start = _parse_dt(row.get("span_start_utc"))
        end = _parse_dt(row.get("span_end_utc")) or start
        cam = row.get("camera_id") or row.get("source_id")
        if start is None or cam is None:
            continue
        row["_win"] = (start, end or start)
        by_camera.setdefault(str(cam), []).append(row)

    for cam, members in by_camera.items():
        members.sort(key=lambda r: r["_win"][0])
        scene_idx = -1
        cluster_end: datetime | None = None
        cluster: list[dict[str, Any]] = []

        def _flush(cluster: list[dict[str, Any]], cam: str, idx: int) -> None:
            if len(cluster) > 1:
                sid = f"{cam}#{idx}"
                for m in cluster:
                    m["scene_id"] = sid
                    m["scene_size"] = len(cluster)

        for member in members:
            start, end = member["_win"]
            if cluster_end is not None and start <= cluster_end:
                cluster.append(member)
                cluster_end = max(cluster_end, end)
            else:
                _flush(cluster, cam, scene_idx)
                scene_idx += 1
                cluster = [member]
                cluster_end = end
        _flush(cluster, cam, scene_idx)

    for row in rows:
        row.pop("_win", None)


def _neg_ts_key(row: dict[str, Any]) -> float:
    # Newest first: negative epoch seconds, so a plain ascending sort yields
    # descending time. Sorts on the full ``span_start_utc`` timestamp (falling
    # back to the day ``date``), so same-day clips stay chronological instead of
    # tie-breaking on the ``span_id`` string. Unparseable clips sort last (+inf).
    dt = _parse_dt(row.get("span_start_utc")) or _parse_dt(row.get("date"))
    return -dt.timestamp() if dt is not None else float("inf")


def clip_detail(clip_dir: Path, root: str | Path | None = None) -> dict[str, Any]:
    """Full payload for the labeling screen: geometry, tracks, existing labels.

    ``tracks`` groups this clip's own recorded detection boxes by ``track_id``
    (sorted by clip frame), giving the UI a per-frame box to draw and bind a range
    to. ``present_tracks`` additionally folds in **sibling tracks** — other dogs
    detected in overlapping clips on the same camera, with their boxes mapped into
    *this* clip's frame timeline — so a multi-dog scene can be labeled by selecting
    the right box (and jumping to a sibling's own clip). The labels block is the
    round-tripped ``labels.json`` (empty ranges if none yet).
    """

    meta = _read_metadata(clip_dir)
    if root is None:
        root = clip_dir.parent
    camera_names = load_camera_names(root)
    summary = summarize_clip(clip_dir, camera_names=camera_names)
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

    present_tracks = _present_tracks(clip_dir, root, meta, tracks, camera_names)
    labels = _labels_for(clip_dir) or ClipLabels()
    return {
        **summary,
        "tracks": tracks,
        "present_tracks": present_tracks,
        "n_tracks": len(present_tracks),
        "labels": labels.to_dict(),
    }


def _present_tracks(
    clip_dir: Path,
    root: str | Path,
    meta: dict[str, Any],
    own_tracks: dict[str, list[dict[str, Any]]],
    camera_names: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """All tracked dogs visible in this clip's window, keyed ``<span_id>:<track>``.

    Includes this clip's own track(s) (``is_self=True``) plus sibling clips on the
    same camera whose detections fall inside this clip's frame range, remapped to
    this clip's frame indices. Pure read over recorded metadata — no decode/detect.
    """

    span_id = str(meta.get("span_id") or clip_dir.name)
    cam_key = _parse_camera_id(str(meta.get("source_id") or "")) or str(
        meta.get("source_id") or ""
    )
    fps, frame_count, span_start, _ = _clip_geom(meta)

    result: dict[str, dict[str, Any]] = {}
    for track_id, boxes in own_tracks.items():
        result[f"{span_id}:{track_id}"] = {
            "span_id": span_id,
            "track_id": track_id,
            "is_self": True,
            "camera_name": _resolve_camera_name(meta, camera_names),
            "boxes": boxes,
        }

    if not cam_key or span_start is None or fps <= 0 or frame_count <= 0:
        return result

    last_frame = frame_count - 1
    for sib_dir in discover_clip_dirs(root):
        if sib_dir.resolve() == clip_dir.resolve():
            continue
        try:
            sib = _read_metadata(sib_dir)
        except (OSError, ValueError):
            continue
        sib_source_id = str(sib.get("source_id") or "")
        if (_parse_camera_id(sib_source_id) or sib_source_id) != cam_key:
            continue
        _, _, _, sib_source_start = _clip_geom(sib)
        if sib_source_start is None:
            continue
        sib_span_id = str(sib.get("span_id") or sib_dir.name)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for det in sib.get("detections", []):
            abs_t = sib_source_start + timedelta(seconds=float(det.get("time_s", 0.0)))
            cf = round((abs_t - span_start).total_seconds() * fps)
            if cf < 0 or cf > last_frame:
                continue
            box = det.get("bbox") or {}
            grouped.setdefault(str(det.get("track_id")), []).append(
                {
                    "clip_frame_idx": cf,
                    "bbox": {
                        "x1": float(box.get("x1", 0.0)),
                        "y1": float(box.get("y1", 0.0)),
                        "x2": float(box.get("x2", 0.0)),
                        "y2": float(box.get("y2", 0.0)),
                    },
                    "confidence": float(det.get("confidence", 0.0)),
                }
            )
        for track_id, boxes in grouped.items():
            boxes.sort(key=lambda item: item["clip_frame_idx"])
            result[f"{sib_span_id}:{track_id}"] = {
                "span_id": sib_span_id,
                "track_id": track_id,
                "is_self": False,
                "camera_name": _resolve_camera_name(sib, camera_names),
                "boxes": boxes,
            }
    return result


def save_clip_labels(
    clip_dir: Path, payload: dict[str, Any], root: str | Path | None = None
) -> dict[str, Any]:
    """Validate + atomically persist ``labels.json``; return the new clip detail.

    Validation (enum membership, inverted ranges, non-negative frames) is owned
    by :class:`~detectivepotty.labels.ClipLabels` / ``LabelRange`` and surfaces as
    ``ValueError`` for the API to map to 400.
    """

    labels = ClipLabels.from_dict(payload)
    labels.clip = CLIP_NAME
    save_labels(labels, clip_dir)
    return clip_detail(clip_dir, root)


def label_vocabulary() -> dict[str, list[str]]:
    """The fixed enum choices the UI renders (behaviors + dog identities)."""

    return {
        "behaviors": [b.value for b in Behavior],
        "dogs": [d.value for d in Dog],
    }
