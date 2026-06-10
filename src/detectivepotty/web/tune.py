"""Backend helpers for the in-browser detection tuner (``/api/tune/*``).

The tuner lets a human scrub a local clip, watch live YOLO boxes, drag a
confidence threshold, and (optionally) overlay pose keypoints — all in the
review web app instead of the native OpenCV window. This module holds the
GUI-free, unit-testable pieces:

* a **traversal-guarded file browser** restricted to ``data/`` + the dirs of any
  configured file cameras + the dataset dir, so the browser can only ever reach
  clips the operator already pointed the app at;
* **frame decoding** (seek + read a single frame) and JPEG/data-URL encoding;
* **payload shaping** for detections and pose keypoints;
* a **pose-estimator resolver** that degrades gracefully to "unavailable" when
  the optional ``pose`` extra (DeepLabCut) is not installed.

Inference itself (YOLO / pose model construction) lives behind injectable seams
in :mod:`detectivepotty.web.app` so the offline test suite never loads a real
model. Everything here works on an already-decoded frame plus a detector/pose
object handed in by the caller.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Sequence
import importlib.util
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from detectivepotty.config import Config
    from detectivepotty.events import Detection
    from detectivepotty.geometry import BBox
    from detectivepotty.pose.base import PoseEstimator
    from detectivepotty.pose.keypoints import PoseKeypoints

# Container/extension allow-list for the file browser. Lower-cased suffix match.
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
)

# JPEG quality for the streamed preview frames. 85 keeps the data URL small
# without visibly degrading the boxes/keypoints the operator is judging.
_JPEG_QUALITY = 85


def collect_tune_roots(config: Config) -> list[Path]:
    """Return the directories the tuner is allowed to browse, resolved + deduped.

    The browser is intentionally restricted to: the parent dir of every
    configured ``kind: file`` camera (so the clips you already wired in are one
    click away), the conventional ``data/`` drop folder, and the dataset dir
    (recorded event clips). Anything outside these roots is rejected by
    :func:`resolve_tune_dir` / :func:`resolve_tune_file`, which is what keeps the
    endpoint from turning into an arbitrary-file read.
    """

    roots: list[Path] = []

    def add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - defensive (e.g. broken symlink root)
            return
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)

    for camera in config.cameras:
        if camera.input.kind == "file" and camera.input.path is not None:
            add(camera.input.path.parent)
    add(Path("data"))
    add(config.global_settings.dataset_dir)
    return roots


def _is_within_roots(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def resolve_tune_dir(path_str: str, roots: Sequence[Path]) -> Path:
    """Resolve a browse path to a directory inside ``roots`` or raise ``ValueError``.

    ``.resolve()`` collapses ``..`` and follows symlinks before the containment
    check, so a path that escapes a root (via traversal or a symlink pointing
    out) fails closed.
    """

    candidate = Path(path_str).resolve()
    if not _is_within_roots(candidate, roots):
        raise ValueError("path is outside the allowed roots")
    if not candidate.is_dir():
        raise ValueError("path is not a directory")
    return candidate


def resolve_tune_file(path_str: str, roots: Sequence[Path]) -> Path:
    """Resolve a clip path to a video file inside ``roots`` or raise ``ValueError``."""

    candidate = Path(path_str).resolve()
    if not _is_within_roots(candidate, roots):
        raise ValueError("path is outside the allowed roots")
    if not candidate.is_file():
        raise ValueError("path is not a file")
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("path is not a supported video file")
    return candidate


def _parent_for(path: Path, roots: Sequence[Path]) -> str | None:
    """Return the "up" target for ``path``: ``""`` = root list, else a dir str.

    A root's parent is the synthetic top-level root list (``""``). Deeper dirs
    return their parent path. We never expose a parent above a root.
    """

    if any(path == root for root in roots):
        return ""
    parent = path.parent
    if _is_within_roots(parent, roots):
        return str(parent)
    return ""


def list_tune_dir(path_str: str, roots: Sequence[Path]) -> dict[str, Any]:
    """List a browse location for the file browser.

    ``path_str == ""`` returns the synthetic top level (one entry per root).
    Otherwise it lists the sub-directories and video files of the resolved
    directory (dirs first, then videos, each sorted case-insensitively).
    """

    if not path_str:
        entries = [
            {"name": _root_label(root), "kind": "dir", "path": str(root)}
            for root in roots
        ]
        return {"path": "", "parent": None, "entries": entries}

    directory = resolve_tune_dir(path_str, roots)
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        children = list(directory.iterdir())
    except OSError as exc:  # pragma: no cover - permissions/race
        raise ValueError("directory is not readable") from exc

    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                dirs.append({"name": child.name, "kind": "dir", "path": str(child)})
            elif child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(
                    {
                        "name": child.name,
                        "kind": "video",
                        "path": str(child),
                        "size": child.stat().st_size,
                    }
                )
        except OSError:  # pragma: no cover - vanished mid-scan
            continue

    dirs.sort(key=lambda item: item["name"].lower())
    files.sort(key=lambda item: item["name"].lower())
    return {
        "path": str(directory),
        "parent": _parent_for(directory, roots),
        "entries": dirs + files,
    }


def _root_label(root: Path) -> str:
    """A friendly label for a root entry (its name, or the full path if nameless)."""

    return root.name or str(root)


def collect_tune_models(
    config: Config, models_dir: Path = Path("models")
) -> list[str]:
    """Return the YOLO weights the model picker may select, as detector strings.

    Discovers ``*.pt`` files and ``*.mlpackage`` CoreML bundles under
    ``models_dir`` (the conventional weights folder) plus any ``*.mlpackage``
    under ``models_dir/coreml`` (the curated, committable export location), and
    always includes the configured ``global.model_name`` so the active model is
    selectable even if it lives elsewhere or is a bare Ultralytics name. Each
    returned string is usable directly as ``DogDetector(model_name=...)``; the
    client labels options by basename.

    Note ``*.mlpackage`` is a **directory** bundle, not a file, so it is matched
    with ``is_dir()`` while ``*.pt`` weights are matched with ``is_file()``.

    The list doubles as an **allow-list**: ``/api/tune/detect`` only builds a
    detector for a model in this set, so an arbitrary ``model`` query can't be
    turned into a download or filesystem read.
    """

    seen: set[str] = set()
    models: list[str] = []

    def add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            models.append(value)

    def discover(directory: Path, pattern: str, want_dir: bool) -> None:
        try:
            matches = sorted(directory.glob(pattern), key=lambda p: p.name.lower())
        except OSError:  # pragma: no cover - defensive (e.g. unreadable dir)
            return
        for match in matches:
            if match.is_dir() if want_dir else match.is_file():
                add(str(match))

    discover(models_dir, "*.pt", want_dir=False)
    discover(models_dir, "*.mlpackage", want_dir=True)
    discover(models_dir / "coreml", "*.mlpackage", want_dir=True)

    add(config.global_settings.model_name)
    return models


def read_meta(path: Path) -> tuple[int, float, int, int, float]:
    """Return ``(total_frames, fps, width, height, duration_s)`` for ``path``.

    Reads container properties without decoding a frame, so the client can map
    ``video.currentTime`` to a frame index cheaply (no YOLO). ``total_frames`` is
    ``0`` when the container does not report a reliable count; ``duration_s`` is
    ``0.0`` when it cannot be derived. Routes through the persistent reader cache
    so selecting a clip also warms its decoder for the subsequent frame buffer.
    """

    return get_clip_reader(path).meta()


def read_frame(
    path: Path, index: int
) -> tuple[np.ndarray, int, int, float, int, int]:
    """Decode a single frame from ``path`` at ``index``.

    Returns ``(frame_bgr, idx_used, total_frames, fps, width, height)``.
    ``total_frames`` is ``0`` when the container does not report a reliable
    count. Backed by a process-wide :class:`ClipFrameReader` cache that keeps the
    ``cv2.VideoCapture`` open and reads sequentially when possible — the tuner's
    background filler walks frames forward, and a persistent sequential read is
    ~20x cheaper than reopening + keyframe-seeking per request. Raises
    ``IndexError`` if the requested frame cannot be read (e.g. past the end).
    """

    # If the reader we hold is retired out from under us by a concurrent
    # eviction, fetch a fresh one and retry rather than touching a closed
    # capture. Bounded so a pathological churn can't spin forever.
    last_exc: _ReaderRetired | None = None
    for _ in range(3):
        reader = get_clip_reader(path)
        try:
            return reader.read(index)
        except _ReaderRetired as exc:  # pragma: no cover - needs eviction race
            last_exc = exc
    raise IndexError(f"no frame at index {index} (reader churn)") from last_exc


def read_frames(
    path: Path, start: int, count: int
) -> tuple[list[tuple[int, np.ndarray]], int, float, int, int]:
    """Decode a contiguous run of frames from ``path`` (the batched read path).

    Returns ``(frames, total_frames, fps, width, height)`` where ``frames`` is a
    list of ``(idx, frame_bgr)``. Backed by the same persistent
    :class:`ClipFrameReader` cache as :func:`read_frame`; sequential decoding of
    the run is far cheaper than ``count`` independent seeks, which is what makes
    batched detection over a frame window worthwhile. Raises ``IndexError`` if the
    starting frame cannot be read.
    """

    last_exc: _ReaderRetired | None = None
    for _ in range(3):
        reader = get_clip_reader(path)
        try:
            return reader.read_range(start, count)
        except _ReaderRetired as exc:  # pragma: no cover - needs eviction race
            last_exc = exc
    raise IndexError(f"no frame at index {start} (reader churn)") from last_exc


# --- persistent per-clip decoder cache -----------------------------------
#
# Opening a fresh ``cv2.VideoCapture`` and seeking by frame index re-decodes
# from the preceding keyframe on every call (~65 ms on a 2688x1512 clip),
# whereas a persistent capture read sequentially costs ~3 ms. The tuner's
# buffer fills mostly forward, so we keep one open capture per clip and only
# fall back to a hard seek for backward / large-forward jumps.

# A forward jump of up to this many frames is served by grabbing intervening
# frames (~3 ms each) rather than a hard seek (~42 ms ≈ 16 grabs); chosen a bit
# below break-even from the measured numbers above.
FORWARD_GRAB_MAX = 12
# Bound on simultaneously open captures (file handles). The tuner views one clip
# at a time; a few keeps recently-browsed clips warm without leaking handles.
MAX_OPEN_READERS = 4


class _ReaderRetired(Exception):
    """Raised when a reader was evicted/closed while a caller still held it."""


class ClipFrameReader:
    """A persistent ``cv2.VideoCapture`` with a sequential-read fast path.

    Thread-safe: every capture access is serialized by ``self._lock`` (a single
    ``VideoCapture`` is not safe for concurrent use). ``_next_pos`` tracks the
    0-based index the next ``read()`` would return; it is set to ``None``
    ("unknown") after any decode/seek failure so the following read forces a hard
    seek instead of silently returning the wrong frame.
    """

    def __init__(self, path: Path, mtime_ns: int) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"failed to open video file: {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap = capture
        self.path = path
        self.mtime_ns = mtime_ns
        self._total = total if total > 0 else 0
        self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_pos: int | None = 0
        self._retired = False
        self._lock = threading.Lock()

    def meta(self) -> tuple[int, float, int, int, float]:
        duration = (self._total / self._fps) if (self._total > 0 and self._fps > 0) else 0.0
        return self._total, self._fps, self._width, self._height, duration

    @property
    def retired(self) -> bool:
        return self._retired

    def read(self, index: int) -> tuple[np.ndarray, int, int, float, int, int]:
        with self._lock:
            if self._retired:
                raise _ReaderRetired
            idx = index if index >= 0 else 0
            if self._total > 0:
                idx = min(idx, self._total - 1)
            ok, frame = self._decode_at(idx)
            if not ok or frame is None:
                # Position is no longer trustworthy after a failed grab/read/seek.
                self._next_pos = None
                raise IndexError(f"no frame at index {index}")
            self._next_pos = idx + 1
            height, width = frame.shape[:2]
            return frame, idx, self._total, self._fps, width, height

    def read_range(
        self, start: int, count: int
    ) -> tuple[list[tuple[int, np.ndarray]], int, float, int, int]:
        """Decode up to ``count`` contiguous frames starting at ``start``.

        Returns ``(frames, total, fps, width, height)`` where ``frames`` is a list
        of ``(idx, frame_bgr)`` in increasing index order. The run is read
        sequentially (each decode advances ``_next_pos`` so the next is a cheap
        sequential read) and stops short at EOF. Raises ``IndexError`` if not even
        the first frame can be decoded.
        """

        with self._lock:
            if self._retired:
                raise _ReaderRetired
            idx = start if start >= 0 else 0
            if self._total > 0:
                idx = min(idx, self._total - 1)
            frames: list[tuple[int, np.ndarray]] = []
            for _ in range(max(1, count)):
                if self._total > 0 and idx > self._total - 1:
                    break
                ok, frame = self._decode_at(idx)
                if not ok or frame is None:
                    # A failed decode mid-run leaves the position untrustworthy;
                    # stop here and let the caller use what we already have.
                    self._next_pos = None
                    break
                self._next_pos = idx + 1
                frames.append((idx, frame))
                idx += 1
            if not frames:
                raise IndexError(f"no frame at index {start}")
            return frames, self._total, self._fps, self._width, self._height

    def _decode_at(self, idx: int) -> tuple[bool, np.ndarray | None]:
        """Advance the capture to ``idx`` and decode it. Caller holds the lock."""

        pos = self._next_pos
        if pos is not None and pos == idx:
            return self._cap.read()
        if pos is not None and 0 <= (idx - pos) <= FORWARD_GRAB_MAX:
            for _ in range(idx - pos):
                if not self._cap.grab():
                    return False, None
            return self._cap.read()
        # Backward, large forward jump, or unknown position -> hard seek.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        return self._cap.read()

    def close(self) -> None:
        """Retire the reader and release its capture (waits for any in-flight read)."""

        with self._lock:
            self._retired = True
            if self._cap is not None:
                self._cap.release()
                self._cap = None  # type: ignore[assignment]


_READER_CACHE: OrderedDict[str, ClipFrameReader] = OrderedDict()
_READER_CACHE_LOCK = threading.Lock()


def get_clip_reader(path: Path) -> ClipFrameReader:
    """Return a cached open reader for ``path``, creating/evicting as needed.

    Keyed by path string; an entry is invalidated when the file's mtime changes
    (the clip was rewritten). Evicted/stale readers are ``close()``d *after* the
    cache lock is released so a slow close can never block unrelated lookups.
    """

    key = str(path)
    mtime_ns = os.stat(path).st_mtime_ns  # FileNotFoundError if the clip is gone
    victims: list[ClipFrameReader] = []
    try:
        with _READER_CACHE_LOCK:
            existing = _READER_CACHE.get(key)
            if existing is not None and existing.mtime_ns == mtime_ns and not existing.retired:
                _READER_CACHE.move_to_end(key)
                return existing
            if existing is not None:
                victims.append(_READER_CACHE.pop(key))
            reader = ClipFrameReader(path, mtime_ns)
            _READER_CACHE[key] = reader
            while len(_READER_CACHE) > MAX_OPEN_READERS:
                _, evicted = _READER_CACHE.popitem(last=False)
                victims.append(evicted)
            return reader
    finally:
        for victim in victims:
            victim.close()


def clear_clip_reader_cache() -> None:
    """Release and drop all cached readers (test hygiene / explicit reset)."""

    with _READER_CACHE_LOCK:
        readers = list(_READER_CACHE.values())
        _READER_CACHE.clear()
    for reader in readers:
        reader.close()



def encode_jpeg_dataurl(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame as a ``data:image/jpeg;base64,...`` URL for the ``<img>``."""

    ok, buffer = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not ok:  # pragma: no cover - imencode failure is environment-level
        raise RuntimeError("failed to JPEG-encode frame")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def detections_payload(detections: Sequence[Detection]) -> list[dict[str, Any]]:
    """Shape detector output for the client (original-resolution pixel boxes)."""

    return [
        {
            "x1": float(det.bbox.x1),
            "y1": float(det.bbox.y1),
            "x2": float(det.bbox.x2),
            "y2": float(det.bbox.y2),
            "confidence": float(det.confidence),
            "class_name": det.class_name,
        }
        for det in detections
    ]


def _pose_entry(bbox: BBox, keypoints: PoseKeypoints) -> dict[str, Any]:
    """Shape one (bbox, keypoints) pair into the overlay payload.

    Coordinates are original-resolution pixels (the same space as the boxes), so
    the client draws boxes and keypoints in one coordinate frame.
    """

    return {
        "bbox": [float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)],
        "keypoints": [
            {
                "name": name,
                "x": float(point.x),
                "y": float(point.y),
                "confidence": float(point.confidence),
            }
            for name, point in keypoints.points.items()
        ],
    }


def pose_payload(
    estimator: PoseEstimator,
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Estimate pose for each detection and shape keypoints for the overlay.

    One entry per detection that yields keypoints.
    """

    out: list[dict[str, Any]] = []
    for det in detections:
        keypoints = estimator.estimate(
            frame_bgr,
            det.bbox,
            frame_idx=frame_idx,
            source_id="tune",
        )
        if keypoints is None:
            continue
        out.append(_pose_entry(det.bbox, keypoints))
    return out


def _clamp_valid_bboxes(
    frame_bgr: np.ndarray, boxes: Sequence[Sequence[float]]
) -> list[BBox]:
    """Clamp client ``[x1, y1, x2, y2]`` boxes to the frame, in input order.

    Malformed (not 4 numbers) and degenerate (zero/negative-area after clamping)
    boxes are dropped. Shared by the single-frame and multi-frame pose payloads so
    both validate identically.
    """

    from detectivepotty.geometry import BBox

    height, width = frame_bgr.shape[:2]
    out: list[BBox] = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1 = min(max(float(box[0]), 0.0), float(width))
        y1 = min(max(float(box[1]), 0.0), float(height))
        x2 = min(max(float(box[2]), 0.0), float(width))
        y2 = min(max(float(box[3]), 0.0), float(height))
        bbox = BBox(x1, y1, x2, y2)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        out.append(bbox)
    return out


def pose_payload_for_boxes(
    estimator: PoseEstimator,
    frame_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Estimate pose for client-supplied ``[x1, y1, x2, y2]`` boxes.

    Drives the decoupled pose pass (``POST /api/tune/pose``): the tuner sends the
    detection boxes it already buffered, so pose runs **without re-running YOLO**.
    Boxes are clamped to the frame and degenerate (zero/negative-area) boxes are
    skipped. All valid boxes on the frame are submitted to ``estimate_batch`` as a
    single batch (one GPU forward where the backend supports it); one entry is
    returned per box that yields keypoints, in input order.
    """

    from detectivepotty.pose.base import PoseRequest

    valid_bboxes = _clamp_valid_bboxes(frame_bgr, boxes)
    if not valid_bboxes:
        return []

    requests = [
        PoseRequest(
            frame_bgr_original=frame_bgr,
            bbox=bbox,
            frame_idx=frame_idx,
            source_id="tune",
        )
        for bbox in valid_bboxes
    ]
    keypoints_list = estimator.estimate_batch(requests)
    out: list[dict[str, Any]] = []
    for bbox, keypoints in zip(valid_bboxes, keypoints_list):
        if keypoints is None:
            continue
        out.append(_pose_entry(bbox, keypoints))
    return out


def pose_payload_for_frames(
    estimator: PoseEstimator,
    items: Sequence[tuple[int, np.ndarray | None, Sequence[Sequence[float]]]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Estimate pose across **multiple frames** in one ``estimate_batch`` forward.

    Drives the batched pose pass (``POST /api/tune/pose_range``). ``items`` is a
    sequence of ``(index, frame_bgr_or_None, boxes)``. Every frame's valid boxes
    are collected into a **single** ``estimate_batch`` call so the backend runs one
    batched GPU forward for the whole window (the SuperAnimal backend measured
    ~9-14x faster than the batch-1 per-frame path), then keypoints are distributed
    back to their source frame in input order.

    A ``None`` frame (e.g. one that failed to decode) contributes no crops but is
    still returned with an empty entry list, so **every requested index appears in
    the output exactly once**. That lets the caller mark un-decodable frames
    terminal instead of retrying them forever. Returns ``list[(index, entries)]``
    aligned 1:1 with ``items``.
    """

    from detectivepotty.pose.base import PoseRequest

    owners: list[int] = []
    bboxes: list[BBox] = []
    requests: list[PoseRequest] = []
    for pos, (index, frame_bgr, boxes) in enumerate(items):
        if frame_bgr is None:
            continue
        for bbox in _clamp_valid_bboxes(frame_bgr, boxes):
            owners.append(pos)
            bboxes.append(bbox)
            requests.append(
                PoseRequest(
                    frame_bgr_original=frame_bgr,
                    bbox=bbox,
                    frame_idx=index,
                    source_id="tune",
                )
            )

    entries_per_item: list[list[dict[str, Any]]] = [[] for _ in items]
    if requests:
        keypoints_list = estimator.estimate_batch(requests)
        for pos, bbox, keypoints in zip(owners, bboxes, keypoints_list):
            if keypoints is None:
                continue
            entries_per_item[pos].append(_pose_entry(bbox, keypoints))
    return [(items[pos][0], entries_per_item[pos]) for pos in range(len(items))]


def build_tune_pose_estimator(config: Config) -> tuple[PoseEstimator | None, bool]:
    """Resolve a pose estimator for the tuner, or ``(None, False)`` if unavailable.

    The tuner overlay is independent of ``pose.enabled`` (that flag gates the
    production gate/classifier), so we force ``enabled=True`` off a copy of the
    pose config. The real SuperAnimal backend needs the optional ``deeplabcut``
    dependency, so we report it unavailable (rather than erroring) when that
    package is absent. A ``mock`` backend always resolves — handy for exercising
    the overlay without the heavy model.

    ``find_spec`` only proves the dependency is importable, not that inference
    will succeed (model files can still be missing); the endpoint downgrades to
    unavailable if the first ``estimate`` call raises.
    """

    pose_config = config.pose.model_copy(update={"enabled": True})
    if (
        pose_config.backend == "superanimal"
        and importlib.util.find_spec("deeplabcut") is None
    ):
        return None, False
    try:
        from detectivepotty.pose.factory import build_pose_estimator

        estimator = build_pose_estimator(pose_config)
    except Exception:  # pragma: no cover - import/build failure path
        return None, False
    return estimator, estimator is not None
