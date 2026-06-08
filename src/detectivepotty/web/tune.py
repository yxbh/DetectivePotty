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
from collections.abc import Sequence
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from detectivepotty.config import Config
    from detectivepotty.events import Detection
    from detectivepotty.pose.base import PoseEstimator

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

    Discovers ``*.pt`` files under ``models_dir`` (the conventional weights
    folder) and always includes the configured ``global.model_name`` so the
    active model is selectable even if it lives elsewhere or is a bare
    Ultralytics name. Each returned string is usable directly as
    ``DogDetector(model_name=...)``; the client labels options by basename.

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

    try:
        children = sorted(
            models_dir.glob("*.pt"), key=lambda p: p.name.lower()
        )
    except OSError:  # pragma: no cover - defensive (e.g. unreadable dir)
        children = []
    for child in children:
        if child.is_file():
            add(str(models_dir / child.name))

    add(config.global_settings.model_name)
    return models


def read_meta(path: Path) -> tuple[int, float, int, int, float]:
    """Return ``(total_frames, fps, width, height, duration_s)`` for ``path``.

    Reads container properties without decoding a frame, so the client can map
    ``video.currentTime`` to a frame index cheaply (no YOLO). ``total_frames`` is
    ``0`` when the container does not report a reliable count; ``duration_s`` is
    ``0.0`` when it cannot be derived.
    """

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"failed to open video file: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total = total if total > 0 else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    duration = (total / fps) if (total > 0 and fps > 0) else 0.0
    return total, fps, width, height, duration


def read_frame(
    path: Path, index: int
) -> tuple[np.ndarray, int, int, float, int, int]:
    """Decode a single frame from ``path`` at ``index``.

    Returns ``(frame_bgr, idx_used, total_frames, fps, width, height)``.
    ``total_frames`` is ``0`` when the container does not report a reliable
    count. The capture is opened and released per call so the endpoint stays
    stateless and thread-safe under ``run_in_threadpool``. Raises ``IndexError``
    if the requested frame cannot be read (e.g. past the end).
    """

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"failed to open video file: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total = total if total > 0 else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        idx = max(0, index)
        if total > 0:
            idx = min(idx, total - 1)
        if idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise IndexError(f"no frame at index {index}")
    height, width = frame.shape[:2]
    return frame, idx, total, fps, width, height


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


def pose_payload(
    estimator: PoseEstimator,
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Estimate pose for each detection and shape keypoints for the overlay.

    One entry per detection that yields keypoints; coordinates are
    original-resolution pixels (same space as the boxes), so the client draws
    boxes and keypoints in one coordinate frame.
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
        out.append(
            {
                "bbox": [
                    float(det.bbox.x1),
                    float(det.bbox.y1),
                    float(det.bbox.x2),
                    float(det.bbox.y2),
                ],
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
        )
    return out


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
