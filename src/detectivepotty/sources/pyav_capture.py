"""PyAV-backed capture adapter exposing the ``cv2.VideoCapture`` duck interface.

Multithreaded software H.264 decode via FFmpeg's libav (PyAV) runs ~2.5–3x faster
than OpenCV's single-threaded ``cv2.VideoCapture`` on this project's 2688x1512
footage, while yielding the same BGR frames in the same order. Hardware decode
(VideoToolbox) was measured *slower* for this download-to-numpy workload — a single
HW decode session consumed serially loses to FFmpeg spreading frame/slice threading
across the performance cores, and HW frames must be read back from the GPU before the
``bgr24`` conversion — so it is deliberately **not** used here.

:class:`PyAvCapture` implements only the slice of the ``cv2.VideoCapture`` API the
codebase relies on (``read`` / ``get`` / ``set`` / ``isOpened`` / ``release``) so it
drops into the existing ``capture_factory`` seams (``FileSource``, ``harvest``,
``dataset_export``, ``experiment.groundtruth``) with no call-site changes. ``set``
supports only frame-accurate forward seeking (``CAP_PROP_POS_FRAMES``), which the
harvest cut pass uses to decode only dog-present windows; the tuner / preview
random-seek surfaces keep ``cv2``.

The factory is selectable via :func:`make_capture_factory` (or the
``DETECTIVEPOTTY_DECODE_BACKEND`` env var: ``auto`` (default) / ``pyav`` / ``opencv``)
and always falls back to OpenCV when PyAV cannot open or decode a file, so a decode
backend problem degrades gracefully instead of breaking a run.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import cv2

logger = logging.getLogger(__name__)

ENV_VAR = "DETECTIVEPOTTY_DECODE_BACKEND"

# Thread model for FFmpeg software decode. "AUTO" lets libav pick frame/slice
# threading across cores — the source of the ~2.5–3x speedup over OpenCV.
_THREAD_TYPE = "AUTO"


class PyAvDecodeError(RuntimeError):
    """Raised when PyAV opens a video but fails while decoding frames."""


class PyAvCapture:
    """A ``cv2.VideoCapture``-compatible reader backed by PyAV.

    Sequential ``read()`` decodes frames in container order as contiguous BGR
    ``uint8`` ndarrays (matching OpenCV) so the surrounding frame-index/timestamp
    math is unchanged. ``set(cv2.CAP_PROP_POS_FRAMES, n)`` adds a frame-accurate
    forward seek (used by the harvest clip-extraction pass to decode only the
    dog-present windows); arbitrary random seeking is otherwise out of scope.
    """

    def __init__(self, path: str, *, thread_type: str = _THREAD_TYPE) -> None:
        self._path = str(path)
        self._container: Any | None = None
        self._frames: Any | None = None
        self._stream: Any | None = None
        self._time_base: Any | None = None
        self._opened = False
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count = 0
        self._pending: Any | None = None
        self._open(thread_type)

    def _open(self, thread_type: str) -> None:
        try:
            import av

            container = av.open(self._path)
            stream = container.streams.video[0]
            try:
                stream.thread_type = thread_type
            except Exception:  # pragma: no cover - older PyAV without thread_type
                pass
            codec_context = stream.codec_context
            self._width = int(getattr(codec_context, "width", 0) or 0)
            self._height = int(getattr(codec_context, "height", 0) or 0)
            rate = stream.average_rate or stream.base_rate or getattr(
                codec_context, "framerate", None
            )
            self._fps = float(rate) if rate else 0.0
            self._frame_count = int(stream.frames or 0)
            self._container = container
            self._stream = stream
            self._time_base = stream.time_base
            self._frames = container.decode(video=0)
            self._opened = True
        except Exception as exc:
            logger.debug("PyAvCapture could not open %s: %s", self._path, exc)
            self._opened = False
            self._close_container()

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2.VideoCapture
        return self._opened

    def read(self):  # -> tuple[bool, np.ndarray | None]
        if not self._opened or self._frames is None:
            return False, None
        if self._pending is not None:
            array = self._pending
            self._pending = None
            return True, array
        try:
            frame = next(self._frames)
        except StopIteration:
            return False, None
        except Exception as exc:  # pragma: no cover - corrupt/truncated stream
            raise PyAvDecodeError(f"PyAV decode error on {self._path}") from exc
        try:
            array = frame.to_ndarray(format="bgr24")
        except Exception as exc:  # pragma: no cover - unexpected pixel format
            raise PyAvDecodeError(
                f"PyAV frame conversion error on {self._path}"
            ) from exc
        return True, array

    def set(self, prop: int, value: float) -> bool:  # noqa: N802 - mirror cv2
        """Frame-accurate forward seek for ``cv2.CAP_PROP_POS_FRAMES``.

        Seeks the container to the keyframe at/before frame ``value`` then decodes
        forward, discarding pre-roll, so the next :meth:`read` returns *exactly*
        that frame. Relies on the constant frame rate of the NVR clips (frame
        index is derived from each frame's ``pts``). Returns ``True`` on success;
        any other property or a failed seek returns ``False`` (callers fall back
        to a sequential pass). Only ``CAP_PROP_POS_FRAMES`` is supported.
        """

        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        if not self._opened or self._container is None or self._stream is None:
            return False
        if self._fps <= 0 or not self._time_base:
            return False
        target = int(value)
        if target < 0:
            return False
        try:
            import av  # noqa: F401  (ensure backend present)

            offset = int(target / self._fps / self._time_base)
            self._container.seek(
                offset, stream=self._stream, backward=True, any_frame=False
            )
            self._frames = self._container.decode(video=0)
            self._pending = None
            while True:
                frame = next(self._frames)
                idx = self._frame_index(frame)
                if idx is None:
                    return False
                if idx >= target:
                    self._pending = frame.to_ndarray(format="bgr24")
                    return True
        except StopIteration:
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PyAvCapture seek error on %s: %s", self._path, exc)
            return False

    def _frame_index(self, frame: Any) -> int | None:
        """CFR frame index from a decoded frame's ``pts`` (or ``None``)."""

        pts = getattr(frame, "pts", None)
        if pts is None:
            return None
        return int(round(float(pts * self._time_base) * self._fps))

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return 0.0

    def release(self) -> None:
        self._frames = None
        self._pending = None
        self._close_container()
        self._opened = False

    def _close_container(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:  # pragma: no cover - best-effort close
                pass
        self._container = None


def _pyav_factory(path: str) -> Any:
    """Open via PyAV; transparently fall back to OpenCV on any failure."""

    try:
        capture = PyAvCapture(path)
    except Exception as exc:  # pragma: no cover - defensive; PyAvCapture swallows
        logger.info("PyAV capture unavailable for %s (%s); using OpenCV", path, exc)
        return cv2.VideoCapture(path)
    if capture.isOpened():
        return capture
    logger.info("PyAV could not open %s; falling back to OpenCV", path)
    return cv2.VideoCapture(path)


def make_capture_factory(backend: str = "auto") -> Callable[[str], Any]:
    """Return a ``path -> capture`` factory for the requested decode backend.

    ``opencv`` -> ``cv2.VideoCapture``; ``pyav`` / ``auto`` -> PyAV with automatic
    OpenCV fallback. ``auto`` also falls back to OpenCV if PyAV cannot be imported.
    """

    normalized = (backend or "auto").strip().lower()
    if normalized == "opencv":
        return cv2.VideoCapture
    if normalized in ("pyav", "auto"):
        if normalized == "auto":
            try:
                import av  # noqa: F401
            except Exception:  # pragma: no cover - av is a hard dep of uiprotect
                logger.info("PyAV not importable; defaulting decode backend to OpenCV")
                return cv2.VideoCapture
        return _pyav_factory
    raise ValueError(
        f"unknown decode backend {backend!r} (expected 'auto', 'pyav', or 'opencv')"
    )


def default_backend() -> str:
    """The decode backend name from the env var, defaulting to ``auto``."""

    return os.environ.get(ENV_VAR, "auto")


def open_capture(path: str) -> Any:
    """Default capture factory: open ``path`` with the env-selected backend.

    Used as the default ``capture_factory`` across the decode seams so the backend
    can be overridden at runtime via ``DETECTIVEPOTTY_DECODE_BACKEND`` without
    changing any call sites.
    """

    return make_capture_factory(default_backend())(str(path))
