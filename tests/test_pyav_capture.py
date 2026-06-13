"""Offline tests for the PyAV capture adapter + backend selector.

A tiny clip is synthesized in-process with PyAV (no network/model/GPU), then decoded
through both ``cv2.VideoCapture`` and :class:`PyAvCapture` to assert the adapter is a
faithful, frame-for-frame drop-in for the existing ``capture_factory`` seam.
"""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from detectivepotty.sources.pyav_capture import (
    ENV_VAR,
    PyAvCapture,
    PyAvDecodeError,
    default_backend,
    make_capture_factory,
    open_capture,
)

CLIP_FRAMES = 24
CLIP_W = 160
CLIP_H = 120
CLIP_FPS = 12


def _make_clip(path: Path, *, n: int = CLIP_FRAMES, codec: str = "mpeg4") -> None:
    """Encode a small deterministic clip (mpeg4 = always available in FFmpeg LGPL)."""

    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=CLIP_FPS)
    stream.width = CLIP_W
    stream.height = CLIP_H
    stream.pix_fmt = "yuv420p"
    for i in range(n):
        img = np.zeros((CLIP_H, CLIP_W, 3), dtype=np.uint8)
        img[:, :, 2] = (i * 9) % 256  # ramp the red channel per frame
        img[20:40, 20:40] = 255  # a constant white square
        for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.mp4"
    _make_clip(path)
    return path


def _read_all(capture) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_reads_all_frames_with_correct_shape(clip: Path) -> None:
    capture = PyAvCapture(str(clip))
    assert capture.isOpened()
    frames = _read_all(capture)
    assert len(frames) == CLIP_FRAMES
    assert frames[0].shape == (CLIP_H, CLIP_W, 3)
    assert frames[0].dtype == np.uint8


def test_get_reports_stream_metadata(clip: Path) -> None:
    capture = PyAvCapture(str(clip))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(CLIP_FPS, abs=0.01)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == CLIP_W
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == CLIP_H
        # FRAME_COUNT may be 0 for some containers; when present it must be right.
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        assert count in (0.0, float(CLIP_FRAMES))
        assert capture.get(cv2.CAP_PROP_POS_MSEC) == 0.0  # unknown props -> 0.0
    finally:
        capture.release()


def test_frame_parity_with_opencv(clip: Path) -> None:
    """Same file, same frames, same order — within codec rounding tolerance."""

    cv2_frames = _read_all(cv2.VideoCapture(str(clip)))
    pyav_frames = _read_all(PyAvCapture(str(clip)))

    assert len(cv2_frames) == len(pyav_frames) == CLIP_FRAMES
    for a, b in zip(cv2_frames, pyav_frames):
        assert a.shape == b.shape
        mean_abs = float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
        assert mean_abs < 2.0


def test_read_after_release_returns_eof(clip: Path) -> None:
    capture = PyAvCapture(str(clip))
    capture.release()
    assert capture.read() == (False, None)


def test_open_failure_reports_not_opened(tmp_path: Path) -> None:
    capture = PyAvCapture(str(tmp_path / "does_not_exist.mp4"))
    assert capture.isOpened() is False
    assert capture.read() == (False, None)


def test_decode_error_is_not_reported_as_eof() -> None:
    def broken_frames():
        raise RuntimeError("bad packet")
        yield  # pragma: no cover - makes this a generator

    capture = object.__new__(PyAvCapture)
    capture._path = "broken.mp4"
    capture._opened = True
    capture._frames = broken_frames()
    capture._pending = None

    with pytest.raises(PyAvDecodeError, match="PyAV decode error"):
        capture.read()


def test_make_capture_factory_opencv_is_cv2() -> None:
    assert make_capture_factory("opencv") is cv2.VideoCapture


def test_make_capture_factory_pyav_opens_via_adapter(clip: Path) -> None:
    factory = make_capture_factory("pyav")
    capture = factory(str(clip))
    try:
        assert isinstance(capture, PyAvCapture)
        assert capture.isOpened()
    finally:
        capture.release()


def test_factory_falls_back_to_opencv_on_open_failure(tmp_path: Path) -> None:
    factory = make_capture_factory("auto")
    capture = factory(str(tmp_path / "missing.mp4"))
    # PyAV could not open -> the factory hands back a (closed) cv2 capture.
    assert isinstance(capture, cv2.VideoCapture)
    assert capture.isOpened() is False


def test_make_capture_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        make_capture_factory("ffmpeg-magic")


def test_default_backend_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert default_backend() == "auto"
    monkeypatch.setenv(ENV_VAR, "opencv")
    assert default_backend() == "opencv"


def test_open_capture_honours_env_override(
    clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, "opencv")
    capture = open_capture(str(clip))
    try:
        assert isinstance(capture, cv2.VideoCapture)
    finally:
        capture.release()

    monkeypatch.setenv(ENV_VAR, "pyav")
    capture = open_capture(str(clip))
    try:
        assert isinstance(capture, PyAvCapture)
    finally:
        capture.release()
