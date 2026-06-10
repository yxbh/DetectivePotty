"""Browser-playable clip encoding (``video_encode.open_h264_writer``).

These tests need a real ffmpeg/ffprobe on PATH to encode + verify the codec, so
they skip cleanly in environments without them (keeping the suite offline-safe).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from detectivepotty.video_encode import open_h264_writer

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    _FFMPEG is None or _FFPROBE is None,
    reason="ffmpeg/ffprobe not available",
)


def _probe_codec(path) -> str:
    out = subprocess.run(
        [
            _FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_open_h264_writer_emits_h264(tmp_path):
    clip = tmp_path / "clip.mp4"
    writer = open_h264_writer(clip, 30.0, (320, 240))
    try:
        for _ in range(15):
            writer.write((np.random.rand(240, 320, 3) * 255).astype(np.uint8))
    finally:
        writer.release()

    assert clip.exists() and clip.stat().st_size > 0
    assert _probe_codec(clip) == "h264"


def test_open_h264_writer_handles_odd_dimensions(tmp_path):
    """libx264/yuv420p needs even dims; odd-sized frames must still encode."""

    clip = tmp_path / "odd.mp4"
    writer = open_h264_writer(clip, 25.0, (321, 241))
    try:
        for _ in range(10):
            writer.write(np.zeros((241, 321, 3), dtype=np.uint8))
    finally:
        writer.release()

    assert clip.exists() and clip.stat().st_size > 0
    assert _probe_codec(clip) == "h264"
