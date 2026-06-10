"""Compressed-domain motion: per-second P-frame byte energy via ``ffprobe``.

The cheapest camera-agnostic motion signal we have. In inter-frame video (H.264/H.265)
a predicted frame's *encoded byte size* tracks how much changed since the last frame —
a still scene compresses to almost nothing, a moving dog spends bits. So summing
**non-keyframe packet sizes per second** gives a motion-energy curve **without decoding
a single pixel**, letting us scan 48h in minutes and pick the seconds worth running YOLO
on. Works on ONVIF cameras too (no NVR events required), unlike UniFi smart-detect.

Keyframes (I-frames) are excluded: they are periodically huge regardless of motion and
would swamp the signal. The pure parsing/energy/threshold helpers are unit-tested on
synthetic packet lists; only :func:`probe_packets` shells out to ffprobe.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from .timeline import SecondTimeline


@dataclass(frozen=True)
class Packet:
    """One compressed video packet: when it plays, its encoded size, keyframe-ness."""

    pts_time: float | None
    size: int
    is_keyframe: bool


def parse_packets(obj: dict) -> list[Packet]:
    """Parse ``ffprobe -show_packets -of json`` output into :class:`Packet`s.

    Tolerant of missing ``pts_time`` (some packets carry only ``dts_time``) and of
    string-typed numeric fields (ffprobe emits everything as JSON strings).
    """

    packets: list[Packet] = []
    for raw in obj.get("packets", []):
        t = raw.get("pts_time", raw.get("dts_time"))
        try:
            pts = float(t) if t is not None else None
        except (TypeError, ValueError):
            pts = None
        try:
            size = int(raw.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        flags = str(raw.get("flags", ""))
        packets.append(Packet(pts_time=pts, size=size, is_keyframe="K" in flags))
    return packets


def per_second_energy(
    packets: list[Packet],
    duration_s: int,
    *,
    exclude_keyframes: bool = True,
) -> list[float]:
    """Sum packet bytes into per-second buckets (index = ``int(pts_time)``).

    Returns a list of length ``duration_s``. Packets without a timestamp, or beyond
    ``duration_s``, are ignored. Keyframes are excluded by default so the curve
    reflects *change* (motion), not periodic I-frame refreshes.
    """

    energy = [0.0] * max(0, duration_s)
    for p in packets:
        if p.pts_time is None:
            continue
        if exclude_keyframes and p.is_keyframe:
            continue
        sec = int(p.pts_time)
        if 0 <= sec < len(energy):
            energy[sec] += float(p.size)
    return energy


def select_by_threshold(
    energy: list[float],
    *,
    threshold_frac: float = 0.15,
) -> SecondTimeline:
    """Select seconds whose energy exceeds ``threshold_frac`` of the peak second.

    A relative threshold (fraction of the per-video max) auto-adapts to each
    camera's bitrate without hand-tuned byte counts. The bake-off sweeps
    ``threshold_frac`` to trace the recall vs compute-saved curve; lower keeps more
    seconds (higher recall, less saved), higher is greedier.
    """

    duration = len(energy)
    if duration == 0:
        return SecondTimeline.empty(0)
    peak = max(energy)
    if peak <= 0:
        return SecondTimeline.empty(duration)
    cutoff = threshold_frac * peak
    selected = (s for s, e in enumerate(energy) if e >= cutoff and e > 0)
    return SecondTimeline.from_iterable(selected, duration)


def probe_packets(video_path: str) -> list[Packet]:
    """Run ffprobe to list this video's packets (CLI use; shells out to ffprobe)."""

    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,size,flags",
        "-of",
        "json",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_packets(json.loads(proc.stdout or "{}"))


def compressed_motion_timeline(
    video_path: str,
    duration_s: int,
    *,
    threshold_frac: float = 0.15,
    pad_s: int = 1,
) -> SecondTimeline:
    """End-to-end: ffprobe → per-second energy → threshold → ±``pad_s`` dilation."""

    packets = probe_packets(video_path)
    energy = per_second_energy(packets, duration_s)
    return select_by_threshold(energy, threshold_frac=threshold_frac).dilate(pad_s)
