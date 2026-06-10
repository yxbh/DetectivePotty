"""Orchestrate the retro-harvest strategy bake-off over one acquired window.

Builds the exhaustive dense-YOLO ground truth, runs each candidate window-selection
strategy, and scores them all against it (recall of true dog-seconds vs compute
saved). The exhaustive pass itself is entered as the ``blind-scrub`` baseline
(recall 1.0, 0 compute saved); ``compressed-motion`` is swept across thresholds to
trace its recall/compute trade-off curve.

A long window is acquired as several chunk files; :func:`run_bakeoff_dir` runs one
ground-truth pass per chunk and aggregates them into a single window-wide report.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .groundtruth import (
    DetectorLike,
    GroundTruth,
    detect_frames,
    iter_video_frames,
    video_fps,
)
from .motion import (
    compressed_motion_timeline,
    per_second_energy,
    probe_packets,
    select_by_threshold,
)
from ..sources.prefetch import DEFAULT_PREFETCH, prefetch
from .timeline import (
    BakeoffReport,
    SecondTimeline,
    StrategyScore,
    aggregate_scores,
    score_strategy,
)

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.40)
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".m4v", ".ts")


def build_ground_truth(
    video_path: str,
    detector: DetectorLike,
    *,
    batch_size: int = 32,
    prefetch_frames: int = DEFAULT_PREFETCH,
) -> GroundTruth:
    """Exhaustive every-frame detection over ``video_path`` (the reference pass).

    Decode and inference are pipelined: a background thread reads ahead up to
    ``prefetch_frames`` frames while the main thread runs batched inference, so the
    decode-bound pass becomes inference-bound. ``prefetch_frames <= 0`` disables the
    read-ahead thread (synchronous decode, e.g. for deterministic debugging).
    """

    fps = video_fps(video_path)
    frames = iter_video_frames(video_path, every_n=1)
    if prefetch_frames and prefetch_frames > 0:
        frames = prefetch(frames, max_prefetch=prefetch_frames)
    return detect_frames(frames, detector, fps=fps, batch_size=batch_size)


def _score_one_video(
    video_path: str,
    detector: DetectorLike,
    *,
    thresholds,
    min_dog_frames: int,
    pad_s: int,
    batch_size: int,
    prefetch_frames: int = DEFAULT_PREFETCH,
) -> tuple[list[StrategyScore], int, int]:
    """Score every strategy on one video; return (scores, duration_s, dog_seconds)."""

    gt = build_ground_truth(
        video_path, detector, batch_size=batch_size, prefetch_frames=prefetch_frames
    )
    ground = gt.dog_seconds(min_dog_frames=min_dog_frames)
    duration = ground.duration_s

    # blind-scrub baseline = run YOLO on everything (recall 1.0, no compute saved).
    scores: list[StrategyScore] = [
        score_strategy("blind-scrub", SecondTimeline.full(duration), ground)
    ]
    # compressed-domain motion, swept across thresholds (pixel-free pre-filter).
    packets = probe_packets(video_path)
    energy = per_second_energy(packets, duration)
    for frac in thresholds:
        sel = select_by_threshold(energy, threshold_frac=frac).dilate(pad_s)
        scores.append(score_strategy(f"motion@{frac:.2f}", sel, ground))
    return scores, duration, ground.count


def run_bakeoff(
    video_path: str,
    detector: DetectorLike,
    *,
    source: str | None = None,
    thresholds=DEFAULT_THRESHOLDS,
    min_dog_frames: int = 1,
    pad_s: int = 1,
    batch_size: int = 32,
    prefetch_frames: int = DEFAULT_PREFETCH,
) -> BakeoffReport:
    """Score every strategy over one window file. Heavy (builds ground truth)."""

    scores, duration, dog_seconds = _score_one_video(
        video_path,
        detector,
        thresholds=thresholds,
        min_dog_frames=min_dog_frames,
        pad_s=pad_s,
        batch_size=batch_size,
        prefetch_frames=prefetch_frames,
    )
    report = BakeoffReport(
        source=source or video_path,
        duration_s=duration,
        ground_truth_dog_seconds=dog_seconds,
    )
    report.scores.extend(scores)
    return report


def find_chunk_videos(chunk_dir: str | Path) -> list[Path]:
    """Sorted list of chunk video files in ``chunk_dir`` (acquire output)."""

    root = Path(chunk_dir)
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def run_bakeoff_dir(
    video_paths,
    detector: DetectorLike,
    *,
    source: str | None = None,
    thresholds=DEFAULT_THRESHOLDS,
    min_dog_frames: int = 1,
    pad_s: int = 1,
    batch_size: int = 32,
    prefetch_frames: int = DEFAULT_PREFETCH,
    progress=None,
) -> BakeoffReport:
    """Score a sequence of chunk videos and aggregate into one window-wide report.

    Each chunk gets its own exhaustive ground-truth pass; per-strategy scores are
    summed across chunks (see :func:`aggregate_scores`) so the table reflects the
    whole acquired window. ``progress(i, n, path)`` is called before each chunk.
    """

    paths = [Path(p) for p in video_paths]
    if not paths:
        raise ValueError("run_bakeoff_dir requires at least one video")

    per_strategy: "OrderedDict[str, list[StrategyScore]]" = OrderedDict()
    total_duration = 0
    total_dog = 0
    for i, path in enumerate(paths):
        if progress is not None:
            progress(i, len(paths), path)
        scores, duration, dog_seconds = _score_one_video(
            str(path),
            detector,
            thresholds=thresholds,
            min_dog_frames=min_dog_frames,
            pad_s=pad_s,
            batch_size=batch_size,
            prefetch_frames=prefetch_frames,
        )
        total_duration += duration
        total_dog += dog_seconds
        for s in scores:
            per_strategy.setdefault(s.name, []).append(s)

    report = BakeoffReport(
        source=source or f"{paths[0].parent} ({len(paths)} chunks)",
        duration_s=total_duration,
        ground_truth_dog_seconds=total_dog,
    )
    for name, scores in per_strategy.items():
        report.scores.append(aggregate_scores(name, scores))
    return report


__all__ = [
    "build_ground_truth",
    "find_chunk_videos",
    "run_bakeoff",
    "run_bakeoff_dir",
    "DEFAULT_THRESHOLDS",
    "compressed_motion_timeline",
]
