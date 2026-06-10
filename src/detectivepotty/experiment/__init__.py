"""Retro-harvest strategy bake-off (P1c-exp).

Measures the cheapest motion pre-filter that recovers dog windows from long
historical footage, scored against an exhaustive dense-YOLO ground truth. See
``plan.md`` (P1c-exp) for the design and the backend benchmark.
"""

from __future__ import annotations

from .bakeoff import (
    DEFAULT_THRESHOLDS,
    build_ground_truth,
    compressed_motion_timeline,
    find_chunk_videos,
    run_bakeoff,
    run_bakeoff_dir,
)
from .groundtruth import (
    FrameDetection,
    GroundTruth,
    detect_frames,
    iter_video_frames,
    video_fps,
)
from .motion import (
    Packet,
    parse_packets,
    per_second_energy,
    probe_packets,
    select_by_threshold,
)
from .timeline import (
    BakeoffReport,
    SecondTimeline,
    StrategyScore,
    aggregate_scores,
    score_strategy,
)

__all__ = [
    "BakeoffReport",
    "DEFAULT_THRESHOLDS",
    "FrameDetection",
    "GroundTruth",
    "Packet",
    "SecondTimeline",
    "StrategyScore",
    "aggregate_scores",
    "build_ground_truth",
    "compressed_motion_timeline",
    "detect_frames",
    "find_chunk_videos",
    "iter_video_frames",
    "parse_packets",
    "per_second_energy",
    "probe_packets",
    "run_bakeoff",
    "run_bakeoff_dir",
    "score_strategy",
    "select_by_threshold",
    "video_fps",
]
