"""Event-window recording helpers shared by file and live pipeline loops."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.config import CameraConfig
from detectivepotty.pipeline_runtime import (
    FrameHistory,
    PendingCandidate,
    candidate_mono_bounds,
    dedupe_frames,
    primary_track,
)
from detectivepotty.potty_event import PottyCandidate
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.sources.base import Frame, sanitize_source_id
from detectivepotty.sources.rolling_buffer import RollingBuffer

LOGGER = logging.getLogger(__name__)


def record_ready_pending(
    pending: list[PendingCandidate],
    *,
    current_wall_ts: datetime,
    buffer: RollingBuffer,
    history: FrameHistory,
    camera_config: CameraConfig,
    classifier: PottyClassifier,
    recorder: EventRecorder,
) -> list[Path]:
    ready: list[PendingCandidate] = []
    waiting: list[PendingCandidate] = []
    for item in pending:
        ready_at = item.candidate.end_ts + timedelta(seconds=camera_config.post_roll_s)
        if current_wall_ts >= ready_at:
            ready.append(item)
        else:
            waiting.append(item)
    pending[:] = waiting
    return [
        path
        for item in ready
        if (
            path := record_candidate(
                item,
                buffer=buffer,
                history=history,
                camera_config=camera_config,
                classifier=classifier,
                recorder=recorder,
            )
        )
        is not None
    ]


def record_all_pending(
    pending: list[PendingCandidate],
    *,
    buffer: RollingBuffer,
    history: FrameHistory,
    camera_config: CameraConfig,
    classifier: PottyClassifier,
    recorder: EventRecorder,
) -> list[Path]:
    ready = list(pending)
    pending.clear()
    return [
        path
        for item in ready
        if (
            path := record_candidate(
                item,
                buffer=buffer,
                history=history,
                camera_config=camera_config,
                classifier=classifier,
                recorder=recorder,
            )
        )
        is not None
    ]


def record_candidate(
    pending: PendingCandidate,
    *,
    buffer: RollingBuffer,
    history: FrameHistory,
    camera_config: CameraConfig,
    classifier: PottyClassifier,
    recorder: EventRecorder,
) -> Path | None:
    candidate = pending.candidate
    frames = assemble_event_window(buffer, history, candidate, camera_config)
    if not frames:
        LOGGER.warning(
            "Skipping empty event window for camera %s at %s",
            sanitize_source_id(camera_config.id),
            candidate.start_ts.isoformat(),
        )
        return None

    classifier_result: ClassifierResult | None = None
    primary = primary_track(candidate)
    if primary is not None:
        classifier_result = classifier.classify(primary, frames)

    event_dir = recorder.record(
        candidate,
        frames,
        camera_config,
        classifier_result=classifier_result,
        protect_meta=pending.protect_meta,
    )
    LOGGER.info("Recorded event %s", event_dir)
    return event_dir


def assemble_event_window(
    buffer: RollingBuffer,
    history: FrameHistory,
    candidate: PottyCandidate,
    camera_config: CameraConfig,
) -> list[Frame]:
    frames = EventRecorder.assemble_window(buffer, candidate, camera_config)
    start_wall = candidate.start_ts - timedelta(seconds=camera_config.pre_roll_s)
    end_wall = candidate.end_ts + timedelta(seconds=camera_config.post_roll_s)
    fallback = history.by_wall(start_wall, end_wall)
    if len(fallback) > len(frames):
        frames = fallback

    mono_bounds = candidate_mono_bounds(candidate, camera_config)
    if mono_bounds is not None:
        mono_fallback = history.by_mono(*mono_bounds)
        if len(mono_fallback) > len(frames):
            frames = mono_fallback
    return dedupe_frames(frames)
