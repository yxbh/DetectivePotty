"""Finite file-camera processing for the potty pipeline."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import threading
from typing import Any

from detectivepotty.classify.base import PottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.events import Detection
from detectivepotty.pipeline_recording import record_all_pending, record_ready_pending
from detectivepotty.pipeline_runtime import (
    Detector,
    FileSourceFactory,
    FrameHistory,
    PendingCandidate,
    RecorderFactory,
    buffer_window_s,
    detect_frames_batched,
    history_max_frames,
    retimestamp_file_frame,
    sample_every,
    source_fps,
)
from detectivepotty.potty_event import PottyEventDetector
from detectivepotty.recording.retention import enforce_retention
from detectivepotty.sources.base import Frame, sanitize_source_id
from detectivepotty.sources.rolling_buffer import RollingBuffer

LOGGER = logging.getLogger(__name__)


def process_file_camera(
    camera_config: CameraConfig,
    *,
    config: Config,
    new_detector: Callable[[CameraConfig], Detector],
    new_classifier: Callable[[CameraConfig], PottyClassifier],
    new_state_machine: Callable[[CameraConfig], PottyEventDetector],
    recorder_factory: RecorderFactory,
    file_source_factory: FileSourceFactory,
    inference_lock: Any,
    stop_event: threading.Event,
) -> list[Path]:
    if camera_config.input.path is None:
        LOGGER.warning("File camera %s has no input path", sanitize_source_id(camera_config.id))
        return []

    detector = new_detector(camera_config)
    classifier = new_classifier(camera_config)
    state_machine = new_state_machine(camera_config)
    recorder = recorder_factory(config, None)
    window_s = buffer_window_s(camera_config)
    buffer = RollingBuffer(window_s)
    source = file_source_factory(camera_config)
    event_dirs: list[Path] = []
    pending: list[PendingCandidate] = []

    with source:
        fps = source_fps(source, camera_config)
        every = sample_every(fps, camera_config.sample_rate_fps)
        history = FrameHistory(window_s, max_frames=history_max_frames(fps, window_s))
        base_mono: float | None = None
        batch_size = max(1, config.global_settings.file_detection_batch_size)
        max_lookahead = max(1, config.global_settings.max_lookahead_frames)

        while not stop_event.is_set():
            # Read a bounded segment ahead WITHOUT touching buffer/history/state
            # so its sampled frames can be detected in one batched forward. The
            # segment ends once it holds ``batch_size`` sampled frames, hits the
            # lookahead cap, or reaches EOF.
            segment: list[Frame] = []
            sampled_in_segment = 0
            while (
                not stop_event.is_set()
                and sampled_in_segment < batch_size
                and len(segment) < max_lookahead
            ):
                raw_frame = source.read()
                if raw_frame is None:
                    break
                if base_mono is None:
                    base_mono = raw_frame.mono_ts
                frame = retimestamp_file_frame(raw_frame, base_mono, fps)
                segment.append(frame)
                if frame.frame_idx % every == 0:
                    sampled_in_segment += 1

            if not segment:
                break

            sampled = [f for f in segment if f.frame_idx % every == 0]
            detections_by_idx: dict[int, list[Detection]] = {}
            if sampled:
                batch_results = detect_frames_batched(detector, sampled, lock=inference_lock)
                detections_by_idx = {
                    f.frame_idx: dets for f, dets in zip(sampled, batch_results)
                }

            # Replay the segment in frame order: this reproduces the original
            # per-frame loop exactly (buffer/history/state/recording advance in
            # order, up to the processed point), only the detections were
            # precomputed in a batch.
            for frame in segment:
                buffer.append(frame)
                history.append(frame)

                if frame.frame_idx % every == 0:
                    emitted = state_machine.process(frame, detections_by_idx[frame.frame_idx])
                    pending.extend(PendingCandidate(candidate) for candidate in emitted)

                event_dirs.extend(
                    record_ready_pending(
                        pending,
                        current_wall_ts=frame.wall_ts,
                        buffer=buffer,
                        history=history,
                        camera_config=camera_config,
                        classifier=classifier,
                        recorder=recorder,
                    ),
                )

        pending.extend(PendingCandidate(candidate) for candidate in state_machine.flush())
        event_dirs.extend(
            record_all_pending(
                pending,
                buffer=buffer,
                history=history,
                camera_config=camera_config,
                classifier=classifier,
                recorder=recorder,
            ),
        )

    summary = enforce_retention(config.global_settings.dataset_dir, camera_config)
    LOGGER.info(
        "Recorded %d event(s) for file camera %s; retention deleted %d event(s)",
        len(event_dirs),
        sanitize_source_id(camera_config.id),
        summary.deleted_events,
    )
    return event_dirs
