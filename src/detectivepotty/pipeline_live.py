"""Live camera worker and detection loop for the potty pipeline."""

from __future__ import annotations

from collections.abc import Callable
import asyncio
import math
from pathlib import Path
import threading
from typing import Any

from detectivepotty.classify.base import PottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.pipeline_recording import record_all_pending, record_ready_pending
from detectivepotty.pipeline_runtime import (
    Detector,
    FrameHistory,
    PendingCandidate,
    RecorderFactory,
    buffer_window_s,
    detect_frames_batched,
    live_buffer_max_frames,
)
from detectivepotty.potty_event import PottyEventDetector
from detectivepotty.recording.retention import enforce_retention
from detectivepotty.sources.base import Frame, VideoSource, sanitize_source_id
from detectivepotty.sources.rolling_buffer import BufferedSourceWorker, RollingBuffer


async def run_live_camera_async(
    camera_config: CameraConfig,
    url: str,
    recorder_client: Any | None,
    *,
    config: Config,
    new_detector: Callable[[CameraConfig], Detector],
    new_classifier: Callable[[CameraConfig], PottyClassifier],
    new_state_machine: Callable[[CameraConfig], PottyEventDetector],
    recorder_factory: RecorderFactory,
    make_live_source: Callable[[str], VideoSource],
    inference_lock: Any,
    stop_event: threading.Event,
    max_live_frames: int | None,
) -> list[Path]:
    source = make_live_source(url)
    window_s = buffer_window_s(camera_config)
    buffer_max_frames = live_buffer_max_frames(window_s)
    buffer = RollingBuffer(window_s, max_frames=buffer_max_frames)
    worker = BufferedSourceWorker(
        source,
        buffer,
        name=f"buffer-{sanitize_source_id(camera_config.id)}",
    )
    worker.start()
    try:
        return await _run_live_detection_loop(
            camera_config,
            buffer,
            recorder_client,
            window_s,
            config=config,
            new_detector=new_detector,
            new_classifier=new_classifier,
            new_state_machine=new_state_machine,
            recorder_factory=recorder_factory,
            inference_lock=inference_lock,
            stop_event=stop_event,
            max_live_frames=max_live_frames,
        )
    finally:
        worker.stop()


async def _run_live_detection_loop(
    camera_config: CameraConfig,
    buffer: RollingBuffer,
    recorder_client: Any,
    window_s: float,
    *,
    config: Config,
    new_detector: Callable[[CameraConfig], Detector],
    new_classifier: Callable[[CameraConfig], PottyClassifier],
    new_state_machine: Callable[[CameraConfig], PottyEventDetector],
    recorder_factory: RecorderFactory,
    inference_lock: Any,
    stop_event: threading.Event,
    max_live_frames: int | None,
) -> list[Path]:
    detector = new_detector(camera_config)
    classifier = new_classifier(camera_config)
    state_machine = new_state_machine(camera_config)
    recorder = recorder_factory(config, recorder_client)
    history = FrameHistory(window_s, max_frames=live_buffer_max_frames(window_s))
    pending: list[PendingCandidate] = []
    event_dirs: list[Path] = []
    last_key: tuple[str, int] | None = None
    last_detection_mono = -math.inf
    processed = 0
    sample_interval_s = 1.0 / camera_config.sample_rate_fps
    live_batch_size = max(1, config.global_settings.live_detection_batch_size)
    max_batch_wait_s = config.global_settings.max_batch_wait_s

    # Sampled frames accumulate here until the batch is full or has waited
    # ``max_batch_wait_s``; then they are detected in one forward and replayed
    # through the state machine in order. ``live_batch_size == 1`` (default)
    # flushes immediately, i.e. exactly the original per-frame behavior.
    batch: list[Frame] = []
    batch_started_mono: float | None = None

    def flush_live_batch() -> None:
        nonlocal batch_started_mono
        if not batch:
            return
        results = detect_frames_batched(detector, batch, lock=inference_lock)
        for batched_frame, detections in zip(batch, results):
            emitted = state_machine.process(batched_frame, detections)
            pending.extend(PendingCandidate(candidate) for candidate in emitted)
        batch.clear()
        batch_started_mono = None

    while not stop_event.is_set() and (
        max_live_frames is None or processed < max_live_frames
    ):
        snapshot = buffer.snapshot()
        if not snapshot:
            await asyncio.sleep(0.05)
            continue
        frame = snapshot[-1]
        key = (frame.source_id, frame.frame_idx)
        if key == last_key:
            await asyncio.sleep(0.02)
            continue
        last_key = key
        processed += 1
        history.append(frame)

        if frame.mono_ts - last_detection_mono >= sample_interval_s:
            last_detection_mono = frame.mono_ts
            batch.append(frame)
            if batch_started_mono is None:
                batch_started_mono = frame.mono_ts
            if (
                len(batch) >= live_batch_size
                or frame.mono_ts - batch_started_mono >= max_batch_wait_s
            ):
                flush_live_batch()

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

    # Drain any partial batch accumulated before the loop exited so its frames
    # still reach the state machine before the final flush.
    flush_live_batch()
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
    enforce_retention(config.global_settings.dataset_dir, camera_config)
    return event_dirs
