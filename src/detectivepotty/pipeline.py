"""End-to-end potty detection pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import asyncio
import logging
import math
from pathlib import Path
import threading
from typing import Any

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.classify.heuristic import HeuristicPottyClassifier
from detectivepotty.classify.pose import PosePottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.detect.yolo import DogDetector
from detectivepotty.events import Detection
from detectivepotty.pipeline_runtime import (
    ClassifierFactory,
    Detector,
    DetectorFactory,
    FileSourceFactory,
    FrameHistory as _FrameHistory,
    PendingCandidate as _PendingCandidate,
    RecorderFactory,
    RTSPSourceFactory,
    StateMachineFactory,
    buffer_window_s as _buffer_window_s,
    call_camera_factory as _call_camera_factory,
    candidate_mono_bounds as _candidate_mono_bounds,
    dedupe_frames as _dedupe_frames,
    detect_frames_batched as _detect_frames_batched,
    history_max_frames as _history_max_frames,
    is_live_kind as _is_live_kind,
    is_valid_rtsp_url as _is_valid_rtsp_url,
    live_buffer_max_frames as _live_buffer_max_frames,
    primary_track as _primary_track,
    redact_url as _redact_url,
    retimestamp_file_frame as _retimestamp_file_frame,
    sample_every as _sample_every,
    source_fps as _source_fps,
)
from detectivepotty.pose.base import PoseEstimator
from detectivepotty.pose.factory import build_pose_estimator
from detectivepotty.pose.gate import PoseGate
from detectivepotty.pose.keypoints import PoseKeypoints
from detectivepotty.potty_event import PottyCandidate, PottyEventDetector
from detectivepotty.recording.dataset import camera_dataset_dir
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.recording.retention import enforce_retention
from detectivepotty.sources.base import Frame, VideoSource, sanitize_source_id
from detectivepotty.sources.file import FileSource
from detectivepotty.sources.rolling_buffer import BufferedSourceWorker, RollingBuffer

LOGGER = logging.getLogger(__name__)


class PottyPipeline:
    """Wire sources, dog detection, event state, classification, and recording."""

    def __init__(
        self,
        config: Config,
        *,
        detector_factory: DetectorFactory | None = None,
        classifier_factory: ClassifierFactory | None = None,
        file_source_factory: FileSourceFactory | None = None,
        state_machine_factory: StateMachineFactory | None = None,
        recorder_factory: RecorderFactory | None = None,
        rtsp_source_factory: RTSPSourceFactory | None = None,
        max_live_frames: int | None = None,
        max_workers: int | None = None,
        continue_on_camera_error: bool = True,
    ) -> None:
        self.config = config
        self.detector_factory = detector_factory or self._default_detector_factory
        self.classifier_factory = classifier_factory or self._default_classifier_factory
        self.file_source_factory = file_source_factory or self._default_file_source_factory
        self.state_machine_factory = state_machine_factory or PottyEventDetector
        self.recorder_factory = recorder_factory or self._default_recorder_factory
        self.rtsp_source_factory = rtsp_source_factory
        self.max_live_frames = max_live_frames
        self.max_workers = max_workers
        self.continue_on_camera_error = continue_on_camera_error
        # Cameras run on their own threads; this gates GPU inference because the
        # MPS/torch backend is not reliably safe for concurrent model execution.
        self._inference_lock = threading.Lock()
        # One pose estimator shared across cameras (mirrors the shared detector):
        # None when pose is disabled, in which case the default classifier stays the
        # bbox heuristic. Built once so the heavy model loads at most once per run.
        self._pose_estimator: PoseEstimator | None = build_pose_estimator(config.pose)
        # Cooperative stop flag so live camera loops can be interrupted (Ctrl-C).
        self._stop_event = threading.Event()

    def run(self, camera_ids: Sequence[str] | None = None) -> list[Path]:
        """Run selected cameras and return recorded event directories.

        Cameras run concurrently (one worker thread each by default) when more
        than one is selected, so multiple live cameras are monitored at the same
        time rather than the first one blocking the rest. Each live camera always
        gets a dedicated thread because its loop never returns; finite file
        cameras may share threads. For live cameras the run continues until
        interrupted (Ctrl-C); the returned list is only "complete" for finite
        file/batch cameras.
        """

        self._stop_event.clear()
        selected = self._selected_cameras(camera_ids)
        if not selected:
            LOGGER.info("No enabled cameras selected")
            return []

        self._warn_on_dataset_collisions(selected)
        workers = self._resolve_max_workers(selected)
        # workers == 1 only when there are no live cameras (see _resolve_max_workers),
        # so sequential processing cannot starve a never-ending live loop here.
        if len(selected) == 1 or workers == 1:
            return self._run_sequential(selected)
        return self._run_concurrent(selected, workers)

    def _run_sequential(self, selected: Sequence[CameraConfig]) -> list[Path]:
        event_dirs: list[Path] = []
        try:
            for camera in selected:
                event_dirs.extend(self._process_camera_safe(camera))
        except KeyboardInterrupt:
            LOGGER.info("Interrupted; stopping pipeline")
            self._stop_event.set()
        return event_dirs

    def _run_concurrent(self, selected: Sequence[CameraConfig], workers: int) -> list[Path]:
        LOGGER.info("Processing %d cameras concurrently with %d worker(s)", len(selected), workers)
        results: dict[int, list[Path]] = {}
        # Submit live (protect/rtsp) cameras first so their never-ending loops
        # claim worker slots immediately instead of waiting behind finite file
        # cameras.
        order = sorted(
            range(len(selected)),
            key=lambda i: 0 if _is_live_kind(selected[i].input.kind) else 1,
        )
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="potty-cam")
        try:
            future_to_index = {
                executor.submit(self._process_camera_safe, selected[i]): i for i in order
            }
            try:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        if not self.continue_on_camera_error:
                            self._stop_event.set()
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        LOGGER.warning(
                            "Camera %s failed: %s",
                            sanitize_source_id(selected[index].id),
                            exc,
                        )
                        results[index] = []
            except KeyboardInterrupt:
                LOGGER.info("Interrupted; signaling cameras to stop")
                self._stop_event.set()
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            executor.shutdown(wait=True)
        return [path for index in range(len(selected)) for path in results.get(index, [])]

    def _process_camera_safe(self, camera_config: CameraConfig) -> list[Path]:
        try:
            return self.process_camera(camera_config)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not self.continue_on_camera_error:
                raise
            LOGGER.warning("Camera %s failed: %s", sanitize_source_id(camera_config.id), exc)
            return []

    def _min_safe_workers(self, selected: Sequence[CameraConfig]) -> int:
        # Each live camera needs its own thread because its loop never returns;
        # finite file cameras can share one additional slot.
        live = sum(1 for camera in selected if _is_live_kind(camera.input.kind))
        has_file = any(camera.input.kind == "file" for camera in selected)
        return max(1, live + (1 if has_file else 0))

    def _resolve_max_workers(self, selected: Sequence[CameraConfig]) -> int:
        min_safe = self._min_safe_workers(selected)
        if self.max_workers is not None:
            workers = max(1, self.max_workers)
            if workers < min_safe:
                LOGGER.warning(
                    "max_workers=%d is too low to monitor all selected cameras without "
                    "starving a live camera (need >= %d); using %d instead",
                    workers,
                    min_safe,
                    min_safe,
                )
                workers = min_safe
            return workers
        # Default: one dedicated thread per camera so nothing is ever queued.
        return max(1, len(selected))

    def _warn_on_dataset_collisions(self, selected: Sequence[CameraConfig]) -> None:
        seen: dict[Path, str] = {}
        for camera in selected:
            dataset_dir = camera_dataset_dir(
                self.config.global_settings.dataset_dir,
                camera.id,
                camera.name,
            )
            existing = seen.get(dataset_dir)
            if existing is not None:
                LOGGER.warning(
                    "Cameras %s and %s map to the same dataset directory; "
                    "concurrent retention may race",
                    sanitize_source_id(existing),
                    sanitize_source_id(camera.id),
                )
            else:
                seen[dataset_dir] = camera.id

    def process_camera(self, camera_config: CameraConfig) -> list[Path]:
        if not camera_config.enabled:
            LOGGER.info("Skipping disabled camera %s", sanitize_source_id(camera_config.id))
            return []
        if camera_config.input.kind == "file":
            return self.process_file_camera(camera_config)
        if camera_config.input.kind == "protect":
            return self.process_protect_camera(camera_config)
        if camera_config.input.kind == "rtsp":
            return self.process_rtsp_camera(camera_config)
        LOGGER.warning("Unsupported input kind for camera %s", sanitize_source_id(camera_config.id))
        return []

    def process_file_camera(self, camera_config: CameraConfig) -> list[Path]:
        if camera_config.input.path is None:
            LOGGER.warning("File camera %s has no input path", sanitize_source_id(camera_config.id))
            return []

        detector = self._new_detector(camera_config)
        classifier = self._new_classifier(camera_config)
        state_machine = self._new_state_machine(camera_config)
        recorder = self.recorder_factory(self.config, None)
        buffer_window_s = _buffer_window_s(camera_config)
        buffer = RollingBuffer(buffer_window_s)
        source = self.file_source_factory(camera_config)
        event_dirs: list[Path] = []
        pending: list[_PendingCandidate] = []

        with source:
            source_fps = _source_fps(source, camera_config)
            sample_every = _sample_every(source_fps, camera_config.sample_rate_fps)
            history = _FrameHistory(buffer_window_s, max_frames=_history_max_frames(source_fps, buffer_window_s))
            base_mono: float | None = None
            batch_size = max(1, self.config.global_settings.file_detection_batch_size)
            max_lookahead = max(1, self.config.global_settings.max_lookahead_frames)

            while not self._stop_event.is_set():
                # Read a bounded segment ahead WITHOUT touching buffer/history/state
                # so its sampled frames can be detected in one batched forward. The
                # segment ends once it holds ``batch_size`` sampled frames, hits the
                # lookahead cap, or reaches EOF.
                segment: list[Frame] = []
                sampled_in_segment = 0
                while (
                    not self._stop_event.is_set()
                    and sampled_in_segment < batch_size
                    and len(segment) < max_lookahead
                ):
                    raw_frame = source.read()
                    if raw_frame is None:
                        break
                    if base_mono is None:
                        base_mono = raw_frame.mono_ts
                    frame = _retimestamp_file_frame(raw_frame, base_mono, source_fps)
                    segment.append(frame)
                    if frame.frame_idx % sample_every == 0:
                        sampled_in_segment += 1

                if not segment:
                    break

                sampled = [f for f in segment if f.frame_idx % sample_every == 0]
                detections_by_idx: dict[int, list[Detection]] = {}
                if sampled:
                    batch_results = _detect_frames_batched(
                        detector, sampled, lock=self._inference_lock
                    )
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

                    if frame.frame_idx % sample_every == 0:
                        emitted = state_machine.process(
                            frame, detections_by_idx[frame.frame_idx]
                        )
                        pending.extend(
                            _PendingCandidate(candidate) for candidate in emitted
                        )

                    event_dirs.extend(
                        self._record_ready_pending(
                            pending,
                            current_wall_ts=frame.wall_ts,
                            buffer=buffer,
                            history=history,
                            camera_config=camera_config,
                            classifier=classifier,
                            recorder=recorder,
                        ),
                    )

            pending.extend(_PendingCandidate(candidate) for candidate in state_machine.flush())
            event_dirs.extend(
                self._record_all_pending(
                    pending,
                    buffer=buffer,
                    history=history,
                    camera_config=camera_config,
                    classifier=classifier,
                    recorder=recorder,
                ),
            )

        summary = enforce_retention(self.config.global_settings.dataset_dir, camera_config)
        LOGGER.info(
            "Recorded %d event(s) for file camera %s; retention deleted %d event(s)",
            len(event_dirs),
            sanitize_source_id(camera_config.id),
            summary.deleted_events,
        )
        return event_dirs

    def process_protect_camera(self, camera_config: CameraConfig) -> list[Path]:
        if not self.config.protect_configured():
            LOGGER.warning(
                "Protect is not configured; skipping camera %s",
                sanitize_source_id(camera_config.id),
            )
            return []
        try:
            return asyncio.run(self._process_protect_camera_async(camera_config))
        except Exception as exc:
            if not self.continue_on_camera_error:
                raise
            LOGGER.warning(
                "Protect camera %s skipped: %s",
                sanitize_source_id(camera_config.id),
                exc,
            )
            return []

    def process_rtsp_camera(self, camera_config: CameraConfig) -> list[Path]:
        url = camera_config.input.resolve_url()
        if not url:
            LOGGER.warning(
                "RTSP camera %s has no URL (env var %s is unset or empty)",
                sanitize_source_id(camera_config.id),
                camera_config.input.url_env,
            )
            return []
        if not _is_valid_rtsp_url(url):
            LOGGER.warning(
                "RTSP camera %s has an invalid URL in env var %s (expected rtsp:// or rtsps://)",
                sanitize_source_id(camera_config.id),
                camera_config.input.url_env,
            )
            return []
        try:
            return asyncio.run(self._run_live_camera_async(camera_config, url, None))
        except Exception as exc:
            if not self.continue_on_camera_error:
                raise
            LOGGER.warning(
                "RTSP camera %s skipped: %s",
                sanitize_source_id(camera_config.id),
                _redact_url(str(exc), url),
            )
            return []

    async def _process_protect_camera_async(self, camera_config: CameraConfig) -> list[Path]:
        try:
            from detectivepotty.protect.client import ProtectClient
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("Protect dependencies are unavailable") from exc

        protect_client = ProtectClient(self.config)
        try:
            await protect_client.connect()
            url = protect_client.rtsps_url(camera_config.id, camera_config.substream_choice)
            if not url:
                LOGGER.warning(
                    "No RTSPS URL for camera %s substream %s",
                    sanitize_source_id(camera_config.id),
                    camera_config.substream_choice,
                )
                return []
            return await self._run_live_camera_async(camera_config, url, protect_client)
        finally:
            await protect_client.close()

    def _make_live_source(self, url: str) -> VideoSource:
        if self.rtsp_source_factory is not None:
            return self.rtsp_source_factory(url)
        try:
            from detectivepotty.sources.rtsp import RTSPSource
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("RTSP dependencies are unavailable") from exc
        return RTSPSource(url)

    async def _run_live_camera_async(
        self,
        camera_config: CameraConfig,
        url: str,
        recorder_client: Any | None,
    ) -> list[Path]:
        source = self._make_live_source(url)
        buffer_window_s = _buffer_window_s(camera_config)
        buffer_max_frames = _live_buffer_max_frames(buffer_window_s)
        buffer = RollingBuffer(buffer_window_s, max_frames=buffer_max_frames)
        worker = BufferedSourceWorker(
            source,
            buffer,
            name=f"buffer-{sanitize_source_id(camera_config.id)}",
        )
        worker.start()
        try:
            return await self._run_live_detection_loop(
                camera_config,
                buffer,
                recorder_client,
                buffer_window_s,
            )
        finally:
            worker.stop()

    async def _run_live_detection_loop(
        self,
        camera_config: CameraConfig,
        buffer: RollingBuffer,
        recorder_client: Any,
        buffer_window_s: float,
    ) -> list[Path]:
        detector = self._new_detector(camera_config)
        classifier = self._new_classifier(camera_config)
        state_machine = self._new_state_machine(camera_config)
        recorder = self.recorder_factory(self.config, recorder_client)
        history = _FrameHistory(buffer_window_s, max_frames=_live_buffer_max_frames(buffer_window_s))
        pending: list[_PendingCandidate] = []
        event_dirs: list[Path] = []
        last_key: tuple[str, int] | None = None
        last_detection_mono = -math.inf
        processed = 0
        sample_interval_s = 1.0 / camera_config.sample_rate_fps
        live_batch_size = max(1, self.config.global_settings.live_detection_batch_size)
        max_batch_wait_s = self.config.global_settings.max_batch_wait_s

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
            results = _detect_frames_batched(
                detector, batch, lock=self._inference_lock
            )
            for batched_frame, detections in zip(batch, results):
                emitted = state_machine.process(batched_frame, detections)
                pending.extend(_PendingCandidate(candidate) for candidate in emitted)
            batch.clear()
            batch_started_mono = None

        while not self._stop_event.is_set() and (
            self.max_live_frames is None or processed < self.max_live_frames
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
                self._record_ready_pending(
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
        pending.extend(_PendingCandidate(candidate) for candidate in state_machine.flush())
        event_dirs.extend(
            self._record_all_pending(
                pending,
                buffer=buffer,
                history=history,
                camera_config=camera_config,
                classifier=classifier,
                recorder=recorder,
            ),
        )
        enforce_retention(self.config.global_settings.dataset_dir, camera_config)
        return event_dirs

    def _record_ready_pending(
        self,
        pending: list[_PendingCandidate],
        *,
        current_wall_ts: datetime,
        buffer: RollingBuffer,
        history: _FrameHistory,
        camera_config: CameraConfig,
        classifier: PottyClassifier,
        recorder: EventRecorder,
    ) -> list[Path]:
        ready: list[_PendingCandidate] = []
        waiting: list[_PendingCandidate] = []
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
                path := self._record_candidate(
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

    def _record_all_pending(
        self,
        pending: list[_PendingCandidate],
        *,
        buffer: RollingBuffer,
        history: _FrameHistory,
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
                path := self._record_candidate(
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

    def _record_candidate(
        self,
        pending: _PendingCandidate,
        *,
        buffer: RollingBuffer,
        history: _FrameHistory,
        camera_config: CameraConfig,
        classifier: PottyClassifier,
        recorder: EventRecorder,
    ) -> Path | None:
        candidate = pending.candidate
        frames = self._assemble_window(buffer, history, candidate, camera_config)
        if not frames:
            LOGGER.warning(
                "Skipping empty event window for camera %s at %s",
                sanitize_source_id(camera_config.id),
                candidate.start_ts.isoformat(),
            )
            return None

        classifier_result: ClassifierResult | None = None
        primary_track = _primary_track(candidate)
        if primary_track is not None:
            classifier_result = classifier.classify(primary_track, frames)

        event_dir = recorder.record(
            candidate,
            frames,
            camera_config,
            classifier_result=classifier_result,
            protect_meta=pending.protect_meta,
        )
        LOGGER.info("Recorded event %s", event_dir)
        return event_dir

    def _assemble_window(
        self,
        buffer: RollingBuffer,
        history: _FrameHistory,
        candidate: PottyCandidate,
        camera_config: CameraConfig,
    ) -> list[Frame]:
        frames = EventRecorder.assemble_window(buffer, candidate, camera_config)
        start_wall = candidate.start_ts - timedelta(seconds=camera_config.pre_roll_s)
        end_wall = candidate.end_ts + timedelta(seconds=camera_config.post_roll_s)
        fallback = history.by_wall(start_wall, end_wall)
        if len(fallback) > len(frames):
            frames = fallback

        mono_bounds = _candidate_mono_bounds(candidate, camera_config)
        if mono_bounds is not None:
            mono_fallback = history.by_mono(*mono_bounds)
            if len(mono_fallback) > len(frames):
                frames = mono_fallback
        return _dedupe_frames(frames)

    def _new_detector(self, camera_config: CameraConfig) -> Detector:
        return _call_camera_factory(self.detector_factory, camera_config, self.config)

    def _new_classifier(self, camera_config: CameraConfig) -> PottyClassifier:
        return _call_camera_factory(self.classifier_factory, camera_config, self.config)

    def _new_state_machine(self, camera_config: CameraConfig) -> PottyEventDetector:
        state_machine = self.state_machine_factory(camera_config)
        pose_gate = self._new_pose_gate(camera_config)
        if pose_gate is not None and hasattr(state_machine, "pose_gate"):
            state_machine.pose_gate = pose_gate
        return state_machine

    def _new_pose_gate(self, _camera_config: CameraConfig) -> PoseGate | None:
        pose_cfg = self.config.pose
        if not (pose_cfg.enabled and pose_cfg.enable_pose_gate):
            return None
        estimator = self._pose_estimator
        if estimator is None:
            return None
        lock = self._inference_lock

        def _estimate(frame: Frame, detection: Detection) -> PoseKeypoints | None:
            with lock:
                return estimator.estimate(
                    frame.bgr,
                    detection.bbox,
                    frame_idx=detection.frame_idx,
                    mono_ts=detection.mono_ts,
                    wall_ts=detection.wall_ts,
                    source_id=frame.source_id,
                )

        return PoseGate(
            _estimate,
            min_keypoint_conf=pose_cfg.min_keypoint_conf,
            min_required_frames=pose_cfg.min_required_frames,
            min_pose_coverage=pose_cfg.min_pose_coverage,
            min_torso_keypoints=pose_cfg.min_torso_keypoints,
        )

    def _selected_cameras(self, camera_ids: Sequence[str] | None) -> list[CameraConfig]:
        requested = set(camera_ids or [])
        selected = [camera for camera in self.config.cameras if camera.enabled]
        if requested:
            selected = [camera for camera in selected if camera.id in requested]
            missing = requested - {camera.id for camera in selected}
            for camera_id in sorted(missing):
                LOGGER.warning("Requested camera %s was not found or is disabled", sanitize_source_id(camera_id))
        return selected

    @staticmethod
    def _default_detector_factory(camera_config: CameraConfig, config: Config) -> Detector:
        return DogDetector(
            model_name=config.global_settings.model_name,
            long_edge=config.global_settings.inference_long_edge_px,
            conf_threshold=camera_config.detection_conf_threshold,
            device=config.global_settings.device,
            alias_classes=config.global_settings.dog_alias_classes,
            alias_nms_iou=config.global_settings.dog_alias_nms_iou,
        )

    def _default_classifier_factory(
        self,
        _camera_config: CameraConfig,
        config: Config,
    ) -> PottyClassifier:
        heuristic = HeuristicPottyClassifier()
        if (
            config.pose.enabled
            and config.pose.enable_pose_classifier
            and self._pose_estimator is not None
        ):
            return PosePottyClassifier(
                self._pose_estimator,
                config.pose,
                heuristic,
                inference_lock=self._inference_lock,
            )
        return heuristic

    @staticmethod
    def _default_file_source_factory(camera_config: CameraConfig) -> VideoSource:
        assert camera_config.input.path is not None
        return FileSource(camera_config.input.path)

    @staticmethod
    def _default_recorder_factory(config: Config, protect_client: Any | None) -> EventRecorder:
        return EventRecorder(config, protect_client=protect_client)

def run_pipeline(
    config: Config,
    camera_ids: Sequence[str] | None = None,
    *,
    detector_factory: DetectorFactory | None = None,
    classifier_factory: ClassifierFactory | None = None,
    file_source_factory: FileSourceFactory | None = None,
    state_machine_factory: StateMachineFactory | None = None,
    recorder_factory: RecorderFactory | None = None,
    rtsp_source_factory: RTSPSourceFactory | None = None,
    max_live_frames: int | None = None,
    max_workers: int | None = None,
    continue_on_camera_error: bool = True,
) -> list[Path]:
    """Run the configured pipeline and return dataset event directories created.

    When multiple cameras are selected they run concurrently (one worker thread
    each), so several live cameras are monitored simultaneously. ``max_workers``
    caps the thread pool (default: ``min(cameras, 4)``); GPU inference is
    serialized internally for MPS safety.
    """

    return PottyPipeline(
        config,
        detector_factory=detector_factory,
        classifier_factory=classifier_factory,
        file_source_factory=file_source_factory,
        state_machine_factory=state_machine_factory,
        recorder_factory=recorder_factory,
        rtsp_source_factory=rtsp_source_factory,
        max_live_frames=max_live_frames,
        max_workers=max_workers,
        continue_on_camera_error=continue_on_camera_error,
    ).run(camera_ids)
