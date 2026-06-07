"""End-to-end potty detection pipeline orchestration."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
import math
from pathlib import Path
import threading
from typing import Any, Protocol

from detectivepotty.classify.base import ClassifierResult, PottyClassifier
from detectivepotty.classify.heuristic import HeuristicPottyClassifier
from detectivepotty.classify.pose import PosePottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.detect.yolo import DogDetector
from detectivepotty.events import Detection, Track
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

# Assumed worst-case decode rate used to bound per-camera live buffer memory.
_LIVE_ASSUMED_MAX_FPS = 30.0


class Detector(Protocol):
    def detect(
        self,
        frame_bgr: Any,
        *,
        frame_idx: int,
        mono_ts: float,
        wall_ts: datetime,
    ) -> list[Detection]: ...


DetectorFactory = Callable[..., Detector]
ClassifierFactory = Callable[..., PottyClassifier]
FileSourceFactory = Callable[[CameraConfig], VideoSource]
StateMachineFactory = Callable[[CameraConfig], PottyEventDetector]
RecorderFactory = Callable[[Config, Any | None], EventRecorder]


@dataclass(slots=True)
class _PendingCandidate:
    candidate: PottyCandidate
    protect_meta: dict[str, Any] | None = None


class _FrameHistory:
    def __init__(self, window_s: float, *, max_frames: int | None = None) -> None:
        self.window_s = max(0.0, window_s)
        self.max_frames = max_frames
        self._frames: deque[Frame] = deque()

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)
        cutoff = frame.mono_ts - self.window_s
        while self._frames and self._frames[0].mono_ts < cutoff:
            self._frames.popleft()
        if self.max_frames is not None:
            while len(self._frames) > self.max_frames:
                self._frames.popleft()

    def snapshot(self) -> list[Frame]:
        return list(self._frames)

    def by_wall(self, start: datetime, end: datetime) -> list[Frame]:
        return [frame for frame in self._frames if start <= frame.wall_ts <= end]

    def by_mono(self, start: float, end: float) -> list[Frame]:
        return [frame for frame in self._frames if start <= frame.mono_ts <= end]


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
        self._configure_logging(config.global_settings.log_level)

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
        # Submit live (protect) cameras first so their never-ending loops claim
        # worker slots immediately instead of waiting behind finite file cameras.
        order = sorted(
            range(len(selected)),
            key=lambda i: 0 if selected[i].input.kind == "protect" else 1,
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
        live = sum(1 for camera in selected if camera.input.kind == "protect")
        has_file = any(camera.input.kind != "protect" for camera in selected)
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

            while True:
                raw_frame = source.read()
                if raw_frame is None:
                    break
                if base_mono is None:
                    base_mono = raw_frame.mono_ts
                frame = _retimestamp_file_frame(raw_frame, base_mono, source_fps)
                buffer.append(frame)
                history.append(frame)

                if frame.frame_idx % sample_every == 0:
                    with self._inference_lock:
                        detections = detector.detect(
                            frame.bgr,
                            frame_idx=frame.frame_idx,
                            mono_ts=frame.mono_ts,
                            wall_ts=frame.wall_ts,
                        )
                    emitted = state_machine.process(frame, detections)
                    pending.extend(_PendingCandidate(candidate) for candidate in emitted)

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
        if not _protect_configured(self.config):
            LOGGER.warning(
                "Protect is not configured; skipping camera %s",
                sanitize_source_id(camera_config.id),
            )
            return []
        try:
            return asyncio.run(self._process_protect_camera_async(camera_config))
        except Exception as exc:
            LOGGER.warning(
                "Protect camera %s skipped: %s",
                sanitize_source_id(camera_config.id),
                exc,
            )
            return []

    async def _process_protect_camera_async(self, camera_config: CameraConfig) -> list[Path]:
        try:
            from detectivepotty.protect.client import ProtectClient
            from detectivepotty.sources.rtsp import RTSPSource
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

            source = RTSPSource(url)
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
                    protect_client,
                    buffer_window_s,
                )
            finally:
                worker.stop()
        finally:
            await protect_client.close()

    async def _run_live_detection_loop(
        self,
        camera_config: CameraConfig,
        buffer: RollingBuffer,
        protect_client: Any,
        buffer_window_s: float,
    ) -> list[Path]:
        detector = self._new_detector(camera_config)
        classifier = self._new_classifier(camera_config)
        state_machine = self._new_state_machine(camera_config)
        recorder = self.recorder_factory(self.config, protect_client)
        history = _FrameHistory(buffer_window_s, max_frames=_live_buffer_max_frames(buffer_window_s))
        pending: list[_PendingCandidate] = []
        event_dirs: list[Path] = []
        last_key: tuple[str, int] | None = None
        last_detection_mono = -math.inf
        processed = 0
        sample_interval_s = 1.0 / camera_config.sample_rate_fps

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
                with self._inference_lock:
                    detections = detector.detect(
                        frame.bgr,
                        frame_idx=frame.frame_idx,
                        mono_ts=frame.mono_ts,
                        wall_ts=frame.wall_ts,
                    )
                emitted = state_machine.process(frame, detections)
                pending.extend(_PendingCandidate(candidate) for candidate in emitted)

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

    @staticmethod
    def _configure_logging(level_name: str) -> None:
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        LOGGER.setLevel(level)


def run_pipeline(
    config: Config,
    camera_ids: Sequence[str] | None = None,
    *,
    detector_factory: DetectorFactory | None = None,
    classifier_factory: ClassifierFactory | None = None,
    file_source_factory: FileSourceFactory | None = None,
    state_machine_factory: StateMachineFactory | None = None,
    recorder_factory: RecorderFactory | None = None,
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
        max_live_frames=max_live_frames,
        max_workers=max_workers,
        continue_on_camera_error=continue_on_camera_error,
    ).run(camera_ids)


def _call_camera_factory(factory: Callable[..., Any], camera_config: CameraConfig, config: Config) -> Any:
    try:
        return factory(camera_config, config)
    except TypeError as original_exc:
        try:
            return factory(camera_config)
        except TypeError:
            raise original_exc



def _buffer_window_s(camera_config: CameraConfig) -> float:
    return max(
        1.0,
        camera_config.pre_roll_s
        + camera_config.stationary_threshold_s
        + camera_config.event_duration_s
        + camera_config.post_roll_s
        + 2.0,
    )


def _history_max_frames(source_fps: float, window_s: float) -> int:
    return max(1, math.ceil(source_fps * window_s) + 2)


def _live_buffer_max_frames(window_s: float) -> int:
    # Bound per-camera warm-buffer memory; without a cap N concurrent live
    # cameras decoding high-fps streams could exhaust memory.
    return max(1, math.ceil(window_s * _LIVE_ASSUMED_MAX_FPS) + 2)


def _source_fps(source: VideoSource, camera_config: CameraConfig) -> float:
    fps = source.fps or camera_config.sample_rate_fps
    return fps if fps > 0 else camera_config.sample_rate_fps


def _sample_every(source_fps: float, sample_rate_fps: float) -> int:
    if source_fps <= 0 or sample_rate_fps <= 0:
        return 1
    return max(1, round(source_fps / sample_rate_fps))


def _retimestamp_file_frame(frame: Frame, base_mono: float, fps: float) -> Frame:
    # File decoding is faster than real time; use the file timeline for state durations.
    synthetic_mono = base_mono + frame.frame_idx / fps
    return Frame(
        bgr=frame.bgr,
        frame_idx=frame.frame_idx,
        mono_ts=synthetic_mono,
        wall_ts=frame.wall_ts,
        source_id=frame.source_id,
    )


def _primary_track(candidate: PottyCandidate) -> Track | None:
    for track in candidate.tracks:
        if track.track_id == candidate.primary_track_id:
            return track
    return candidate.tracks[0] if candidate.tracks else None


def _candidate_mono_bounds(
    candidate: PottyCandidate,
    camera_config: CameraConfig,
) -> tuple[float, float] | None:
    mono_values = [detection.mono_ts for detection in _candidate_detections(candidate)]
    if not mono_values:
        return None
    start = min(mono_values) - camera_config.pre_roll_s
    end = max(mono_values) + camera_config.post_roll_s + camera_config.event_duration_s
    return start, end


def _candidate_detections(candidate: PottyCandidate) -> Iterable[Detection]:
    yield from candidate.detections
    for track in candidate.tracks:
        yield from track.detections


def _dedupe_frames(frames: Sequence[Frame]) -> list[Frame]:
    seen: set[tuple[str, int]] = set()
    unique: list[Frame] = []
    for frame in frames:
        key = (frame.source_id, frame.frame_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
    return unique


def _protect_configured(config: Config) -> bool:
    has_host = bool(config.protect.nvr_host)
    has_api_key = bool(config.resolve_secret("api_key"))
    has_userpass = bool(config.resolve_secret("username") and config.resolve_secret("password"))
    return has_host and (has_api_key or has_userpass)
