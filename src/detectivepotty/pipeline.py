"""End-to-end potty detection pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import logging
from pathlib import Path
import threading
from typing import Any

from detectivepotty.classify.base import PottyClassifier
from detectivepotty.classify.heuristic import HeuristicPottyClassifier
from detectivepotty.classify.pose import PosePottyClassifier
from detectivepotty.config import CameraConfig, Config
from detectivepotty.detect.yolo import DogDetector
from detectivepotty.events import Detection
from detectivepotty.pipeline_file import process_file_camera as run_file_camera
from detectivepotty.pipeline_live import run_live_camera_async
from detectivepotty.pipeline_runtime import (
    ClassifierFactory,
    Detector,
    DetectorFactory,
    FileSourceFactory,
    RecorderFactory,
    RTSPSourceFactory,
    StateMachineFactory,
    call_camera_factory as _call_camera_factory,
    is_live_kind as _is_live_kind,
    is_valid_rtsp_url as _is_valid_rtsp_url,
    redact_url as _redact_url,
)
from detectivepotty.pose.base import PoseEstimator
from detectivepotty.pose.factory import build_pose_estimator
from detectivepotty.pose.gate import PoseGate
from detectivepotty.pose.keypoints import PoseKeypoints
from detectivepotty.potty_event import PottyEventDetector
from detectivepotty.recording.dataset import camera_dataset_dir
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.sources.base import Frame, VideoSource, sanitize_source_id
from detectivepotty.sources.file import FileSource

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
        return run_file_camera(
            camera_config,
            config=self.config,
            new_detector=self._new_detector,
            new_classifier=self._new_classifier,
            new_state_machine=self._new_state_machine,
            recorder_factory=self.recorder_factory,
            file_source_factory=self.file_source_factory,
            inference_lock=self._inference_lock,
            stop_event=self._stop_event,
        )

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
        return await run_live_camera_async(
            camera_config,
            url,
            recorder_client,
            config=self.config,
            new_detector=self._new_detector,
            new_classifier=self._new_classifier,
            new_state_machine=self._new_state_machine,
            recorder_factory=self.recorder_factory,
            make_live_source=self._make_live_source,
            inference_lock=self._inference_lock,
            stop_event=self._stop_event,
            max_live_frames=self.max_live_frames,
        )

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
