from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np

from detectivepotty.config import CameraConfig, CameraInputConfig, Config, GlobalSettings
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pipeline import run_pipeline
from detectivepotty.sources.base import Frame, VideoSource

def make_config(dataset_dir: Path, video_path: Path | None, *, enabled: bool = True) -> Config:
    return Config(
        global_settings=GlobalSettings(dataset_dir=dataset_dir, model_name="fake.pt", device="cpu"),
        cameras=[
            CameraConfig(
                id="cam-1",
                name="Backyard",
                enabled=enabled,
                input=CameraInputConfig(kind="file", path=video_path),
                detection_conf_threshold=0.25,
                event_duration_s=1.0,
                stationary_threshold_s=1.0,
                dwell_trigger_s=2.0,
                sample_rate_fps=1.0,
                pre_roll_s=1.0,
                post_roll_s=1.0,
                retention_days=30,
            ),
        ],
    )


def write_synthetic_video(path: Path, *, frames: int = 6, fps: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    try:
        for frame_idx in range(frames):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:, :, 1] = 20
            x1 = 40 if frame_idx < 2 else 25
            y1 = 20 if frame_idx < 2 else 35
            x2 = 80 if frame_idx < 2 else 95
            y2 = 100 if frame_idx < 2 else 85
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 240), -1)
            writer.write(image)
    finally:
        writer.release()
    assert path.exists() and path.stat().st_size > 0


class FakeDogDetector:
    device = "cpu"
    model_name = "fake.pt"
    last_inference = None

    def detect(self, frame_bgr, *, frame_idx: int, mono_ts: float, wall_ts):  # noqa: ANN001
        del frame_bgr
        if frame_idx < 2:
            bbox = BBox(40, 20, 80, 100)
        elif frame_idx <= 4:
            bbox = BBox(25, 35, 95, 85)
        else:
            return []
        return [
            Detection(
                bbox=bbox,
                confidence=0.9,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=mono_ts,
                wall_ts=wall_ts,
            ),
        ]


def fake_detector_factory(_camera_config):  # noqa: ANN001
    return FakeDogDetector()


def test_run_pipeline_records_file_event_with_injected_detector(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    dataset_dir = tmp_path / "dataset"
    write_synthetic_video(video_path)
    config = make_config(dataset_dir, video_path)

    event_dirs = run_pipeline(config, detector_factory=fake_detector_factory)

    assert len(event_dirs) == 1
    event_dir = event_dirs[0]
    assert event_dir.is_dir()
    clip_path = event_dir / "clip.mp4"
    metadata_path = event_dir / "metadata.json"
    assert clip_path.exists() and clip_path.stat().st_size > 0
    assert metadata_path.exists()
    assert sorted((event_dir / "frames").glob("*.jpg"))
    assert sorted((event_dir / "crops").glob("*.jpg"))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["camera_id"] == "cam-1"
    assert metadata["camera_name"] == "Backyard"
    assert metadata["trigger_reason"] == "yolo"
    assert metadata["classifier_guess"] == "pee"
    assert metadata["extra"]["primary_track_id"] == "1"
    assert len(metadata["frame_records"]) >= 4
    assert len(metadata["crop_boxes"]) >= 3


def test_run_pipeline_no_enabled_cameras_returns_cleanly(tmp_path) -> None:
    config = make_config(tmp_path / "dataset", None, enabled=False)

    assert run_pipeline(config, detector_factory=fake_detector_factory) == []


def make_multi_camera_config(dataset_dir: Path, cameras: list[tuple[str, str, Path]]) -> Config:
    return Config(
        global_settings=GlobalSettings(dataset_dir=dataset_dir, model_name="fake.pt", device="cpu"),
        cameras=[
            CameraConfig(
                id=cam_id,
                name=cam_name,
                enabled=True,
                input=CameraInputConfig(kind="file", path=video_path),
                detection_conf_threshold=0.25,
                event_duration_s=1.0,
                stationary_threshold_s=1.0,
                dwell_trigger_s=2.0,
                sample_rate_fps=1.0,
                pre_roll_s=1.0,
                post_roll_s=1.0,
                retention_days=30,
            )
            for cam_id, cam_name, video_path in cameras
        ],
    )


def test_run_pipeline_processes_multiple_cameras_concurrently(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    cameras = []
    for idx in range(3):
        video_path = tmp_path / f"sample-{idx}.mp4"
        write_synthetic_video(video_path)
        cameras.append((f"cam-{idx}", f"Camera {idx}", video_path))
    config = make_multi_camera_config(dataset_dir, cameras)

    event_dirs = run_pipeline(config, detector_factory=fake_detector_factory)

    # One event per camera, results flattened in selected-camera order.
    assert len(event_dirs) == 3
    camera_components = [event_dir.parent.parent.parent.name for event_dir in event_dirs]
    assert camera_components == ["Camera_0", "Camera_1", "Camera_2"]


def test_run_pipeline_isolates_failing_camera(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    cameras = []
    for idx in range(2):
        video_path = tmp_path / f"sample-{idx}.mp4"
        write_synthetic_video(video_path)
        cameras.append((f"cam-{idx}", f"Camera {idx}", video_path))
    config = make_multi_camera_config(dataset_dir, cameras)

    def flaky_detector_factory(camera_config):  # noqa: ANN001
        if camera_config.id == "cam-0":
            raise RuntimeError("boom")
        return FakeDogDetector()

    # Default continue_on_camera_error=True: cam-0 fails, cam-1 still records.
    event_dirs = run_pipeline(config, detector_factory=flaky_detector_factory)

    assert len(event_dirs) == 1
    assert event_dirs[0].parent.parent.parent.name == "Camera_1"


def test_run_pipeline_max_workers_one_forces_sequential(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    cameras = []
    for idx in range(2):
        video_path = tmp_path / f"sample-{idx}.mp4"
        write_synthetic_video(video_path)
        cameras.append((f"cam-{idx}", f"Camera {idx}", video_path))
    config = make_multi_camera_config(dataset_dir, cameras)

    event_dirs = run_pipeline(config, detector_factory=fake_detector_factory, max_workers=1)

    assert len(event_dirs) == 2
    camera_components = [event_dir.parent.parent.parent.name for event_dir in event_dirs]
    assert camera_components == ["Camera_0", "Camera_1"]


def _make_camera(cam_id: str, kind: str) -> CameraConfig:
    return CameraConfig(id=cam_id, name=cam_id, input=CameraInputConfig(kind=kind))


def test_default_classifier_factory_uses_pose_when_enabled() -> None:
    from detectivepotty.classify.heuristic import HeuristicPottyClassifier
    from detectivepotty.classify.pose import PosePottyClassifier
    from detectivepotty.config import PoseConfig
    from detectivepotty.pipeline import PottyPipeline

    camera = _make_camera("a", "file")
    enabled = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        pose=PoseConfig(enabled=True, backend="mock", enable_pose_classifier=True),
        cameras=[camera],
    )
    pipeline = PottyPipeline(enabled, detector_factory=fake_detector_factory)
    assert isinstance(pipeline._new_classifier(camera), PosePottyClassifier)

    # Pose enabled but classifier consumer gate off -> heuristic.
    gate_off = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        pose=PoseConfig(enabled=True, backend="mock", enable_pose_classifier=False),
        cameras=[camera],
    )
    pipeline_off = PottyPipeline(gate_off, detector_factory=fake_detector_factory)
    assert isinstance(pipeline_off._new_classifier(camera), HeuristicPottyClassifier)

    # Pose disabled entirely -> heuristic and no estimator built.
    disabled = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        cameras=[camera],
    )
    pipeline_disabled = PottyPipeline(disabled, detector_factory=fake_detector_factory)
    assert pipeline_disabled._pose_estimator is None
    assert isinstance(pipeline_disabled._new_classifier(camera), HeuristicPottyClassifier)



def test_resolve_max_workers_defaults_to_one_thread_per_camera() -> None:
    from detectivepotty.pipeline import PottyPipeline

    config = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        cameras=[_make_camera("a", "protect"), _make_camera("b", "protect"), _make_camera("c", "file")],
    )
    pipeline = PottyPipeline(config, detector_factory=fake_detector_factory)

    assert pipeline._resolve_max_workers(config.cameras) == 3


def test_resolve_max_workers_raises_floor_for_live_cameras() -> None:
    from detectivepotty.pipeline import PottyPipeline

    config = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        cameras=[_make_camera("a", "protect"), _make_camera("b", "protect"), _make_camera("c", "protect")],
    )
    # Three live cameras each need a dedicated thread; an explicit cap of 1 is
    # raised back up so none are starved.
    pipeline = PottyPipeline(config, detector_factory=fake_detector_factory, max_workers=1)

    assert pipeline._resolve_max_workers(config.cameras) == 3


def test_resolve_max_workers_reserves_extra_slot_for_file_cameras() -> None:
    from detectivepotty.pipeline import PottyPipeline

    config = Config(
        global_settings=GlobalSettings(model_name="fake.pt", device="cpu"),
        cameras=[_make_camera("a", "protect"), _make_camera("b", "file"), _make_camera("c", "file")],
    )
    # 1 live + file cameras: explicit cap of 1 is raised to live(1)+1 shared file slot.
    pipeline = PottyPipeline(config, detector_factory=fake_detector_factory, max_workers=1)

    assert pipeline._resolve_max_workers(config.cameras) == 2

class _NoDetector:
    device = "cpu"
    model_name = "fake.pt"

    def detect(self, frame_bgr, *, frame_idx, mono_ts, wall_ts):  # noqa: ANN001
        del frame_bgr, frame_idx, mono_ts, wall_ts
        return []


def _no_detector_factory(_camera_config):  # noqa: ANN001
    return _NoDetector()


class _FakeLiveSource(VideoSource):
    """Endless synthetic live source for routing tests (no network/cv2)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._idx = 0

    def open(self):
        return self

    def read(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = Frame(
            bgr=image,
            frame_idx=self._idx,
            mono_ts=float(self._idx),
            wall_ts=datetime.now(timezone.utc),
            source_id="pool-rtsp",
        )
        self._idx += 1
        time.sleep(0.001)
        return frame

    def close(self) -> None:
        return None

    @property
    def fps(self):
        return 10.0

    @property
    def resolution(self):
        return (160, 120)

    @property
    def is_live(self):
        return True


def _rtsp_config(dataset_dir: Path, url_env: str) -> Config:
    return Config(
        global_settings=GlobalSettings(dataset_dir=dataset_dir, model_name="fake.pt", device="cpu"),
        cameras=[
            CameraConfig(
                id="cam-pool",
                name="Pool",
                input=CameraInputConfig(kind="rtsp", url_env=url_env),
                sample_rate_fps=1.0,
            ),
        ],
    )


def test_rtsp_camera_routes_to_live_path_with_resolved_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("POOL_RTSP_URL", "rtsp://user:pass@192.168.1.37:554/cam")
    captured: dict[str, str] = {}

    def factory(url: str) -> VideoSource:
        captured["url"] = url
        return _FakeLiveSource(url)

    config = _rtsp_config(tmp_path / "dataset", "POOL_RTSP_URL")
    result = run_pipeline(
        config,
        detector_factory=_no_detector_factory,
        rtsp_source_factory=factory,
        max_live_frames=2,
    )

    assert result == []
    assert captured["url"] == "rtsp://user:pass@192.168.1.37:554/cam"


def test_rtsp_camera_missing_env_warns_and_skips(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.delenv("POOL_RTSP_URL", raising=False)
    config = _rtsp_config(tmp_path / "dataset", "POOL_RTSP_URL")

    with caplog.at_level("WARNING"):
        result = run_pipeline(config, detector_factory=_no_detector_factory)

    assert result == []
    assert "has no URL" in caplog.text


def test_rtsp_camera_invalid_url_warns_and_skips(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("POOL_RTSP_URL", "http://not-rtsp/stream")
    config = _rtsp_config(tmp_path / "dataset", "POOL_RTSP_URL")

    with caplog.at_level("WARNING"):
        result = run_pipeline(config, detector_factory=_no_detector_factory)

    assert result == []
    assert "invalid URL" in caplog.text


class _StopAfterFirstFrameSource(VideoSource):
    def __init__(self, on_first_read):
        self._on_first_read = on_first_read
        self.read_count = 0

    def open(self):
        return self

    def read(self):
        if self.read_count > 0:
            raise AssertionError("file source read again after stop was requested")
        self.read_count += 1
        frame = Frame(
            bgr=np.zeros((120, 160, 3), dtype=np.uint8),
            frame_idx=0,
            mono_ts=0.0,
            wall_ts=datetime.now(timezone.utc),
            source_id="file://stop-test",
        )
        self._on_first_read()
        return frame

    def close(self) -> None:
        return None

    @property
    def fps(self):
        return 10.0

    @property
    def resolution(self):
        return (160, 120)

    @property
    def is_live(self):
        return False


def test_file_camera_honors_stop_event_between_reads(tmp_path) -> None:
    from detectivepotty.pipeline import PottyPipeline

    video_path = tmp_path / "placeholder.mp4"
    config = make_config(tmp_path / "dataset", video_path)
    pipeline = PottyPipeline(config, detector_factory=_no_detector_factory)
    source = _StopAfterFirstFrameSource(lambda: pipeline._stop_event.set())
    pipeline.file_source_factory = lambda _camera: source

    assert pipeline.process_file_camera(config.cameras[0]) == []
    assert source.read_count == 1
