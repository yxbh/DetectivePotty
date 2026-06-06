from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock

import numpy as np

from detectivepotty.classify.base import ClassifierResult
from detectivepotty.config import CameraConfig, Config, GlobalSettings
from detectivepotty.events import ClassifierGuess, Detection, Track, TriggerReason
from detectivepotty.geometry import BBox
from detectivepotty.potty_event import PottyCandidate, PottyLifecycle
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 6, 6, 9, 10, 47, tzinfo=timezone.utc)


def camera_config(**overrides: object) -> CameraConfig:
    values = {
        "id": "cam-1",
        "name": "Backyard Grass",
        "pre_roll_s": 2.0,
        "post_roll_s": 3.0,
        "sample_rate_fps": 1.0,
    }
    values.update(overrides)
    return CameraConfig(**values)


def config(tmp_path, camera: CameraConfig) -> Config:
    return Config(
        global_settings=GlobalSettings(
            dataset_dir=tmp_path,
            model_name="test-model.pt",
        ),
        cameras=[camera],
    )


def make_frame(frame_idx: int) -> Frame:
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[8:32, 18:42] = (30, 150, 240)
    return Frame(
        bgr=bgr,
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
        source_id="rtsp://user:pass@cam.local/stream?token=secret&keep=1",
    )


def make_detection(frame_idx: int, bbox: BBox | None = None) -> Detection:
    return Detection(
        bbox=bbox or BBox(18, 8, 42, 32),
        confidence=0.88,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
    )


def candidate() -> PottyCandidate:
    detections = [make_detection(1), make_detection(2, BBox(17, 10, 44, 30))]
    track = Track(track_id="dog-1", detections=detections)
    return PottyCandidate(
        camera_id="cam-1",
        primary_track_id="dog-1",
        start_ts=BASE_TS + timedelta(seconds=1),
        end_ts=BASE_TS + timedelta(seconds=2),
        tracks=[track],
        detections=detections,
        trigger_reason=TriggerReason.YOLO,
        multi_dog=False,
        ambiguous=False,
        lifecycle=PottyLifecycle.EMITTED,
        stationary_duration_s=2.5,
        squat_metric=0.42,
        posture_summary={"height_drop": 0.42},
        near_miss=False,
        confidence=0.77,
    )


def test_event_recorder_writes_dataset_event_and_metadata(tmp_path) -> None:
    camera = camera_config()
    recorder = EventRecorder(config(tmp_path, camera), git_commit="abc123", model_version="v0")
    frames = [make_frame(idx) for idx in range(5)]
    potty_candidate = candidate()

    target = recorder.record(
        potty_candidate,
        frames,
        camera,
        classifier_result=ClassifierResult(ClassifierGuess.PEE, 0.42),
        protect_meta={
            "protect_event_id": "protect-1",
            "smartdetect_score": 0.91,
            "smartdetect_bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "detection_ts": potty_candidate.start_ts,
            "notification_ts": potty_candidate.start_ts + timedelta(seconds=1.25),
        },
    )

    assert (target / "clip.mp4").exists()
    assert (target / "frames").is_dir()
    assert (target / "crops").is_dir()
    metadata_path = target / "metadata.json"
    assert metadata_path.exists()

    raw_json = metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(raw_json)
    assert metadata["camera_id"] == "cam-1"
    assert metadata["camera_name"] == "Backyard Grass"
    assert metadata["trigger_reason"] == "yolo"
    assert metadata["pre_roll_s"] == 2.0
    assert metadata["post_roll_s"] == 3.0
    assert metadata["multi_dog"] is False
    assert metadata["ambiguous"] is False
    assert metadata["label_status"] == "unlabeled"
    assert metadata["classifier_guess"] == "pee"
    assert metadata["classifier_confidence"] == 0.42
    assert metadata["protect_event_id"] == "protect-1"
    assert metadata["trigger_latency_s"] == 1.25
    assert len(metadata["detections"]) == 2
    assert len(metadata["tracks"]) == 1
    assert len(metadata["frame_records"]) == 5
    assert len(metadata["crop_boxes"]) == 2
    assert metadata["extra"]["primary_track_id"] == "dog-1"
    assert metadata["extra"]["stationary_duration_s"] == 2.5
    assert metadata["extra"]["squat_metric"] == 0.42
    assert metadata["extra"]["posture_summary"] == {"height_drop": 0.42}
    assert metadata["extra"]["near_miss"] is False

    for secret in ("user", "pass", "token", "secret"):
        assert secret not in raw_json


def test_maybe_download_protect_recording_awaits_client_with_preroll(tmp_path) -> None:
    camera = camera_config()
    potty_candidate = candidate()
    event_path = tmp_path / "event"
    expected_path = event_path / "protect_recording.mp4"
    protect_client = type("Client", (), {})()
    protect_client.download_recording = AsyncMock(return_value=expected_path)
    recorder = EventRecorder(
        config(tmp_path, camera),
        protect_client=protect_client,
        git_commit="abc123",
    )

    result = asyncio.run(
        recorder.maybe_download_protect_recording(potty_candidate, camera, event_path),
    )

    assert result == expected_path
    protect_client.download_recording.assert_awaited_once()
    args = protect_client.download_recording.await_args.args
    assert args[0] == "cam-1"
    assert args[1] == potty_candidate.start_ts - timedelta(seconds=2.0)
    assert args[2] == potty_candidate.end_ts + timedelta(seconds=3.0)
    assert args[3] == expected_path


def test_maybe_download_protect_recording_is_best_effort(tmp_path) -> None:
    camera = camera_config()
    protect_client = type("Client", (), {})()
    protect_client.download_recording = AsyncMock(side_effect=RuntimeError("boom"))
    recorder = EventRecorder(
        config(tmp_path, camera),
        protect_client=protect_client,
        git_commit="abc123",
    )

    result = asyncio.run(
        recorder.maybe_download_protect_recording(candidate(), camera, tmp_path / "event"),
    )

    assert result is None


def test_assemble_window_uses_pre_and_post_roll() -> None:
    camera = camera_config()
    potty_candidate = candidate()
    frames = [make_frame(idx) for idx in range(7)]

    class Buffer:
        def get_window(self, start, end):
            self.start = start
            self.end = end
            return frames

    buffer = Buffer()

    assert EventRecorder.assemble_window(buffer, potty_candidate, camera) == frames
    assert buffer.start == potty_candidate.start_ts - timedelta(seconds=2.0)
    assert buffer.end == potty_candidate.end_ts + timedelta(seconds=3.0)
