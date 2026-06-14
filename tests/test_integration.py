from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np

from detectivepotty.config import CameraConfig, CameraInputConfig, Config, GlobalSettings
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pipeline import run_pipeline
from detectivepotty.web import create_app


def write_synthetic_video(path: Path, *, frames: int = 50, fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    try:
        for frame_idx in range(frames):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:, :, 1] = 30
            bbox = bbox_for_frame(frame_idx)
            if bbox is not None:
                x1, y1, x2, y2 = bbox.to_int_tuple()
                cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 240), -1)
            writer.write(image)
    finally:
        writer.release()
    assert path.exists() and path.stat().st_size > 0


def bbox_for_frame(frame_idx: int) -> BBox | None:
    if frame_idx < 5:
        return BBox(44 + frame_idx, 20, 84 + frame_idx, 100)
    if frame_idx < 10:
        return BBox(50, 20, 90, 100)
    if frame_idx < 26:
        return BBox(48, 50, 98, 90)
    return None


class FakeDogDetector:
    device = "cpu"
    model_name = "fake.pt"
    last_inference = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_idx: int,
        mono_ts: float,
        wall_ts: datetime,
    ) -> list[Detection]:
        del frame_bgr
        bbox = bbox_for_frame(frame_idx)
        if bbox is None:
            return []
        return [
            Detection(
                bbox=bbox,
                confidence=0.95,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=mono_ts,
                wall_ts=wall_ts,
            ),
        ]


def fake_detector_factory(_camera_config: CameraConfig) -> FakeDogDetector:
    return FakeDogDetector()


def make_config(dataset_dir: Path, video_path: Path) -> Config:
    return Config(
        global_settings=GlobalSettings(
            dataset_dir=dataset_dir,
            model_name="fake.pt",
            device="cpu",
        ),
        cameras=[
            CameraConfig(
                id="synthetic-yard",
                name="Synthetic Yard",
                enabled=True,
                input=CameraInputConfig(
                    kind="file",
                    path=video_path,
                    source_id="file:synthetic-yard",
                ),
                substream_choice="high",
                animal_supported=False,
                detection_conf_threshold=0.1,
                event_duration_s=0.5,
                stationary_threshold_s=0.5,
                dwell_trigger_s=1.0,
                sample_rate_fps=10.0,
                pre_roll_s=0.2,
                post_roll_s=0.3,
                retention_days=30,
            ),
        ],
    )


def test_pipeline_records_event_and_web_app_labels_it(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic-potty.mp4"
    dataset_dir = tmp_path / "dataset"
    write_synthetic_video(video_path)
    config = make_config(dataset_dir, video_path)

    event_dirs = run_pipeline(config, detector_factory=fake_detector_factory)

    assert len(event_dirs) == 1
    event_dir = event_dirs[0]
    metadata_path = event_dir / "metadata.json"
    assert (event_dir / "clip.mp4").is_file()
    assert sorted((event_dir / "frames").glob("*.jpg"))
    assert sorted((event_dir / "crops").glob("*.jpg"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["camera_id"] == "synthetic-yard"
    assert metadata["trigger_reason"] == "yolo"
    assert metadata["label_status"] == "unlabeled"
    assert metadata["label"] == "unknown"
    event_id = metadata["event_id"]

    client = TestClient(create_app(config))
    list_response = client.get("/api/events")
    assert list_response.status_code == 200
    events = list_response.json()["events"]
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["label_status"] == "unlabeled"

    detail_response = client.get(f"/api/events/{event_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["metadata"]["event_id"] == event_id
    assert detail["metadata"]["label_status"] == "unlabeled"
    assert detail["media"]["clip"] == f"/api/events/{event_id}/media/clip"
    assert detail["media"]["frames"]
    assert detail["media"]["crops"]

    frame_response = client.get(detail["media"]["frames"][0]["url"])
    assert frame_response.status_code == 200
    assert frame_response.content
    crop_response = client.get(detail["media"]["crops"][0]["url"])
    assert crop_response.status_code == 200
    assert crop_response.content

    label_response = client.post(
        f"/api/events/{event_id}/label",
        json={"label": "pee", "label_status": "labeled", "note": "integration test"},
    )
    assert label_response.status_code == 200
    label_summary = label_response.json()
    assert label_summary["label"] == "pee"
    assert label_summary["label_status"] == "labeled"

    updated = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert updated["event_id"] == event_id
    assert updated["label"] == "pee"
    assert updated["label_status"] == "labeled"
    assert updated["extra"]["label_note"] == "integration test"
    assert "labeled_at" in updated["extra"]
