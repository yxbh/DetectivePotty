"""Offline tests for the in-browser tuner backend and the SPA fallback route.

Everything here is offline: a fake detector and ``MockPoseEstimator`` are
injected, and clips are tiny synthetic mp4s written with ``cv2.VideoWriter`` — no
real YOLO/pose model, GPU, or network is touched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from detectivepotty.config import (
    CameraConfig,
    CameraInputConfig,
    Config,
    GlobalSettings,
    PoseConfig,
)
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pose.base import MockPoseEstimator
from detectivepotty.web import create_app
from detectivepotty.web import tune as tune_mod


def write_clip(path: Path, frames: int = 8, size: tuple[int, int] = (160, 120)) -> Path:
    """Write a tiny synthetic mp4 so VideoCapture has a real file to seek/read."""

    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height)
    )
    assert writer.isOpened(), "could not open VideoWriter (codec missing?)"
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = ((i * 17) % 255, 60, 200)
        writer.write(frame)
    writer.release()
    assert path.is_file()
    return path


class FakeDetector:
    """Returns one fixed dog box regardless of input (no model)."""

    device = "cpu"
    last_inference = None

    def detect(self, frame, frame_idx=0, mono_ts=None, wall_ts=None):
        return [
            Detection(
                bbox=BBox(10.0, 10.0, 80.0, 90.0),
                confidence=0.7,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=0.0,
                wall_ts=wall_ts or datetime.now(timezone.utc),
            )
        ]


def make_config(tmp_path: Path, clip: Path) -> Config:
    camera = CameraConfig(
        id="cam1",
        name="cam1",
        input=CameraInputConfig(kind="file", path=clip),
    )
    return Config(
        global_settings=GlobalSettings(dataset_dir=tmp_path / "dataset"),
        cameras=[camera],
    )


def make_client(
    tmp_path: Path,
    clip: Path,
    *,
    with_pose: bool = True,
) -> TestClient:
    config = make_config(tmp_path, clip)
    kwargs = {"tune_detector": FakeDetector()}
    if with_pose:
        kwargs["tune_pose_estimator"] = MockPoseEstimator()
    return TestClient(create_app(config, **kwargs))


# --- file browser ---------------------------------------------------------


def test_collect_tune_roots_includes_file_camera_dir(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    clip = write_clip(tmp_path / "clips" / "c.mp4")
    config = make_config(tmp_path, clip)
    roots = tune_mod.collect_tune_roots(config)
    assert (tmp_path / "clips").resolve() in roots


def test_tune_files_top_level_lists_roots(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.get("/api/tune/files").json()
    assert body["path"] == ""
    assert body["parent"] is None
    paths = {entry["path"] for entry in body["entries"]}
    assert str(tmp_path.resolve()) in paths


def test_tune_files_lists_videos_and_dirs(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("ignore me")
    client = make_client(tmp_path, clip)
    body = client.get("/api/tune/files", params={"path": str(tmp_path)}).json()
    names = [entry["name"] for entry in body["entries"]]
    kinds = {entry["name"]: entry["kind"] for entry in body["entries"]}
    assert "sub" in names and kinds["sub"] == "dir"
    assert "c.mp4" in names and kinds["c.mp4"] == "video"
    assert "notes.txt" not in names  # non-video files are hidden
    # dirs sort before videos
    assert names.index("sub") < names.index("c.mp4")


def test_tune_files_rejects_traversal(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    assert client.get("/api/tune/files", params={"path": "/etc"}).status_code == 400
    outside = str((tmp_path / ".." / "..").resolve())
    assert client.get("/api/tune/files", params={"path": outside}).status_code == 400


# --- frame endpoint -------------------------------------------------------


def test_tune_frame_returns_image_and_detections(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=8)
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/frame", params={"path": str(clip), "index": 3, "pose": 0}
    ).json()
    assert body["index"] == 3
    assert body["total_frames"] == 8
    assert body["width"] == 160 and body["height"] == 120
    assert body["detection_floor"] == pytest.approx(0.05)
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert len(body["detections"]) == 1
    assert body["detections"][0]["class_name"] == "dog"
    # pose=0 -> no pose payload even though an estimator is available
    assert body["pose"] == []
    assert body["pose_available"] is True


def test_tune_frame_includes_pose_when_requested(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/frame", params={"path": str(clip), "index": 0, "pose": 1}
    ).json()
    assert body["pose_available"] is True
    assert len(body["pose"]) == 1
    keypoints = body["pose"][0]["keypoints"]
    assert keypoints, "expected mock keypoints"
    names = {kp["name"] for kp in keypoints}
    assert "nose" in names


def test_tune_frame_clamps_index_past_end(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=5)
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/frame", params={"path": str(clip), "index": 999}
    ).json()
    assert body["index"] == 4  # clamped to total - 1


def test_tune_frame_pose_unavailable_without_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    # Force the "pose extra not installed" path deterministically (the dev env
    # may or may not have deeplabcut). With no injected estimator and the
    # superanimal backend reported absent, the endpoint must degrade to
    # pose_available False without attempting real inference.
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    client = make_client(tmp_path, clip, with_pose=False)
    body = client.get(
        "/api/tune/frame", params={"path": str(clip), "index": 0, "pose": 1}
    ).json()
    assert body["pose_available"] is False
    assert body["pose"] == []


def test_tune_frame_rejects_path_outside_roots(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    outside = tmp_path.parent / "evil.mp4"
    client = make_client(tmp_path, clip)
    resp = client.get("/api/tune/frame", params={"path": str(outside), "index": 0})
    assert resp.status_code == 400


def test_tune_frame_rejects_non_video(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    txt = tmp_path / "note.txt"
    txt.write_text("hi")
    client = make_client(tmp_path, clip)
    resp = client.get("/api/tune/frame", params={"path": str(txt), "index": 0})
    assert resp.status_code == 400


# --- pose resolver --------------------------------------------------------


def test_build_tune_pose_estimator_mock_backend(tmp_path: Path) -> None:
    config = Config(
        global_settings=GlobalSettings(dataset_dir=tmp_path / "dataset"),
        pose=PoseConfig(backend="mock"),
    )
    estimator, available = tune_mod.build_tune_pose_estimator(config)
    assert available is True
    assert estimator is not None


def test_build_tune_pose_estimator_superanimal_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    config = Config(
        global_settings=GlobalSettings(dataset_dir=tmp_path / "dataset"),
        pose=PoseConfig(backend="superanimal"),
    )
    estimator, available = tune_mod.build_tune_pose_estimator(config)
    # deeplabcut reported absent -> resolver degrades without building.
    assert available is False
    assert estimator is None


# --- models / meta / clip / detect (round 4) ------------------------------


def test_collect_tune_models_scans_dir_and_appends_default(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "b.pt").write_bytes(b"x")
    (models_dir / "a.pt").write_bytes(b"x")
    (models_dir / "notes.txt").write_text("ignore")
    config = make_config(tmp_path, tmp_path / "c.mp4")  # default model_name
    models = tune_mod.collect_tune_models(config, models_dir=models_dir)
    # discovered *.pt sorted by name, then the configured default appended.
    assert models[:2] == [str(models_dir / "a.pt"), str(models_dir / "b.pt")]
    assert models[-1] == config.global_settings.model_name
    assert all(not m.endswith(".txt") for m in models)


def test_tune_models_endpoint_lists_default(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.get("/api/tune/models").json()
    # Injected detector pins the allow-list to just the configured model.
    assert body["default"] == "models/yolo11m.pt"
    assert body["models"] == ["models/yolo11m.pt"]


def test_tune_meta_returns_clip_properties(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=8, size=(160, 120))
    client = make_client(tmp_path, clip)
    body = client.get("/api/tune/meta", params={"path": str(clip)}).json()
    assert body["total_frames"] == 8
    assert body["fps"] == pytest.approx(5.0)
    assert body["width"] == 160 and body["height"] == 120
    assert body["duration"] == pytest.approx(8 / 5.0)


def test_tune_meta_rejects_traversal(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    assert client.get("/api/tune/meta", params={"path": "/etc/hosts"}).status_code == 400


def test_tune_detect_returns_boxes_without_image(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=8)
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/detect", params={"path": str(clip), "index": 2, "pose": 0}
    ).json()
    assert "image" not in body  # the whole point: cheap, image-free payload
    assert body["index"] == 2
    assert body["model"] == "models/yolo11m.pt"
    assert body["detection_floor"] == pytest.approx(0.05)
    assert len(body["detections"]) == 1
    assert body["detections"][0]["class_name"] == "dog"
    assert body["pose"] == []


def test_tune_detect_includes_pose_when_requested(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/detect", params={"path": str(clip), "index": 0, "pose": 1}
    ).json()
    assert "image" not in body
    assert len(body["pose"]) == 1
    assert {kp["name"] for kp in body["pose"][0]["keypoints"]}


def test_tune_detect_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/detect",
        params={"path": str(clip), "index": 0, "model": "models/bogus.pt"},
    )
    assert resp.status_code == 400


def test_tune_frame_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/frame",
        params={"path": str(clip), "index": 0, "model": "models/bogus.pt"},
    )
    assert resp.status_code == 400


def test_tune_clip_supports_range(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    full = client.get("/api/tune/clip", params={"path": str(clip)})
    assert full.status_code == 200
    assert full.headers.get("accept-ranges") == "bytes"
    assert full.headers["content-type"] == "video/mp4"
    partial = client.get(
        "/api/tune/clip", params={"path": str(clip)}, headers={"Range": "bytes=0-3"}
    )
    assert partial.status_code == 206
    assert partial.headers["content-range"].startswith("bytes 0-3/")
    assert len(partial.content) == 4


def test_tune_clip_rejects_traversal(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    assert client.get("/api/tune/clip", params={"path": "/etc/hosts"}).status_code == 400


# --- SPA fallback (routing) -----------------------------------------------


def test_spa_fallback_serves_shell_for_client_routes(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    for route in ("/tune", "/live", "/", "/deep/link"):
        resp = client.get(route)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


def test_spa_fallback_does_not_shadow_api(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/api").status_code == 404
    # A real API route still works.
    assert client.get("/api/dogs").status_code == 200
