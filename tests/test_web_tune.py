"""Offline tests for the in-browser tuner backend and the SPA fallback route.

Everything here is offline: a fake detector and ``MockPoseEstimator`` are
injected, and clips are tiny synthetic mp4s written with ``cv2.VideoWriter`` — no
real YOLO/pose model, GPU, or network is touched.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
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


def test_collect_tune_models_discovers_mlpackage_dirs(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "coreml").mkdir(parents=True)
    (models_dir / "yolo11m.pt").write_bytes(b"x")
    # *.mlpackage is a DIRECTORY bundle, not a file.
    (models_dir / "yolo11m.mlpackage").mkdir()
    (models_dir / "yolo11m.mlpackage" / "Manifest.json").write_text("{}")
    (models_dir / "coreml" / "yolo11l.mlpackage").mkdir()
    # A regular file that merely ends in .mlpackage must NOT be discovered.
    (models_dir / "stray.mlpackage").write_text("not a bundle")
    config = make_config(tmp_path, tmp_path / "c.mp4")
    models = tune_mod.collect_tune_models(config, models_dir=models_dir)
    assert str(models_dir / "yolo11m.pt") in models
    assert str(models_dir / "yolo11m.mlpackage") in models
    assert str(models_dir / "coreml" / "yolo11l.mlpackage") in models
    assert str(models_dir / "stray.mlpackage") not in models


def test_tune_models_endpoint_lists_default(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.get("/api/tune/models").json()
    # Injected detector pins the allow-list to just the configured model.
    assert body["default"] == "models/yolo11m.pt"
    assert body["models"] == ["models/yolo11m.pt"]


def test_tune_models_endpoint_reports_coreml_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    client.app.state.tune_models.append("models/yolo11m.mlpackage")
    # Avoid touching coremltools / a real package in the offline suite.
    monkeypatch.setattr(
        "detectivepotty.detect.coreml_export.coreml_max_batch", lambda _p: 16
    )
    body = client.get("/api/tune/models").json()
    # Only `.mlpackage` entries get a batch; `.pt` weights are omitted.
    assert body["coreml_batch"] == {"models/yolo11m.mlpackage": 16}


def test_tune_export_coreml_adds_model_and_refreshes_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)

    calls: dict = {}

    def fake_export(weights, out_path=None, imgsz=640, half=True, batch=1):
        calls["weights"] = str(weights)
        calls["imgsz"] = imgsz
        calls["batch"] = batch
        return "models/yolo11m.mlpackage"

    # The endpoint resolves export_coreml off the module at call time, so patching
    # the module attribute keeps the test fully offline (no real CoreML export).
    monkeypatch.setattr(
        "detectivepotty.detect.coreml_export.export_coreml", fake_export
    )
    # Reading the batch label off the (fake) result must not touch coremltools.
    monkeypatch.setattr(
        "detectivepotty.detect.coreml_export.coreml_max_batch", lambda _p: 16
    )

    resp = client.post("/api/tune/export-coreml", json={"model": "models/yolo11m.pt"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "models/yolo11m.mlpackage"
    assert "models/yolo11m.mlpackage" in body["models"]
    assert calls["weights"] == "models/yolo11m.pt"
    assert calls["imgsz"] == client.app.state.config.global_settings.inference_long_edge_px
    # The export is batched to the tuner's batch size so detection can run a whole
    # frame window in one GPU forward.
    assert calls["batch"] == (
        client.app.state.config.global_settings.tune_detection_batch_size
    )
    # The response advertises the new package's baked batch for the picker label.
    assert body["coreml_batch"]["models/yolo11m.mlpackage"] == 16
    # Allow-list now includes the new model so it is selectable.
    listed = client.get("/api/tune/models").json()["models"]
    assert "models/yolo11m.mlpackage" in listed


def test_tune_export_coreml_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.post("/api/tune/export-coreml", json={"model": "models/ghost.pt"})
    assert resp.status_code == 400


def test_tune_export_coreml_rejects_non_pt(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    # An already-exported .mlpackage in the allow-list can't be re-exported.
    client.app.state.tune_models.append("models/yolo11m.mlpackage")
    resp = client.post(
        "/api/tune/export-coreml", json={"model": "models/yolo11m.mlpackage"}
    )
    assert resp.status_code == 400


def test_tune_pose_returns_keypoints_for_boxes(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)  # injects MockPoseEstimator
    resp = client.post(
        "/api/tune/pose",
        json={"path": str(clip), "index": 0, "boxes": [[10.0, 10.0, 80.0, 90.0]]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pose_available"] is True
    assert len(body["pose"]) == 1
    assert body["pose"][0]["keypoints"], "mock estimator should yield keypoints"


def test_tune_pose_unavailable_without_estimator(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip, with_pose=False)
    # Pin the resolver to "unavailable" so no real (heavy) backend is ever built.
    client.app.state.tune_pose_resolved = (None, False)
    resp = client.post(
        "/api/tune/pose",
        json={"path": str(clip), "index": 0, "boxes": [[10.0, 10.0, 80.0, 90.0]]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pose_available"] is False
    assert body["pose"] == []


def test_tune_pose_rejects_invalid_path(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.post(
        "/api/tune/pose",
        json={"path": "/etc/passwd", "index": 0, "boxes": [[0.0, 0.0, 1.0, 1.0]]},
    )
    assert resp.status_code == 400


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


class SceneFakeDetector:
    """Detector exposing ``detect_scene_objects`` with mixed (non-dog) classes."""

    device = "cpu"
    last_inference = None

    def detect(self, frame, frame_idx=0, mono_ts=None, wall_ts=None):  # noqa: ANN001
        return []

    def detect_scene_objects(self, frame, top_n=8):  # noqa: ANN001
        from detectivepotty.geometry import BBox

        ranked = [
            ("dog", 0.42, BBox(1.0, 1.0, 2.0, 2.0)),
            ("cat", 0.88, BBox(3.0, 4.0, 30.0, 40.0)),
            ("person", 0.55, BBox(5.0, 6.0, 50.0, 60.0)),
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_n]


def test_tune_scene_returns_top_all_class_objects(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=6)
    config = make_config(tmp_path, clip)
    client = TestClient(create_app(config, tune_detector=SceneFakeDetector()))
    body = client.get(
        "/api/tune/scene", params={"path": str(clip), "index": 1, "top_n": 2}
    ).json()
    assert body["index"] == 1
    assert body["model"] == "models/yolo11m.pt"
    assert body["detection_floor"] == pytest.approx(0.05)
    # No dog filter: highest-confidence classes surface, sorted desc, capped to top_n.
    assert [o["class_name"] for o in body["objects"]] == ["cat", "person"]
    assert body["objects"][0]["confidence"] == pytest.approx(0.88)
    # Each object carries its original-frame box so the client can overlay it.
    cat = body["objects"][0]
    assert (cat["x1"], cat["y1"], cat["x2"], cat["y2"]) == pytest.approx(
        (3.0, 4.0, 30.0, 40.0)
    )


def test_tune_scene_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    config = make_config(tmp_path, clip)
    client = TestClient(create_app(config, tune_detector=SceneFakeDetector()))
    resp = client.get(
        "/api/tune/scene", params={"path": str(clip), "index": 0, "model": "nope.pt"}
    )
    assert resp.status_code == 400


def test_detect_scene_objects_no_dog_filter_and_floor() -> None:
    """The DogDetector helper surfaces all classes >= floor, sorted desc, with boxes."""

    from detectivepotty.detect.yolo import DogDetector
    from detectivepotty.geometry import BBox

    detector = DogDetector.__new__(DogDetector)
    detector.conf_threshold = 0.05
    detector._predict = lambda frame: ["result"]  # type: ignore[attr-defined]
    detector._iter_boxes = lambda results: [  # type: ignore[attr-defined]
        ((0.0, 0.0, 1.0, 1.0), 0.9, "cat"),
        ((0.0, 0.0, 2.0, 3.0), 0.6, "dog"),
        ((0.0, 0.0, 1.0, 1.0), 0.02, "bird"),  # below floor -> dropped
        ((0.0, 0.0, 10.0, 10.0), 0.3, "person"),  # clipped to the 4x4 frame
    ]
    objects = detector.detect_scene_objects(np.zeros((4, 4, 3), dtype=np.uint8), top_n=3)
    assert [(name, conf) for name, conf, _box in objects] == [
        ("cat", 0.9),
        ("dog", 0.6),
        ("person", 0.3),
    ]
    # Boxes are returned alongside, clipped to the original frame.
    assert objects[0][2] == BBox(0.0, 0.0, 1.0, 1.0)
    assert objects[2][2] == BBox(0.0, 0.0, 4.0, 4.0)


class FrameKeyedBatchDetector:
    """Per-frame-varying detector exposing ``detect`` and ``detect_batch``.

    The box position depends on ``frame_idx`` so a range result can be checked
    frame-for-frame against the single-frame endpoint. ``batch_calls`` records the
    size of every ``detect_batch`` call so a test can prove a real batch formed.
    """

    device = "cpu"
    last_inference = None

    def __init__(self) -> None:
        self.batch_calls: list[int] = []

    @staticmethod
    def _dets(frame_idx: int, wall_ts) -> list[Detection]:  # noqa: ANN001
        return [
            Detection(
                bbox=BBox(10.0 + frame_idx, 10.0, 80.0 + frame_idx, 90.0),
                confidence=0.7,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=0.0,
                wall_ts=wall_ts or datetime.now(timezone.utc),
            )
        ]

    def detect(self, frame, frame_idx=0, mono_ts=None, wall_ts=None):  # noqa: ANN001
        del frame, mono_ts
        return self._dets(frame_idx, wall_ts)

    def detect_batch(self, frames, metas):  # noqa: ANN001
        del frames
        self.batch_calls.append(len(metas))
        return [self._dets(m.frame_idx, m.wall_ts) for m in metas]


def test_tune_detect_range_matches_single_frame_calls(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=8)
    detector = FrameKeyedBatchDetector()
    client = TestClient(create_app(make_config(tmp_path, clip), tune_detector=detector))

    ranged = client.get(
        "/api/tune/detect_range",
        params={"path": str(clip), "start": 1, "count": 4},
    ).json()
    assert [f["index"] for f in ranged["frames"]] == [1, 2, 3, 4]

    # One batched forward of 4 frames, not four single-frame calls.
    assert detector.batch_calls == [4]

    # Each frame's boxes match the dedicated single-frame endpoint exactly.
    for entry in ranged["frames"]:
        single = client.get(
            "/api/tune/detect",
            params={"path": str(clip), "index": entry["index"], "pose": 0},
        ).json()
        assert entry["detections"] == single["detections"]
        assert entry["total_frames"] == single["total_frames"]
        assert entry["width"] == single["width"]
        assert entry["height"] == single["height"]


def test_tune_detect_range_caps_count_to_batch_size(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=40)
    config = make_config(tmp_path, clip)
    config.global_settings.tune_detection_batch_size = 5
    detector = FrameKeyedBatchDetector()
    client = TestClient(create_app(config, tune_detector=detector))

    ranged = client.get(
        "/api/tune/detect_range",
        params={"path": str(clip), "start": 0, "count": 50},
    ).json()
    # The request asked for 50 but the configured cap is 5.
    assert len(ranged["frames"]) == 5
    assert detector.batch_calls == [5]


def test_tune_detect_range_clamps_at_eof(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=6)
    detector = FrameKeyedBatchDetector()
    client = TestClient(create_app(make_config(tmp_path, clip), tune_detector=detector))

    ranged = client.get(
        "/api/tune/detect_range",
        params={"path": str(clip), "start": 4, "count": 8},
    ).json()
    # Only frames 4 and 5 exist; the run stops at EOF rather than erroring.
    assert [f["index"] for f in ranged["frames"]] == [4, 5]


def test_tune_detect_range_falls_back_without_detect_batch(tmp_path: Path) -> None:
    # The plain FakeDetector has no detect_batch; the endpoint must still work by
    # looping detect, returning one entry per frame.
    clip = write_clip(tmp_path / "c.mp4", frames=8)
    client = make_client(tmp_path, clip)
    ranged = client.get(
        "/api/tune/detect_range",
        params={"path": str(clip), "start": 0, "count": 3},
    ).json()
    assert [f["index"] for f in ranged["frames"]] == [0, 1, 2]
    for entry in ranged["frames"]:
        assert entry["detections"][0]["class_name"] == "dog"


def test_tune_detect_range_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/detect_range",
        params={"path": str(clip), "start": 0, "count": 2, "model": "models/bogus.pt"},
    )
    assert resp.status_code == 400


# --- tracking (track_detections + /api/tune/track_range) ------------------


def _det(frame_idx: int, x1: float, y1: float, x2: float, y2: float, conf: float = 0.7):
    """Build one ``Detection`` for a track_detections unit test (no model)."""

    return Detection(
        bbox=BBox(x1, y1, x2, y2),
        confidence=conf,
        class_name="dog",
        frame_idx=frame_idx,
        mono_ts=0.0,
        wall_ts=datetime.now(timezone.utc),
    )


def test_track_detections_keeps_overlapping_boxes_on_one_track() -> None:
    # A single dog drifting a few pixels per sample keeps high IoU, so the tracker
    # holds one persistent id across all sampled frames.
    per_frame = [
        (0, [_det(0, 10, 10, 80, 90)]),
        (5, [_det(5, 15, 10, 85, 90)]),
        (10, [_det(10, 20, 10, 90, 90)]),
    ]
    out = tune_mod.track_detections(per_frame, fps=30.0)
    assert out["stats"]["n_tracks"] == 1
    ids = {d["track_id"] for f in out["frames"] for d in f["detections"]}
    assert len(ids) == 1
    # Every sampled frame carries exactly that one tracked box.
    assert [f["index"] for f in out["frames"]] == [0, 5, 10]
    assert all(len(f["detections"]) == 1 for f in out["frames"])
    assert out["stats"]["n_detections"] == 3
    assert out["stats"]["n_sampled_frames"] == 3


def test_track_detections_center_gate_reassociates_far_jump() -> None:
    # Two sampled frames whose boxes do NOT overlap (IoU=0) but whose centers are
    # within ~0.9 box-diagonals. The center-distance gate decides id continuity.
    per_frame = [
        (0, [_det(0, 10, 10, 50, 50)]),
        (5, [_det(5, 60, 10, 100, 50)]),
    ]
    # Gate disabled (pure IoU): the far jump spawns a second track.
    off = tune_mod.track_detections(per_frame, fps=30.0, center_dist_gate=0.0)
    assert off["stats"]["n_tracks"] == 2
    # Gate on (harvest default 1.5): re-associated to one track.
    on = tune_mod.track_detections(per_frame, fps=30.0, center_dist_gate=1.5)
    assert on["stats"]["n_tracks"] == 1


def test_track_detections_is_order_independent() -> None:
    # Replay must be in ascending frame order regardless of input order, so a
    # shuffled per_frame yields the identical result.
    frames = [
        (0, [_det(0, 10, 10, 80, 90)]),
        (5, [_det(5, 15, 10, 85, 90)]),
        (10, [_det(10, 20, 10, 90, 90)]),
    ]
    ordered = tune_mod.track_detections(list(frames), fps=30.0)
    shuffled = tune_mod.track_detections([frames[2], frames[0], frames[1]], fps=30.0)
    assert ordered == shuffled
    assert [f["index"] for f in shuffled["frames"]] == [0, 5, 10]


def test_track_detections_stats_report_fragmentation() -> None:
    # One clean continuous presence -> one span, one presence window, ratio 1.0.
    per_frame = [(idx, [_det(idx, 10, 10, 80, 90)]) for idx in range(0, 30, 5)]
    out = tune_mod.track_detections(per_frame, fps=30.0, total_frames=30)
    stats = out["stats"]
    assert stats["n_presence_windows"] == 1
    assert stats["spans_per_window"] == stats["n_spans"] / stats["n_presence_windows"]
    # The knobs are echoed back so the UI can label the comparison.
    assert stats["sample_every"] == 5
    assert stats["iou_threshold"] == 0.3
    assert stats["max_age_frames"] == 15
    assert stats["center_dist_gate"] == 1.5


def test_track_range_endpoint_returns_track_ids_and_stats(tmp_path: Path) -> None:
    # The batch detector moves its box +1px/frame, so sampled frames overlap and
    # the tracker keeps one persistent id across the tracked range.
    clip = write_clip(tmp_path / "c.mp4", frames=16)
    detector = FrameKeyedBatchDetector()
    client = TestClient(create_app(make_config(tmp_path, clip), tune_detector=detector))

    body = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "start": 0, "count": 16, "sample_every": 5},
    ).json()
    # Only multiples-of-5 frames are sampled/tracked (absolute source numbering).
    assert [f["index"] for f in body["frames"]] == [0, 5, 10, 15]
    ids = {d["track_id"] for f in body["frames"] for d in f["detections"]}
    assert ids == {"1"}
    assert body["stats"]["n_tracks"] == 1
    assert body["stats"]["n_sampled_frames"] == 4
    assert body["model"]
    # Boxes carry a track_id on top of the detection fields.
    first = body["frames"][0]["detections"][0]
    assert set(first) >= {"x1", "y1", "x2", "y2", "confidence", "class_name", "track_id"}


def test_track_range_endpoint_falls_back_without_detect_batch(tmp_path: Path) -> None:
    # The plain FakeDetector has no detect_batch; the endpoint must still track by
    # looping detect over the sampled frames.
    clip = write_clip(tmp_path / "c.mp4", frames=16)
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "start": 0, "count": 16, "sample_every": 5},
    ).json()
    assert [f["index"] for f in body["frames"]] == [0, 5, 10, 15]
    assert body["stats"]["n_tracks"] == 1


def test_track_range_endpoint_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "start": 0, "count": 4, "model": "models/bogus.pt"},
    )
    assert resp.status_code == 400


def test_track_detections_stats_tagged_ours() -> None:
    per_frame = [(idx, [_det(idx, 10, 10, 80, 90)]) for idx in range(0, 30, 5)]
    out = tune_mod.track_detections(per_frame, fps=30.0, total_frames=30)
    assert out["stats"]["tracker"] == "ours"


def test_summarize_tracked_frames_counts_tracks_and_windows() -> None:
    # Hand-built tracked frames (the shape the Ultralytics path emits): two distinct
    # ids -> two tracks; ultra path passes no ours-knobs so they read back None.
    box = {"x1": 10.0, "y1": 10.0, "x2": 80.0, "y2": 90.0, "class_name": "dog"}
    out_frames = [
        {"index": 0, "detections": [{**box, "confidence": 0.8, "track_id": "1"}]},
        {"index": 5, "detections": [{**box, "confidence": 0.7, "track_id": "1"}]},
        {"index": 10, "detections": [{**box, "confidence": 0.6, "track_id": "2"}]},
    ]
    stats = tune_mod.summarize_tracked_frames(
        out_frames, fps=30.0, total_frames=30, sample_every=5, tracker="botsort"
    )
    assert stats["tracker"] == "botsort"
    assert stats["n_tracks"] == 2
    assert stats["track_ids"] == ["1", "2"]
    assert stats["n_detections"] == 3
    assert stats["n_sampled_frames"] == 3
    assert stats["sample_every"] == 5
    assert stats["iou_threshold"] is None
    assert stats["max_age_frames"] is None
    assert stats["center_dist_gate"] is None
    assert stats["n_presence_windows"] >= 1
    assert stats["n_spans"] >= stats["n_presence_windows"]
    assert stats["spans_per_window"] == stats["n_spans"] / stats["n_presence_windows"]


def test_summarize_tracked_frames_matches_track_detections() -> None:
    # The summarizer, replayed over track_detections' own frames, reproduces the
    # span/window/track stats it embedded — so ours + ultra share one stats path.
    per_frame = [(idx, [_det(idx, 10, 10, 80, 90)]) for idx in range(0, 30, 5)]
    out = tune_mod.track_detections(per_frame, fps=30.0, total_frames=30)
    again = tune_mod.summarize_tracked_frames(
        out["frames"], fps=30.0, total_frames=30, sample_every=5, tracker="ours"
    )
    for key in ("n_tracks", "n_spans", "n_presence_windows", "spans_per_window"):
        assert again[key] == out["stats"][key]


def test_track_range_default_tracker_is_ours(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=16)
    client = make_client(tmp_path, clip)
    body = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "start": 0, "count": 16, "sample_every": 5},
    ).json()
    assert body["stats"]["tracker"] == "ours"


def test_track_range_rejects_unknown_tracker(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "count": 4, "tracker": "nope"},
    )
    assert resp.status_code == 400


def test_track_range_ultralytics_requires_pt_model(tmp_path: Path) -> None:
    # An Ultralytics tracker against a CoreML model is rejected before any model
    # build — the .pt guard, defense-in-depth behind the disabled UI option.
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    client.app.state.tune_models.append("models/yolo11m.mlpackage")
    resp = client.get(
        "/api/tune/track_range",
        params={
            "path": str(clip),
            "count": 4,
            "model": "models/yolo11m.mlpackage",
            "tracker": "botsort",
        },
    )
    assert resp.status_code == 400
    assert ".pt" in resp.json()["detail"]


def test_track_range_ultralytics_unavailable_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the `lap` association dep is missing, the ultra path 400s with a clear
    # message instead of building a model and erroring mid-request.
    import detectivepotty.web.app as app_mod

    monkeypatch.setattr(app_mod, "_ultralytics_tracking_available", lambda: False)
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "count": 4, "tracker": "bytetrack"},
    )
    assert resp.status_code == 400
    assert "lap" in resp.json()["detail"]


# --- track_range_stream (NDJSON forward-fill) -----------------------------


def _read_ndjson(resp) -> list[dict]:  # noqa: ANN001
    """Parse a non-empty-line NDJSON body into a list of records."""

    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_track_range_stream_emits_frames_then_done(tmp_path: Path) -> None:
    # The ours backend streams one or more `frames` records then a final `done`,
    # and the union of the streamed frames equals the non-streaming payload's frames.
    clip = write_clip(tmp_path / "c.mp4", frames=16)
    detector = FrameKeyedBatchDetector()
    client = TestClient(create_app(make_config(tmp_path, clip), tune_detector=detector))

    resp = client.get(
        "/api/tune/track_range_stream",
        params={"path": str(clip), "start": 0, "count": 16, "sample_every": 5},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    records = _read_ndjson(resp)
    assert records, "stream produced no records"
    # Last record is the terminal done; everything before is a frames batch.
    assert records[-1]["type"] == "done"
    assert all(r["type"] == "frames" for r in records[:-1])

    streamed_frames = [f for r in records[:-1] for f in r["frames"]]
    assert [f["index"] for f in streamed_frames] == [0, 5, 10, 15]

    # Draining-parity: the streamed frames + done.stats match the non-streaming call.
    non_stream = client.get(
        "/api/tune/track_range",
        params={"path": str(clip), "start": 0, "count": 16, "sample_every": 5},
    ).json()
    assert streamed_frames == non_stream["frames"]
    assert records[-1]["stats"] == non_stream["stats"]
    assert records[-1]["model"] == non_stream["model"]


def test_track_range_stream_rejects_unknown_tracker(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/track_range_stream",
        params={"path": str(clip), "count": 4, "tracker": "nope"},
    )
    assert resp.status_code == 400


def test_track_range_stream_ultralytics_requires_pt_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    client.app.state.tune_models.append("models/yolo11m.mlpackage")
    resp = client.get(
        "/api/tune/track_range_stream",
        params={
            "path": str(clip),
            "count": 4,
            "model": "models/yolo11m.mlpackage",
            "tracker": "botsort",
        },
    )
    assert resp.status_code == 400
    assert ".pt" in resp.json()["detail"]


def test_track_range_stream_ultralytics_unavailable_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import detectivepotty.web.app as app_mod

    monkeypatch.setattr(app_mod, "_ultralytics_tracking_available", lambda: False)
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/track_range_stream",
        params={"path": str(clip), "count": 4, "tracker": "bytetrack"},
    )
    assert resp.status_code == 400
    assert "lap" in resp.json()["detail"]


class _BatchRecordingEstimator(MockPoseEstimator):
    """MockPoseEstimator that records ``estimate_batch`` sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def estimate_batch(self, requests):  # noqa: ANN001
        self.batch_sizes.append(len(requests))
        return super().estimate_batch(requests)


def test_tune_pose_batches_boxes_through_estimate_batch(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    estimator = _BatchRecordingEstimator()
    client = TestClient(
        create_app(
            make_config(tmp_path, clip),
            tune_detector=FakeDetector(),
            tune_pose_estimator=estimator,
        )
    )
    body = client.post(
        "/api/tune/pose",
        json={
            "path": str(clip),
            "index": 0,
            "boxes": [[10, 10, 80, 90], [20, 20, 100, 110], [5, 5, 50, 60]],
        },
    ).json()
    assert body["pose_available"] is True
    assert len(body["pose"]) == 3
    # All three boxes went through a single estimate_batch call.
    assert estimator.batch_sizes == [3]


def test_tune_detect_rejects_unknown_model(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.get(
        "/api/tune/detect",
        params={"path": str(clip), "index": 0, "model": "models/bogus.pt"},
    )
    assert resp.status_code == 400


# --- batched pose pass (POST /api/tune/pose_range) ------------------------


def test_tune_pose_range_matches_per_frame_pose(tmp_path: Path) -> None:
    """Batched multi-frame pose == repeated single-frame /api/tune/pose."""
    clip = write_clip(tmp_path / "c.mp4")
    boxes = [[10.0, 10.0, 80.0, 90.0]]
    indices = [0, 1, 2]

    per_frame = make_client(tmp_path, clip)
    expected = {}
    for idx in indices:
        body = per_frame.post(
            "/api/tune/pose", json={"path": str(clip), "index": idx, "boxes": boxes}
        ).json()
        expected[idx] = body["pose"]

    batched = make_client(tmp_path, clip)
    body = batched.post(
        "/api/tune/pose_range",
        json={"path": str(clip), "frames": [{"index": i, "boxes": boxes} for i in indices]},
    ).json()
    got = {f["index"]: f["pose"] for f in body["frames"]}
    assert [f["index"] for f in body["frames"]] == indices  # request order preserved
    assert got == expected


def test_tune_pose_range_single_estimate_batch_across_frames(tmp_path: Path) -> None:
    """All boxes from all frames go through ONE estimate_batch forward."""
    clip = write_clip(tmp_path / "c.mp4")
    estimator = _BatchRecordingEstimator()
    client = TestClient(
        create_app(
            make_config(tmp_path, clip),
            tune_detector=FakeDetector(),
            tune_pose_estimator=estimator,
        )
    )
    box = [[10.0, 10.0, 80.0, 90.0]]
    body = client.post(
        "/api/tune/pose_range",
        json={"path": str(clip), "frames": [{"index": i, "boxes": box} for i in (0, 1, 2)]},
    ).json()
    assert len(body["frames"]) == 3
    # 3 frames x 1 box = a single batched forward of 3 (not 3 batch-1 forwards).
    assert estimator.batch_sizes == [3]


def test_tune_pose_range_skips_degenerate_boxes_without_misalign(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    body = client.post(
        "/api/tune/pose_range",
        json={
            "path": str(clip),
            "frames": [
                {"index": 0, "boxes": [[10, 10, 80, 90], [5, 5, 5, 60]]},  # 2nd zero-width
                {"index": 1, "boxes": [[20, 20, 100, 110]]},
            ],
        },
    ).json()
    by_idx = {f["index"]: f["pose"] for f in body["frames"]}
    assert len(by_idx[0]) == 1  # degenerate box dropped, valid one kept
    assert len(by_idx[1]) == 1


def test_tune_pose_range_caps_frames_by_batch_size(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4", frames=8)
    config = make_config(tmp_path, clip)
    config.global_settings.tune_detection_batch_size = 2
    estimator = _BatchRecordingEstimator()
    client = TestClient(
        create_app(config, tune_detector=FakeDetector(), tune_pose_estimator=estimator)
    )
    box = [[10.0, 10.0, 80.0, 90.0]]
    body = client.post(
        "/api/tune/pose_range",
        json={"path": str(clip), "frames": [{"index": i, "boxes": box} for i in range(4)]},
    ).json()
    assert [f["index"] for f in body["frames"]] == [0, 1]  # capped to 2 frames
    assert estimator.batch_sizes == [2]


def test_tune_pose_range_caps_total_crops(tmp_path: Path) -> None:
    from detectivepotty.web.app import TUNE_POSE_MAX_CROPS

    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    full_boxes = [[10.0, 10.0, 80.0, 90.0]] * TUNE_POSE_MAX_CROPS  # one frame hits cap
    body = client.post(
        "/api/tune/pose_range",
        json={
            "path": str(clip),
            "frames": [{"index": 0, "boxes": full_boxes}, {"index": 1, "boxes": full_boxes}],
        },
    ).json()
    assert [f["index"] for f in body["frames"]] == [0]  # 2nd frame dropped by crop cap


def test_tune_pose_range_unavailable_without_estimator(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip, with_pose=False)
    client.app.state.tune_pose_resolved = (None, False)
    body = client.post(
        "/api/tune/pose_range",
        json={
            "path": str(clip),
            "frames": [{"index": 0, "boxes": [[10, 10, 80, 90]]}, {"index": 1, "boxes": []}],
        },
    ).json()
    assert [f["index"] for f in body["frames"]] == [0, 1]
    assert all(f["pose_available"] is False and f["pose"] == [] for f in body["frames"])


def test_tune_pose_range_rejects_invalid_path(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "c.mp4")
    client = make_client(tmp_path, clip)
    resp = client.post(
        "/api/tune/pose_range",
        json={"path": "/etc/passwd", "frames": [{"index": 0, "boxes": [[0, 0, 1, 1]]}]},
    )
    assert resp.status_code == 400


def test_pose_payload_for_frames_handles_none_frame_terminal() -> None:
    """A None (un-decodable) frame still returns its index with empty entries."""
    estimator = _BatchRecordingEstimator()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    results = tune_mod.pose_payload_for_frames(
        estimator,
        [
            (5, None, [[10.0, 10.0, 80.0, 90.0]]),  # decode failed
            (6, frame, [[10.0, 10.0, 80.0, 90.0]]),
        ],
    )
    assert [idx for idx, _ in results] == [5, 6]  # every requested index present, in order
    assert results[0][1] == []  # None frame -> no entries (terminal, not retried)
    assert len(results[1][1]) == 1
    # The None frame contributed no crops: a single batched forward of just 1.
    assert estimator.batch_sizes == [1]


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


# --- persistent decoder cache (ClipFrameReader) --------------------------


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Keep the process-wide reader cache from leaking between tests."""

    tune_mod.clear_clip_reader_cache()
    yield
    tune_mod.clear_clip_reader_cache()


def _decode_all_sequential(path: Path) -> list[np.ndarray]:
    """Ground-truth frames: a fresh capture read straight through, in order."""

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def test_reader_sequential_matches_ground_truth(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "seq.mp4", frames=20)
    truth = _decode_all_sequential(clip)
    reader = tune_mod.get_clip_reader(clip)
    for i in range(len(truth)):
        frame, idx, total, _fps, _w, _h = reader.read(i)
        assert idx == i
        assert total == len(truth)
        assert np.array_equal(frame, truth[i]), f"sequential frame {i} mismatch"


def test_reader_random_access_matches_ground_truth(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "rand.mp4", frames=20)
    truth = _decode_all_sequential(clip)
    reader = tune_mod.get_clip_reader(clip)
    # Mix of forward jumps (grab-skip path), backward jumps (seek path), repeats,
    # a large forward jump past FORWARD_GRAB_MAX, and the very first frame again.
    order = [0, 1, 2, 5, 6, 3, 19, 4, 0, 18, 10, 11, 12]
    for i in order:
        frame, idx, _total, _fps, _w, _h = reader.read(i)
        assert idx == i
        assert np.array_equal(frame, truth[i]), f"random frame {i} mismatch"


def test_reader_clamps_index_past_end(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "clamp.mp4", frames=8)
    reader = tune_mod.get_clip_reader(clip)
    frame, idx, total, _fps, _w, _h = reader.read(999)
    assert idx == total - 1 == 7
    assert frame is not None


def test_reader_cache_reuses_same_instance(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "reuse.mp4", frames=8)
    a = tune_mod.get_clip_reader(clip)
    b = tune_mod.get_clip_reader(clip)
    assert a is b, "same path within cache should return the same open reader"


def test_reader_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "mtime.mp4", frames=8)
    first = tune_mod.get_clip_reader(clip)
    assert first.meta()[0] == 8

    # Rewrite the clip with a different frame count and force a newer mtime so the
    # cache must rebuild the reader rather than serve the stale (now-wrong) one.
    write_clip(clip, frames=15)
    bump = os.stat(clip).st_mtime_ns + 2_000_000_000
    os.utime(clip, ns=(bump, bump))

    second = tune_mod.get_clip_reader(clip)
    assert second is not first
    assert second.meta()[0] == 15
    assert first.retired, "stale reader should be retired (capture released)"


def test_reader_cache_evicts_oldest_over_capacity(tmp_path: Path) -> None:
    readers = []
    for i in range(tune_mod.MAX_OPEN_READERS + 2):
        clip = write_clip(tmp_path / f"evict_{i}.mp4", frames=6)
        readers.append(tune_mod.get_clip_reader(clip))

    with tune_mod._READER_CACHE_LOCK:
        assert len(tune_mod._READER_CACHE) == tune_mod.MAX_OPEN_READERS
    # The two oldest readers were evicted and must have been closed.
    assert readers[0].retired
    assert readers[1].retired
    # The most-recent ones are still live.
    assert not readers[-1].retired


def test_read_frame_retries_after_reader_retired(tmp_path: Path) -> None:
    """read_frame transparently recovers if its reader is retired mid-use."""

    clip = write_clip(tmp_path / "retire.mp4", frames=8)
    stale = tune_mod.get_clip_reader(clip)
    stale.close()  # simulate an eviction racing ahead of the read
    # The cache still holds the retired instance; read_frame should fetch a fresh
    # one and succeed rather than raising on the closed capture.
    frame, idx, _total, _fps, _w, _h = tune_mod.read_frame(clip, 3)
    assert idx == 3 and frame is not None


def test_reader_concurrent_reads_are_safe(tmp_path: Path) -> None:
    import threading as _threading

    clip = write_clip(tmp_path / "threads.mp4", frames=20)
    truth = _decode_all_sequential(clip)
    reader = tune_mod.get_clip_reader(clip)
    errors: list[str] = []

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(40):
            i = int(rng.integers(0, len(truth)))
            try:
                frame, idx, _t, _f, _w, _h = reader.read(i)
                if idx != i or not np.array_equal(frame, truth[i]):
                    errors.append(f"seed {seed} frame {i} mismatch")
            except Exception as exc:  # noqa: BLE001 - surface any crash
                errors.append(f"seed {seed}: {exc!r}")

    workers = [_threading.Thread(target=worker, args=(s,)) for s in range(6)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert not errors, errors[:5]
