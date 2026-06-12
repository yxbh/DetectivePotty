"""Offline tests for the range-labeling backend (``/api/label`` + pure helpers).

A fake harvested clip dir (tiny synthetic ``clip.mp4`` + a hand-built
``metadata.json``) stands in for real harvest output; no model, GPU, or network
is touched. A ``FakeDetector`` is injected only to satisfy app creation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from detectivepotty.config import Config, GlobalSettings
from detectivepotty.web import create_app
from detectivepotty.web import labeling


def _write_clip(path: Path, frames: int = 12, size: tuple[int, int] = (160, 120)) -> None:
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )
    assert writer.isOpened(), "could not open VideoWriter (codec missing?)"
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = ((i * 17) % 255, 60, 200)
        writer.write(frame)
    writer.release()


def _make_clip_dir(
    root: Path,
    span_id: str,
    *,
    track_id: str = "1",
    date: str = "2026-06-08",
    n_det: int = 4,
    source_id: str = "cam_backyard",
    camera_name: str | None = None,
    detect_conf: float | None = None,
    span_start_utc: str | None = None,
    span_end_utc: str | None = None,
    source_start_utc: str | None = None,
    start_s: float | None = None,
    det_time_s: list[float] | None = None,
    model_name: str | None = None,
    class_names: list[str] | None = None,
) -> Path:
    clip_dir = root / span_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    _write_clip(clip_dir / "clip.mp4")
    detections = [
        {
            "clip_frame_idx": i * 2,
            "source_frame_idx": 100 + i * 2,
            "time_s": (det_time_s[i] if det_time_s is not None else float(i)),
            "track_id": track_id,
            "bbox": {"x1": 10.0 + i, "y1": 12.0, "x2": 80.0 + i, "y2": 90.0},
            "confidence": 0.7,
            **(
                {"class_name": class_names[i % len(class_names)]}
                if class_names
                else {}
            ),
        }
        for i in range(n_det)
    ]
    meta: dict = {
        "schema_version": "harvest-1.0",
        "span_id": span_id,
        "source_id": source_id,
        "source_span_start_utc": span_start_utc or f"{date}T17:29:38+00:00",
        "fps": 30.0,
        "frame_count": 12,
        "width": 160,
        "height": 120,
        "timebase": "clip_frames",
        "track_id": track_id,
        "detections": detections,
    }
    if camera_name is not None:
        meta["camera_name"] = camera_name
    if detect_conf is not None:
        meta["detect_conf"] = detect_conf
    if model_name is not None:
        meta["model_name"] = model_name
    if span_end_utc is not None:
        meta["source_span_end_utc"] = span_end_utc
    if source_start_utc is not None:
        meta["source_start_utc"] = source_start_utc
    if start_s is not None:
        meta["start_s"] = start_s
    (clip_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return clip_dir


def _config(harvest_root: Path, tmp_path: Path) -> Config:
    return Config(
        global_settings=GlobalSettings(
            dataset_dir=tmp_path / "dataset", harvest_dir=harvest_root
        )
    )


def _client(harvest_root: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_app(_config(harvest_root, tmp_path)))


# --- pure helpers ---------------------------------------------------------


def test_discover_and_summarize(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a")
    dirs = labeling.discover_clip_dirs(root)
    assert [d.name for d in dirs] == ["span_a"]
    summary = labeling.summarize_clip(dirs[0])
    assert summary["span_id"] == "span_a"
    assert summary["track_id"] == "1"
    assert summary["n_detections"] == 4
    assert summary["labeled"] is False
    assert summary["date"] == "2026-06-08"
    assert summary["duration_s"] == pytest.approx(0.4, abs=1e-3)


def test_discover_skips_dirs_without_clip(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    bare = root / "no_clip"
    bare.mkdir(parents=True)
    (bare / "metadata.json").write_text("{}", encoding="utf-8")
    assert labeling.discover_clip_dirs(root) == []


def test_clip_dir_for_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a")
    assert labeling.clip_dir_for(root, "span_a").name == "span_a"
    for bad in ("..", "../span_a", "a/b", "", "."):
        with pytest.raises(ValueError):
            labeling.clip_dir_for(root, bad)
    with pytest.raises(ValueError):
        labeling.clip_dir_for(root, "missing_span")


def test_clip_detail_groups_tracks(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    clip_dir = _make_clip_dir(root, "span_a", track_id="3", n_det=3)
    detail = labeling.clip_detail(clip_dir)
    assert set(detail["tracks"]) == {"3"}
    boxes = detail["tracks"]["3"]
    assert [b["clip_frame_idx"] for b in boxes] == [0, 2, 4]
    assert boxes[0]["bbox"]["x1"] == 10.0
    assert detail["labels"]["ranges"] == []
    # Own track is surfaced as a present (self) track.
    assert detail["n_tracks"] == 1
    self_track = detail["present_tracks"]["span_a:3"]
    assert self_track["is_self"] is True
    assert self_track["track_id"] == "3"


def test_summarize_records_model_and_class_distribution(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    # dog ×3, sheep ×1 -> distribution sorted by count desc.
    clip_dir = _make_clip_dir(
        root,
        "span_a",
        n_det=4,
        model_name="models/yolo11m.pt",
        class_names=["dog", "dog", "dog", "sheep"],
    )
    summary = labeling.summarize_clip(clip_dir)
    assert summary["model_name"] == "models/yolo11m.pt"
    assert summary["class_distribution"] == [
        {"class_name": "dog", "count": 3},
        {"class_name": "sheep", "count": 1},
    ]
    # The alias class is also carried per-box in the detail tracks.
    detail = labeling.clip_detail(clip_dir)
    classes = {b.get("class_name") for b in detail["tracks"]["1"]}
    assert classes == {"dog", "sheep"}


def test_summarize_legacy_metadata_defaults_model_and_class(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    # Legacy clip: no model_name, no per-detection class_name.
    clip_dir = _make_clip_dir(root, "span_a", n_det=4)
    summary = labeling.summarize_clip(clip_dir)
    assert summary["model_name"] is None  # frontend renders "unknown"
    assert summary["class_distribution"] == [{"class_name": "dog", "count": 4}]
    detail = labeling.clip_detail(clip_dir)
    assert all(b["class_name"] == "dog" for b in detail["tracks"]["1"])


def test_summarize_exposes_identity_and_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(
        root,
        "span_a",
        source_id="6695ef21030c4603e400040d@20260606T094830Z",
        camera_name="Backyard Grass",
        detect_conf=0.25,
        span_start_utc="2026-06-06T09:49:22+00:00",
        span_end_utc="2026-06-06T09:49:25+00:00",
    )
    summary = labeling.summarize_clip(root / "span_a")
    assert summary["camera_name"] == "Backyard Grass"
    assert summary["camera_id"] == "6695ef21030c4603e400040d"
    assert summary["detect_conf"] == 0.25
    assert summary["span_start_utc"] == "2026-06-06T09:49:22+00:00"
    assert summary["span_end_utc"] == "2026-06-06T09:49:25+00:00"


def test_camera_name_falls_back_to_cameras_json(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    root.mkdir(parents=True)
    (root / "cameras.json").write_text(
        json.dumps({"6695ef21030c4603e400040d": "Backyard Grass"}), encoding="utf-8"
    )
    # No camera_name in metadata -> resolved via the sidecar by camera id.
    _make_clip_dir(root, "span_a", source_id="6695ef21030c4603e400040d@20260606T0Z")
    rows = labeling.list_clips(root)
    assert rows[0]["camera_name"] == "Backyard Grass"


def test_scene_grouping_clusters_overlapping_clips(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    cam = "camA@w"
    # span_a and span_b overlap in time on camera A -> same scene.
    _make_clip_dir(
        root, "span_a", source_id=cam,
        span_start_utc="2026-06-06T09:49:20+00:00",
        span_end_utc="2026-06-06T09:49:26+00:00",
    )
    _make_clip_dir(
        root, "span_b", source_id=cam, track_id="2",
        span_start_utc="2026-06-06T09:49:24+00:00",
        span_end_utc="2026-06-06T09:49:30+00:00",
    )
    # span_c is a different, non-overlapping window -> its own scene (alone).
    _make_clip_dir(
        root, "span_c", source_id=cam,
        span_start_utc="2026-06-06T11:00:00+00:00",
        span_end_utc="2026-06-06T11:00:05+00:00",
    )
    by_id = {r["span_id"]: r for r in labeling.list_clips(root)}
    assert by_id["span_a"]["scene_id"] is not None
    assert by_id["span_a"]["scene_id"] == by_id["span_b"]["scene_id"]
    assert by_id["span_a"]["scene_size"] == 2
    assert by_id["span_c"]["scene_id"] is None
    assert by_id["span_c"]["scene_size"] == 1


def test_present_tracks_maps_sibling_into_clip_timeline(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    cam = "6695ef21030c4603e400040d@20260606T094800Z"
    # Clip A: frame 0 at 09:49:20; source starts 09:49:00 (start_s=20).
    a = _make_clip_dir(
        root, "span_a", source_id=cam, track_id="1", n_det=2,
        span_start_utc="2026-06-06T09:49:20+00:00",
        span_end_utc="2026-06-06T09:49:30+00:00",
        source_start_utc="2026-06-06T09:49:00+00:00",
        start_s=20.0,
        det_time_s=[20.0, 20.1],
    )
    # Clip B (dog 2), same source recording; a detection at source t=20.5s
    # -> abs 09:49:20.5 -> clip-A frame round(0.5*30)=15. frame_count is 12, so
    # that one is out of A's window; t=20.0 -> frame 0 (in window).
    _make_clip_dir(
        root, "span_b", source_id=cam, track_id="2", n_det=2,
        span_start_utc="2026-06-06T09:49:20+00:00",
        span_end_utc="2026-06-06T09:49:30+00:00",
        source_start_utc="2026-06-06T09:49:00+00:00",
        start_s=20.0,
        det_time_s=[20.0, 20.5],
    )
    detail = labeling.clip_detail(a, root)
    assert detail["n_tracks"] == 2
    sib = detail["present_tracks"]["span_b:2"]
    assert sib["is_self"] is False
    assert sib["span_id"] == "span_b"
    # Only the in-window sibling detection (source t=20.0 -> A frame 0) is kept.
    assert [b["clip_frame_idx"] for b in sib["boxes"]] == [0]


def test_present_tracks_groups_file_harvest_siblings_without_at(
    tmp_path: Path,
) -> None:
    # File harvests use the sanitized filename as source_id (no ``@`` camera id),
    # so sibling grouping must fall back to the full source_id. Two spans from the
    # same recording should still see each other as present tracks.
    root = tmp_path / "harvest"
    src = "data_backyard_6_7_2026_clip.mp4"
    a = _make_clip_dir(
        root, "span_a", source_id=src, track_id="1", n_det=2,
        span_start_utc="2026-06-06T09:49:20+00:00",
        span_end_utc="2026-06-06T09:49:30+00:00",
        source_start_utc="2026-06-06T09:49:00+00:00",
        start_s=20.0,
        det_time_s=[20.0, 20.1],
    )
    _make_clip_dir(
        root, "span_b", source_id=src, track_id="2", n_det=2,
        span_start_utc="2026-06-06T09:49:20+00:00",
        span_end_utc="2026-06-06T09:49:30+00:00",
        source_start_utc="2026-06-06T09:49:00+00:00",
        start_s=20.0,
        det_time_s=[20.0, 20.1],
    )
    detail = labeling.clip_detail(a, root)
    assert detail["n_tracks"] == 2
    assert detail["present_tracks"]["span_b:2"]["is_self"] is False


def test_save_clip_labels_roundtrip_and_validation(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    clip_dir = _make_clip_dir(root, "span_a")
    payload = {
        "schema_version": "labels-1.0",
        "clip": "clip.mp4",
        "ranges": [
            {
                "start_frame": 2,
                "end_frame": 6,
                "start_s": 0.066,
                "end_s": 0.2,
                "behavior": "pee",
                "dog": "apollo",
                "track_id": "1",
            }
        ],
    }
    detail = labeling.save_clip_labels(clip_dir, payload)
    assert detail["labeled"] is True
    assert detail["behaviors"] == ["pee"]
    assert detail["dogs"] == ["apollo"]
    # Persisted to disk and reloads.
    again = labeling.clip_detail(clip_dir)
    assert again["labels"]["ranges"][0]["behavior"] == "pee"
    # Inverted range is rejected by the schema.
    bad = json.loads(json.dumps(payload))
    bad["ranges"][0]["end_frame"] = 1
    with pytest.raises(ValueError):
        labeling.save_clip_labels(clip_dir, bad)


# --- API ------------------------------------------------------------------


def test_list_clips_sorts_same_day_by_timestamp_not_span_id(tmp_path: Path) -> None:
    # Two unlabeled clips on the same day. Their span_id lexical order is the
    # reverse of chronological order, so a (day, span_id) string sort would put
    # the 09:00 clip first. The fix sorts on the full span_start_utc timestamp,
    # so the newer 11:00 clip must come first.
    root = tmp_path / "harvest"
    _make_clip_dir(
        root, "a_nine", date="2026-06-06",
        span_start_utc="2026-06-06T09:00:00+00:00",
    )
    _make_clip_dir(
        root, "z_eleven", date="2026-06-06",
        span_start_utc="2026-06-06T11:00:00+00:00",
    )
    rows = labeling.list_clips(root)
    assert [r["span_id"] for r in rows] == ["z_eleven", "a_nine"]


def test_api_list_clips(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a", date="2026-06-08")
    _make_clip_dir(root, "span_b", date="2026-06-09")
    client = _client(root, tmp_path)
    resp = client.get("/api/label/clips")
    assert resp.status_code == 200
    body = resp.json()
    assert {c["span_id"] for c in body["clips"]} == {"span_a", "span_b"}
    assert body["vocabulary"]["behaviors"] == ["pee", "poop", "not_potty", "excluded"]
    assert "apollo" in body["vocabulary"]["dogs"]


def test_api_clip_detail_and_save(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a")
    client = _client(root, tmp_path)

    detail = client.get("/api/label/clips/span_a")
    assert detail.status_code == 200
    assert detail.json()["span_id"] == "span_a"

    payload = {
        "ranges": [
            {
                "start_frame": 0,
                "end_frame": 4,
                "start_s": 0.0,
                "end_s": 0.13,
                "behavior": "poop",
                "dog": "gromit",
                "track_id": "1",
            }
        ]
    }
    saved = client.put("/api/label/clips/span_a/labels", json=payload)
    assert saved.status_code == 200
    assert saved.json()["behaviors"] == ["poop"]

    # Invalid enum -> 400 (not a 500).
    bad = client.put(
        "/api/label/clips/span_a/labels",
        json={"ranges": [{**payload["ranges"][0], "behavior": "nope"}]},
    )
    assert bad.status_code == 400


def test_api_unknown_clip_404_and_traversal(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a")
    client = _client(root, tmp_path)
    assert client.get("/api/label/clips/does_not_exist").status_code == 404
    # Path-traversal span id never resolves to a real dir.
    assert client.get("/api/label/clips/..%2F..").status_code in {400, 404}


def test_api_clip_video_streams(tmp_path: Path) -> None:
    root = tmp_path / "harvest"
    _make_clip_dir(root, "span_a")
    client = _client(root, tmp_path)
    resp = client.get("/api/label/clips/span_a/video")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/")
    assert len(resp.content) > 0
