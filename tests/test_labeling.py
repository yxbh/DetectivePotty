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
) -> Path:
    clip_dir = root / span_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    _write_clip(clip_dir / "clip.mp4")
    detections = [
        {
            "clip_frame_idx": i * 2,
            "source_frame_idx": 100 + i * 2,
            "time_s": float(i),
            "track_id": track_id,
            "bbox": {"x1": 10.0 + i, "y1": 12.0, "x2": 80.0 + i, "y2": 90.0},
            "confidence": 0.7,
        }
        for i in range(n_det)
    ]
    meta = {
        "schema_version": "harvest-1.0",
        "span_id": span_id,
        "source_id": "cam_backyard",
        "source_span_start_utc": f"{date}T17:29:38+00:00",
        "fps": 30.0,
        "frame_count": 12,
        "width": 160,
        "height": 120,
        "timebase": "clip_frames",
        "track_id": track_id,
        "detections": detections,
    }
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
