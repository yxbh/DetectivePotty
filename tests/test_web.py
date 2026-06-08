from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np

from detectivepotty.config import Config, GlobalSettings
from detectivepotty.events import ClassifierGuess, EventMetadata, Label, LabelStatus, TriggerReason
from detectivepotty.web import create_app


BASE_TS = datetime(2026, 6, 6, 9, 10, 47, tzinfo=timezone.utc)


def make_client(dataset_dir: Path, dogs: list[str] | None = None) -> TestClient:
    config = Config(
        global_settings=GlobalSettings(dataset_dir=dataset_dir, dogs=dogs or []),
    )
    return TestClient(create_app(config))


def write_image(path: Path, color: tuple[int, int, int]) -> bytes:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = color
    assert cv2.imwrite(str(path), image)
    return path.read_bytes()


def make_event(
    dataset_dir: Path,
    *,
    event_id: str,
    camera: str,
    utc_ts: datetime,
    label_status: LabelStatus = LabelStatus.UNLABELED,
    label: Label = Label.UNKNOWN,
    protect: bool = False,
) -> Path:
    dir_ts = utc_ts.strftime("%Y%m%dT%H%M%SZ")
    event_dir = (
        dataset_dir
        / camera
        / utc_ts.date().isoformat()
        / "events"
        / f"{dir_ts}_{camera}_track_{event_id}"
    )
    (event_dir / "frames").mkdir(parents=True)
    (event_dir / "crops").mkdir()
    (event_dir / "clip.mp4").write_bytes(b"fake mp4 clip")
    if protect:
        (event_dir / "protect_recording.mp4").write_bytes(b"fake protect clip")
    write_image(event_dir / "frames" / "000.jpg", (0, 0, 255))
    write_image(event_dir / "frames" / "001.jpg", (0, 255, 0))
    write_image(event_dir / "crops" / "000.jpg", (255, 0, 0))

    metadata = EventMetadata(
        event_id=event_id,
        camera_id=f"{camera}-id",
        camera_name=camera,
        sanitized_source_id=f"{camera}-source",
        utc_ts=utc_ts,
        detection_ts=utc_ts,
        notification_ts=utc_ts + timedelta(seconds=1.25),
        trigger_reason=TriggerReason.YOLO,
        model_name="test-model",
        classifier_guess=ClassifierGuess.PEE,
        classifier_confidence=0.82,
        label_status=label_status,
        label=label,
        extra={"posture_summary": "stationary squat"},
    ).to_dict()
    with (event_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return event_dir


def test_events_list_sorting_and_filters(tmp_path: Path) -> None:
    make_event(
        tmp_path,
        event_id="older",
        camera="Backyard",
        utc_ts=BASE_TS,
        label_status=LabelStatus.UNLABELED,
    )
    make_event(
        tmp_path,
        event_id="newer",
        camera="Sideyard",
        utc_ts=BASE_TS + timedelta(hours=1),
        label_status=LabelStatus.LABELED,
        label=Label.PEE,
        protect=True,
    )
    client = make_client(tmp_path)

    response = client.get("/api/events")

    assert response.status_code == 200
    assert response.headers["x-total-count"] == "2"
    assert response.headers["x-unfiltered-count"] == "2"
    assert response.headers["cache-control"] == "no-store"
    events = response.json()
    assert [event["event_id"] for event in events] == ["newer", "older"]
    assert events[0]["camera"] == "Sideyard"
    assert "dog" in events[0]
    assert events[0]["classifier_guess"] == "pee"
    assert events[0]["classifier_confidence"] == 0.82
    assert events[0]["label"] == "pee"
    assert events[0]["label_status"] == "labeled"
    assert events[0]["thumbnail_url"].endswith("/crops/000.jpg")
    assert events[0]["frames_count"] == 2
    assert events[0]["crops_count"] == 1
    assert events[0]["protect_recording_exists"] is True
    assert events[0]["relative_dir"].startswith("Sideyard/2026-06-06/events/")

    unlabeled = client.get("/api/events", params={"label_status": "unlabeled"})
    assert [event["event_id"] for event in unlabeled.json()] == ["older"]

    by_camera = client.get("/api/events", params={"camera": "Sideyard"})
    assert [event["event_id"] for event in by_camera.json()] == ["newer"]

    by_date = client.get("/api/events", params={"date": "2026-06-06"})
    assert [event["event_id"] for event in by_date.json()] == ["newer", "older"]

    paged = client.get("/api/events", params={"limit": 1, "offset": 1})
    assert [event["event_id"] for event in paged.json()] == ["older"]


def test_event_detail_and_media_serving_block_traversal(tmp_path: Path) -> None:
    event_dir = make_event(
        tmp_path,
        event_id="event-a",
        camera="Backyard",
        utc_ts=BASE_TS,
        protect=True,
    )
    client = make_client(tmp_path)

    detail_response = client.get("/api/events/event-a")

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["metadata"]["event_id"] == "event-a"
    assert detail["summary"]["event_id"] == "event-a"
    assert detail["media"]["clip"] == "/api/events/event-a/media/clip"
    assert detail["media"]["protect_recording"] == "/api/events/event-a/media/protect"
    assert [frame["name"] for frame in detail["media"]["frames"]] == ["000.jpg", "001.jpg"]
    assert [crop["name"] for crop in detail["media"]["crops"]] == ["000.jpg"]

    frame_response = client.get("/api/events/event-a/frames/000.jpg")
    assert frame_response.status_code == 200
    assert frame_response.content == (event_dir / "frames" / "000.jpg").read_bytes()

    crop_response = client.get("/api/events/event-a/crops/000.jpg")
    assert crop_response.status_code == 200
    assert crop_response.content == (event_dir / "crops" / "000.jpg").read_bytes()

    clip_response = client.get("/api/events/event-a/media/clip")
    assert clip_response.status_code == 200
    assert clip_response.content == b"fake mp4 clip"

    traversal_response = client.get("/api/events/event-a/frames/..%2f..%2fmetadata.json")
    assert traversal_response.status_code in {400, 404}
    assert traversal_response.content != (event_dir / "metadata.json").read_bytes()

    assert client.get("/api/events/missing").status_code == 404


def test_event_detail_exposes_pose_overlay_media(tmp_path: Path) -> None:
    event_dir = make_event(
        tmp_path,
        event_id="event-pose",
        camera="Backyard",
        utc_ts=BASE_TS,
    )
    overlay_dir = event_dir / "crops_overlay"
    overlay_dir.mkdir()
    overlay_bytes = write_image(overlay_dir / "000.jpg", (10, 200, 255))
    client = make_client(tmp_path)

    detail = client.get("/api/events/event-pose").json()
    assert [item["name"] for item in detail["media"]["crops_overlay"]] == ["000.jpg"]
    assert (
        detail["media"]["crops_overlay"][0]["url"]
        == "/api/events/event-pose/crops_overlay/000.jpg"
    )

    overlay_response = client.get("/api/events/event-pose/crops_overlay/000.jpg")
    assert overlay_response.status_code == 200
    assert overlay_response.content == overlay_bytes

    traversal = client.get("/api/events/event-pose/crops_overlay/..%2f..%2fmetadata.json")
    assert traversal.status_code in {400, 404}


def test_label_update_validates_and_writes_metadata_atomically(tmp_path: Path) -> None:
    event_dir = make_event(
        tmp_path,
        event_id="label-me",
        camera="Backyard",
        utc_ts=BASE_TS,
    )
    client = make_client(tmp_path)

    response = client.post(
        "/api/events/label-me/label",
        json={"label": "poop", "label_status": "labeled", "note": "clear squat"},
    )

    assert response.status_code == 200
    summary = response.json()
    assert summary["label"] == "poop"
    assert summary["label_status"] == "labeled"

    metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["event_id"] == "label-me"
    assert metadata["classifier_guess"] == "pee"
    assert metadata["label"] == "poop"
    assert metadata["label_status"] == "labeled"
    assert metadata["extra"]["label_note"] == "clear squat"
    assert "labeled_at" in metadata["extra"]
    assert list(event_dir.glob(".metadata.*.tmp")) == []

    invalid = client.post(
        "/api/events/label-me/label",
        json={"label": "bad", "label_status": "labeled"},
    )
    assert invalid.status_code == 422

    missing = client.post(
        "/api/events/missing/label",
        json={"label": "pee", "label_status": "labeled"},
    )
    assert missing.status_code == 404


def test_root_serves_html(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "DetectivePotty Review" in response.text


def test_root_serves_build_missing_fallback(tmp_path: Path, monkeypatch) -> None:
    from detectivepotty.web import app as app_module

    monkeypatch.setattr(app_module, "FRONTEND_DIST", tmp_path / "no-build")
    client = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "DetectivePotty Review" in response.text
    assert "npm run build" in response.text


def test_root_serves_built_index(tmp_path: Path, monkeypatch) -> None:
    from detectivepotty.web import app as app_module

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!doctype html><title>DetectivePotty Review</title><div id=app></div>",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "FRONTEND_DIST", dist)
    client = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "id=app" in response.text


def test_dogs_roster_endpoint(tmp_path: Path) -> None:
    client = make_client(tmp_path, dogs=["Gromit", "WALL-E", "Apollo"])

    response = client.get("/api/dogs")

    assert response.status_code == 200
    assert response.json() == {"dogs": ["Gromit", "WALL-E", "Apollo"]}


def test_label_update_sets_and_preserves_dog(tmp_path: Path) -> None:
    event_dir = make_event(tmp_path, event_id="dog-evt", camera="Backyard", utc_ts=BASE_TS)
    client = make_client(tmp_path, dogs=["Gromit", "WALL-E", "Apollo"])

    set_resp = client.post(
        "/api/events/dog-evt/label",
        json={"label": "pee", "label_status": "labeled", "dog": "WALL-E"},
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["dog"] == "WALL-E"
    assert json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))["dog"] == "WALL-E"

    # A later label-only update (dog omitted) must preserve the dog.
    keep_resp = client.post(
        "/api/events/dog-evt/label",
        json={"label": "poop", "label_status": "labeled"},
    )
    assert keep_resp.status_code == 200
    assert keep_resp.json()["dog"] == "WALL-E"
    kept = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    assert kept["dog"] == "WALL-E"
    assert kept["label"] == "poop"

    # Explicit null clears the dog.
    clear_resp = client.post(
        "/api/events/dog-evt/label",
        json={"label": "poop", "label_status": "labeled", "dog": None},
    )
    assert clear_resp.status_code == 200
    assert clear_resp.json()["dog"] is None
    assert json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))["dog"] is None


def test_label_update_rejects_unknown_dog_with_roster(tmp_path: Path) -> None:
    make_event(tmp_path, event_id="dog-evt", camera="Backyard", utc_ts=BASE_TS)
    client = make_client(tmp_path, dogs=["Gromit", "WALL-E", "Apollo"])

    response = client.post(
        "/api/events/dog-evt/label",
        json={"label": "pee", "label_status": "labeled", "dog": "Scooby"},
    )

    assert response.status_code == 422


def test_label_update_allows_freeform_dog_without_roster(tmp_path: Path) -> None:
    make_event(tmp_path, event_id="dog-evt", camera="Backyard", utc_ts=BASE_TS)
    client = make_client(tmp_path)

    response = client.post(
        "/api/events/dog-evt/label",
        json={"label": "pee", "label_status": "labeled", "dog": "Anything"},
    )

    assert response.status_code == 200
    assert response.json()["dog"] == "Anything"


def test_stream_pushes_new_events(tmp_path: Path) -> None:
    import asyncio

    from detectivepotty.web.app import _event_stream
    from detectivepotty.web.dataset_index import DatasetIndex

    make_event(tmp_path, event_id="seed", camera="Backyard", utc_ts=BASE_TS)
    index = DatasetIndex(tmp_path)

    state = {"checks": 0}

    async def is_disconnected() -> bool:
        state["checks"] += 1
        # Create a brand-new event only after the stream has connected and
        # seeded, so it must arrive via a diff (not the initial backfill).
        if state["checks"] == 1:
            make_event(
                tmp_path,
                event_id="fresh",
                camera="Sideyard",
                utc_ts=BASE_TS + timedelta(hours=2),
            )
        return state["checks"] >= 3

    async def no_sleep(_seconds: float) -> None:
        return None

    async def drive() -> list[str]:
        return [
            chunk async for chunk in _event_stream(index, is_disconnected, sleep=no_sleep)
        ]

    text = "".join(asyncio.run(drive()))

    assert "event: ready" in text
    assert text.count("event: new") == 1
    assert '"event_id": "fresh"' in text
    assert '"camera": "Sideyard"' in text
    # The pre-existing event was seeded as known and must not be re-emitted.
    assert '"event_id": "seed"' not in text
