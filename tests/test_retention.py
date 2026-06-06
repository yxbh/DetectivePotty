from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import time

from detectivepotty.config import CameraConfig
from detectivepotty.recording.dataset import event_dir
from detectivepotty.recording.retention import enforce_retention

NOW = datetime.now(timezone.utc)


def camera_config(**overrides: object) -> CameraConfig:
    values = {
        "id": "cam-1",
        "name": "Backyard Grass",
        "retention_days": 7,
    }
    values.update(overrides)
    return CameraConfig(**values)


def make_event(tmp_path, camera: CameraConfig, age_days: int, event_id: str, size: int):
    start = NOW - timedelta(days=age_days)
    path = event_dir(tmp_path, camera.id, camera.name, start, "dog-1", event_id)
    path.mkdir(parents=True)
    (path / "payload.bin").write_bytes(b"x" * size)
    mtime = time.time() - (age_days * 24 * 60 * 60)
    os.utime(path / "payload.bin", (mtime, mtime))
    os.utime(path, (mtime, mtime))
    return path


def test_enforce_retention_deletes_old_events_and_keeps_boundary(tmp_path) -> None:
    camera = camera_config(retention_days=7)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    old_event = make_event(tmp_path, camera, 10, "old", 10)
    new_event = make_event(tmp_path, camera, 1, "new", 10)
    try:
        (old_event / "outside-link").symlink_to(outside)
    except OSError:
        pass
    old_mtime = time.time() - (10 * 24 * 60 * 60)
    os.utime(old_event, (old_mtime, old_mtime))

    summary = enforce_retention(tmp_path, camera)

    assert summary.deleted_events == 1
    assert summary.deleted_bytes >= 10
    assert not old_event.exists()
    assert new_event.exists()
    assert outside.exists()


def test_enforce_retention_deletes_oldest_until_under_size_cap(tmp_path) -> None:
    camera = camera_config(
        retention_days=365,
        retention_max_gb=25 / 1024**3,
    )
    oldest = make_event(tmp_path, camera, 30, "oldest", 10)
    middle = make_event(tmp_path, camera, 20, "middle", 10)
    newest = make_event(tmp_path, camera, 10, "newest", 10)

    summary = enforce_retention(tmp_path, camera)

    assert summary.deleted_events == 1
    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()
    assert summary.remaining_bytes <= 25


def test_enforce_retention_ignores_missing_camera_dir(tmp_path) -> None:
    summary = enforce_retention(tmp_path, camera_config())

    assert summary.deleted_events == 0
    assert summary.deleted_bytes == 0
    assert summary.remaining_bytes == 0
