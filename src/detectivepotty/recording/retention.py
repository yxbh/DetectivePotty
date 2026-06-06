"""Per-camera dataset retention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

from detectivepotty.config import CameraConfig
from detectivepotty.recording.dataset import camera_dataset_dir


@dataclass(frozen=True, slots=True)
class RetentionSummary:
    camera_dir: Path
    deleted_events: int = 0
    deleted_bytes: int = 0
    remaining_bytes: int = 0


def enforce_retention(dataset_dir: str | Path, camera_config: CameraConfig) -> RetentionSummary:
    """Delete old events for one camera and enforce its optional disk cap."""

    camera_root = camera_dataset_dir(dataset_dir, camera_config.id, camera_config.name)
    if not camera_root.exists() or not camera_root.is_dir() or camera_root.is_symlink():
        return RetentionSummary(camera_dir=camera_root)

    root_resolved = camera_root.resolve()
    deleted_events = 0
    deleted_bytes = 0
    cutoff = time.time() - (camera_config.retention_days * 24 * 60 * 60)

    for event_path in _event_dirs(camera_root, root_resolved):
        if event_path.stat(follow_symlinks=False).st_mtime >= cutoff:
            continue
        size = _tree_size(event_path)
        _remove_tree(event_path)
        deleted_events += 1
        deleted_bytes += size

    remaining = _tree_size(camera_root)
    cap_gb = camera_config.retention_max_gb
    if cap_gb is not None:
        cap_bytes = int(cap_gb * 1024**3)
        for event_path in _event_dirs(camera_root, root_resolved):
            if remaining <= cap_bytes:
                break
            size = _tree_size(event_path)
            _remove_tree(event_path)
            deleted_events += 1
            deleted_bytes += size
            remaining = max(0, remaining - size)
        remaining = _tree_size(camera_root)

    return RetentionSummary(
        camera_dir=camera_root,
        deleted_events=deleted_events,
        deleted_bytes=deleted_bytes,
        remaining_bytes=remaining,
    )


def _event_dirs(camera_root: Path, root_resolved: Path) -> list[Path]:
    events: list[Path] = []
    for events_dir in camera_root.glob("*/events"):
        if not events_dir.is_dir() or events_dir.is_symlink():
            continue
        for event_path in events_dir.iterdir():
            if event_path.is_symlink() or not event_path.is_dir():
                continue
            if not _is_relative_to(event_path.resolve(), root_resolved):
                continue
            events.append(event_path)
    return sorted(events, key=lambda path: path.stat(follow_symlinks=False).st_mtime)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _tree_size(path: Path) -> int:
    try:
        stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return 0
    if path.is_symlink() or not path.is_dir():
        return stat.st_size

    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += _tree_size(entry_path)
                else:
                    total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    return total


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return
    with os.scandir(path) as entries:
        for entry in entries:
            child = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                _remove_tree(child)
            else:
                child.unlink(missing_ok=True)
    path.rmdir()
