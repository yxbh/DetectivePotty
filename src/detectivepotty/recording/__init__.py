"""Dataset event recording helpers."""

from detectivepotty.recording.clip_writer import write_frames_to_mp4
from detectivepotty.recording.dataset import (
    DEFAULT_CROP_MARGIN_FRAC,
    camera_dataset_dir,
    event_dir,
    format_event_timestamp,
    sanitize_path_component,
    write_event_images,
)
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.recording.retention import RetentionSummary, enforce_retention

__all__ = [
    "DEFAULT_CROP_MARGIN_FRAC",
    "EventRecorder",
    "RetentionSummary",
    "camera_dataset_dir",
    "enforce_retention",
    "event_dir",
    "format_event_timestamp",
    "sanitize_path_component",
    "write_event_images",
    "write_frames_to_mp4",
]
