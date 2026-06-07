"""Potty candidate to dataset event recorder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from detectivepotty.config import CameraConfig, Config
from detectivepotty.events import ClassifierGuess, EventMetadata
from detectivepotty.geometry import BBox
from detectivepotty.potty_event import PottyCandidate
from detectivepotty.recording.clip_writer import write_frames_to_mp4
from detectivepotty.recording.dataset import (
    event_dir,
    sanitize_path_component,
    write_event_images,
)
from detectivepotty.recording.pose_overlay import write_pose_overlays
from detectivepotty.sources.base import Frame, sanitize_source_id
from detectivepotty.sources.rolling_buffer import RollingBuffer

LOGGER = logging.getLogger(__name__)


class EventRecorder:
    """Persist emitted potty candidates as dataset events."""

    def __init__(
        self,
        config: Config,
        protect_client: Any | None = None,
        *,
        git_commit: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self.config = config
        self.protect_client = protect_client
        self.git_commit = git_commit if git_commit is not None else _resolve_git_commit()
        self.model_version = model_version
        self.config_hash = config.config_hash()
        self.model_name = config.global_settings.model_name

    @staticmethod
    def assemble_window(
        buffer: RollingBuffer,
        candidate: PottyCandidate,
        camera_config: CameraConfig,
    ) -> list[Frame]:
        start = candidate.start_ts - timedelta(seconds=camera_config.pre_roll_s)
        end = candidate.end_ts + timedelta(seconds=camera_config.post_roll_s)
        return buffer.get_window(start, end)

    def record(
        self,
        candidate: PottyCandidate,
        frames: Sequence[Frame],
        camera_config: CameraConfig,
        *,
        classifier_result: Any | None = None,
        protect_meta: Mapping[str, Any] | None = None,
    ) -> Path:
        frame_list = list(frames)
        event_id = _metadata_event_id(protect_meta)
        metadata = self._metadata(
            candidate,
            frame_list,
            camera_config,
            classifier_result,
            protect_meta,
            event_id,
        )
        target_dir = event_dir(
            self.config.global_settings.dataset_dir,
            camera_config.id,
            camera_config.name,
            candidate.start_ts,
            candidate.primary_track_id,
            metadata.event_id,
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        write_frames_to_mp4(frame_list, target_dir / "clip.mp4")
        frame_records, crop_records = write_event_images(
            target_dir,
            frame_list,
            candidate.detections,
            candidate.tracks,
            candidate.primary_track_id,
            substream=camera_config.substream_choice,
        )
        metadata.frame_records = frame_records
        metadata.crop_boxes = crop_records
        self._write_pose_overlays(target_dir, crop_records, frame_records, classifier_result)
        metadata.write_json(target_dir)
        return target_dir

    def _write_pose_overlays(
        self,
        target_dir: Path,
        crop_records: Sequence[Any],
        frame_records: Sequence[Any],
        classifier_result: Any | None,
    ) -> None:
        poses = _value_from_obj(classifier_result, "poses")
        if not poses:
            return
        try:
            write_pose_overlays(
                target_dir,
                crop_records,
                frame_records,
                poses,
                min_conf=self.config.pose.min_keypoint_conf,
            )
        except Exception:
            LOGGER.warning(
                "Pose overlay generation failed for event %s", target_dir.name
            )

    async def maybe_download_protect_recording(
        self,
        candidate: PottyCandidate,
        camera_config: CameraConfig,
        target_event_dir: str | Path,
    ) -> Path | None:
        if self.protect_client is None:
            return None

        start = candidate.start_ts - timedelta(seconds=camera_config.pre_roll_s)
        end = candidate.end_ts + timedelta(seconds=camera_config.post_roll_s)
        dest = Path(target_event_dir) / "protect_recording.mp4"
        try:
            return await self.protect_client.download_recording(
                camera_config.id,
                start,
                end,
                dest,
            )
        except Exception:
            safe_camera = sanitize_path_component(camera_config.id)
            LOGGER.warning("Protect recording download failed for camera %s", safe_camera)
            return None

    def _metadata(
        self,
        candidate: PottyCandidate,
        frames: Sequence[Frame],
        camera_config: CameraConfig,
        classifier_result: Any | None,
        protect_meta: Mapping[str, Any] | None,
        event_id: str,
    ) -> EventMetadata:
        classifier_guess, classifier_confidence = _classifier_fields(classifier_result)
        detection_ts = _datetime_from_meta(protect_meta, "detection_ts")
        notification_ts = _datetime_from_meta(protect_meta, "notification_ts")
        source_id = _source_id(frames, camera_config)
        return EventMetadata(
            event_id=event_id,
            camera_id=sanitize_source_id(camera_config.id),
            camera_name=sanitize_source_id(camera_config.name),
            sanitized_source_id=source_id,
            utc_ts=candidate.start_ts,
            protect_event_id=_str_from_meta(protect_meta, "protect_event_id"),
            smartdetect_score=_float_from_meta(protect_meta, "smartdetect_score"),
            smartdetect_bbox=_bbox_from_meta(protect_meta, "smartdetect_bbox"),
            detection_ts=detection_ts,
            notification_ts=notification_ts,
            model_name=self.model_name,
            model_version=self.model_version,
            config_hash=self.config_hash,
            git_commit=self.git_commit,
            trigger_reason=candidate.trigger_reason,
            pre_roll_s=camera_config.pre_roll_s,
            post_roll_s=camera_config.post_roll_s,
            detections=list(candidate.detections),
            tracks=list(candidate.tracks),
            multi_dog=candidate.multi_dog,
            ambiguous=candidate.ambiguous,
            classifier_guess=classifier_guess,
            classifier_confidence=classifier_confidence,
            extra=_candidate_extra(candidate, classifier_result),
        )


def _metadata_event_id(protect_meta: Mapping[str, Any] | None) -> str:
    if protect_meta is not None and protect_meta.get("event_id"):
        return str(protect_meta["event_id"])
    return str(uuid4())


def _source_id(frames: Sequence[Frame], camera_config: CameraConfig) -> str:
    if frames:
        return sanitize_source_id(frames[0].source_id)
    if camera_config.input.source_id:
        return sanitize_source_id(camera_config.input.source_id)
    if camera_config.input.path is not None:
        return sanitize_source_id(str(camera_config.input.path))
    return sanitize_source_id(camera_config.id)


def _candidate_extra(
    candidate: PottyCandidate,
    classifier_result: Any | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "primary_track_id": candidate.primary_track_id,
        "lifecycle": candidate.lifecycle.value,
        "stationary_duration_s": candidate.stationary_duration_s,
        "squat_metric": candidate.squat_metric,
        "posture_summary": candidate.posture_summary,
        "near_miss": candidate.near_miss,
        "candidate_confidence": candidate.confidence,
    }
    needs_label = _value_from_obj(classifier_result, "needs_label")
    if needs_label is not None:
        extra["classifier_needs_label"] = bool(needs_label)
    pose = _pose_extra(classifier_result)
    if pose is not None:
        extra["pose"] = pose
    return extra


def _pose_extra(classifier_result: Any | None) -> dict[str, Any] | None:
    """Serialize pose keypoints + features carried on the classifier result."""

    poses = _value_from_obj(classifier_result, "poses")
    features = _value_from_obj(classifier_result, "pose_features")
    if not poses and features is None:
        return None
    pose: dict[str, Any] = {}
    if features is not None and hasattr(features, "to_dict"):
        pose["features"] = features.to_dict()
    if poses:
        pose["keypoints"] = [
            keypoints.to_dict() for keypoints in poses if hasattr(keypoints, "to_dict")
        ]
    return pose or None


def _classifier_fields(
    classifier_result: Any | None,
) -> tuple[ClassifierGuess, float | None]:
    guess_value = _value_from_obj(classifier_result, "guess")
    confidence_value = _value_from_obj(classifier_result, "confidence")
    if guess_value is None:
        guess = ClassifierGuess.UNKNOWN
    elif isinstance(guess_value, ClassifierGuess):
        guess = guess_value
    else:
        try:
            guess = ClassifierGuess(str(guess_value))
        except ValueError:
            guess = ClassifierGuess.UNKNOWN

    confidence: float | None = None
    if confidence_value is not None:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = None
    return guess, confidence


def _value_from_obj(value: Any | None, key: str) -> Any | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _str_from_meta(meta: Mapping[str, Any] | None, key: str) -> str | None:
    value = _meta_value(meta, key)
    if value is None:
        return None
    return str(value)


def _float_from_meta(meta: Mapping[str, Any] | None, key: str) -> float | None:
    value = _meta_value(meta, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _datetime_from_meta(meta: Mapping[str, Any] | None, key: str) -> datetime | None:
    value = _meta_value(meta, key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _bbox_from_meta(meta: Mapping[str, Any] | None, key: str) -> BBox | None:
    value = _meta_value(meta, key)
    if value is None or isinstance(value, BBox):
        return value
    try:
        if isinstance(value, Mapping):
            return BBox(
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            x1, y1, x2, y2 = value
            return BBox(float(x1), float(y1), float(x2), float(y2))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _meta_value(meta: Mapping[str, Any] | None, key: str) -> Any | None:
    if meta is None:
        return None
    return meta.get(key)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None
