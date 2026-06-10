"""Potty candidate to dataset event recorder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, NamedTuple
from uuid import uuid4

from detectivepotty.config import CameraConfig, Config
from detectivepotty.events import ClassifierGuess, EventMetadata, Label, LabelStatus
from detectivepotty.geometry import BBox
from detectivepotty.potty_event import PottyCandidate
from detectivepotty.recording.clip_writer import write_frames_to_mp4
from detectivepotty.recording.dataset import (
    camera_dataset_dir,
    event_dir,
    sanitize_path_component,
    write_event_images,
)
from detectivepotty.recording.pose_overlay import write_pose_overlays
from detectivepotty.recording.reconcile import (
    PriorEvent,
    ReconcileResult,
    decide_carry,
    match_priors,
    paths_equal,
    preserve_protect_recording,
    remove_event_dir,
    snapshot_prior_events,
)
from detectivepotty.sources.base import Frame, sanitize_source_id
from detectivepotty.sources.file import derive_base_wall_ts
from detectivepotty.sources.rolling_buffer import RollingBuffer

LOGGER = logging.getLogger(__name__)


class _SourceTimeline(NamedTuple):
    """Deterministic source-relative timing for a recorded event.

    ``time_basis`` mirrors the file source's anchor derivation; the offsets are
    seconds from that anchor and are anchor-independent across reruns, so they
    can dedupe reruns even if the wall-clock anchor ever changed.
    """

    time_basis: str | None
    source_start_s: float | None
    source_end_s: float | None


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
        # Per-run snapshot of prior on-disk events, keyed by (camera_id,
        # source_id). Built lazily on first record() for a key so reruns match
        # only against earlier runs, never events this run just wrote.
        self._snapshots: dict[tuple[str, str], list[PriorEvent]] = {}

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
        source_id = _source_id(frame_list, camera_config)
        protect_event_id = _str_from_meta(protect_meta, "protect_event_id")
        default_event_id = _metadata_event_id(protect_meta)

        timeline = _source_timeline(candidate, camera_config)

        result = self._reconcile(
            camera_config, candidate, source_id, protect_event_id, timeline
        )
        event_id = result.event_id or default_event_id

        metadata = self._metadata(
            candidate,
            frame_list,
            camera_config,
            classifier_result,
            protect_meta,
            event_id,
            timeline,
        )
        _apply_carried(metadata, result.carried)
        target_dir = event_dir(
            self.config.global_settings.dataset_dir,
            camera_config.id,
            camera_config.name,
            candidate.start_ts,
            candidate.primary_track_id,
            metadata.event_id,
        )

        # Stage any in-place collision out of the way so writing the fresh event
        # can never partially overwrite a prior we are about to supersede.
        to_delete = self._prepare_supersede(result.superseded, target_dir)
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

        self._finalize_supersede(to_delete, target_dir, camera_config, source_id, result.superseded)
        return target_dir

    def _reconcile(
        self,
        camera_config: CameraConfig,
        candidate: PottyCandidate,
        source_id: str,
        protect_event_id: str | None,
        timeline: _SourceTimeline,
    ) -> ReconcileResult:
        if not self.config.global_settings.dedupe_reruns:
            return ReconcileResult()
        snapshot = self._snapshot_for(camera_config, source_id)
        matched = match_priors(
            snapshot,
            start_ts=candidate.start_ts,
            end_ts=candidate.end_ts,
            protect_event_id=protect_event_id,
            tolerance_s=self.config.global_settings.rerun_match_tolerance_s,
            source_start_s=timeline.source_start_s,
            source_end_s=timeline.source_end_s,
        )
        return decide_carry(matched)

    def _snapshot_for(
        self,
        camera_config: CameraConfig,
        source_id: str,
    ) -> list[PriorEvent]:
        camera_id = sanitize_source_id(camera_config.id)
        key = (camera_id, source_id)
        if key not in self._snapshots:
            camera_dir = camera_dataset_dir(
                self.config.global_settings.dataset_dir,
                camera_config.id,
                camera_config.name,
            )
            self._snapshots[key] = snapshot_prior_events(camera_dir, camera_id, source_id)
        return self._snapshots[key]

    def _prepare_supersede(
        self,
        superseded: Sequence[PriorEvent],
        target_dir: Path,
    ) -> list[tuple[PriorEvent, Path]]:
        to_delete: list[tuple[PriorEvent, Path]] = []
        for prior in superseded:
            delete_path = prior.dir_path
            if paths_equal(prior.dir_path, target_dir):
                backup = prior.dir_path.with_name(prior.dir_path.name + ".rerun-bak")
                try:
                    if backup.exists():
                        remove_event_dir(backup)
                    os.replace(prior.dir_path, backup)
                    delete_path = backup
                except OSError:
                    delete_path = prior.dir_path
            to_delete.append((prior, delete_path))
        return to_delete

    def _finalize_supersede(
        self,
        to_delete: Sequence[tuple[PriorEvent, Path]],
        target_dir: Path,
        camera_config: CameraConfig,
        source_id: str,
        superseded: Sequence[PriorEvent],
    ) -> None:
        for _prior, path in to_delete:
            if paths_equal(path, target_dir):
                continue
            preserve_protect_recording(path, target_dir)
            remove_event_dir(path)
        if superseded:
            self._drop_from_snapshot(camera_config, source_id, superseded)

    def _drop_from_snapshot(
        self,
        camera_config: CameraConfig,
        source_id: str,
        superseded: Sequence[PriorEvent],
    ) -> None:
        key = (sanitize_source_id(camera_config.id), source_id)
        snapshot = self._snapshots.get(key)
        if not snapshot:
            return
        removed = {id(prior) for prior in superseded}
        self._snapshots[key] = [prior for prior in snapshot if id(prior) not in removed]

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
        timeline: _SourceTimeline,
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
            end_ts=candidate.end_ts,
            recorded_at=datetime.now(timezone.utc),
            source_start_s=timeline.source_start_s,
            source_end_s=timeline.source_end_s,
            time_basis=timeline.time_basis,
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


def _source_timeline(
    candidate: PottyCandidate,
    camera_config: CameraConfig,
) -> _SourceTimeline:
    """Derive deterministic source-relative timing for a file-backed event.

    For non-file (live/Protect) cameras there is no input path, so the offsets
    and basis are ``None`` and the event keeps its real Protect timestamps. For
    file cameras the anchor is re-derived from the same path the
    :class:`FileSource` used, so ``candidate.start_ts - base`` recovers the exact
    in-clip offset. When the anchor can only fall back to ``runtime_now`` (a path
    that cannot be stat-ed) the offsets are dropped — that base is not stable, so
    persisting offsets against it would be misleading — but the basis is kept so
    the UI can flag the timestamp as approximate.
    """

    path = camera_config.input.path
    if path is None:
        return _SourceTimeline(None, None, None)
    base_ts, basis = derive_base_wall_ts(path)
    if basis == "runtime_now":
        return _SourceTimeline(basis, None, None)
    start_s = (candidate.start_ts - base_ts).total_seconds()
    end_s = (candidate.end_ts - base_ts).total_seconds()
    return _SourceTimeline(basis, start_s, end_s)


def _apply_carried(metadata: EventMetadata, carried: Mapping[str, Any]) -> None:
    """Apply human label fields carried forward from a superseded prior event."""

    if not carried:
        return
    label_value = carried.get("label")
    if label_value is not None:
        metadata.label = _coerce_label(label_value)
    status_value = carried.get("label_status")
    if status_value is not None:
        metadata.label_status = _coerce_status(status_value)
    if "dog" in carried:
        metadata.dog = carried.get("dog")
    note = carried.get("label_note")
    if note is not None:
        metadata.extra["label_note"] = note
    labeled_at = carried.get("labeled_at")
    if labeled_at is not None:
        metadata.extra["labeled_at"] = labeled_at


def _coerce_label(value: Any) -> Label:
    try:
        return Label(str(value))
    except ValueError:
        return Label.UNKNOWN


def _coerce_status(value: Any) -> LabelStatus:
    try:
        return LabelStatus(str(value))
    except ValueError:
        return LabelStatus.UNLABELED


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
