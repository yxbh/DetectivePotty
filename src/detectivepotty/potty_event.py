"""Trigger-agnostic potty-candidate state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from pathlib import Path
from typing import Any

from detectivepotty.config import CameraConfig, ZoneConfig
from detectivepotty.events import Detection, Track, TriggerReason
from detectivepotty.pose.gate import PoseGate
from detectivepotty.sources.base import Frame
from detectivepotty.tracking import Tracker, iou

# Float/sampling tolerance for the "covered long enough" comparison against
# ``stationary_threshold_s``. The trailing posture window is defined as detections
# newer than ``current - threshold``, so its span is structurally capped at the
# threshold and -- with discrete sampling and intermittent (e.g. night-time)
# detections -- routinely lands up to a sample interval (or a dropped boundary
# detection) below it. A strict ``span >= threshold`` check therefore almost never
# fired on real footage. The effective tolerance is sized per camera (see
# ``_posture_stats``); this constant is only the float-noise floor.
_DURATION_TOLERANCE_S = 1e-3


class PottyLifecycle(str, Enum):
    """Lifecycle value carried by emitted potty candidates."""

    CANDIDATE = "candidate"
    EMITTED = "emitted"
    NEAR_MISS = "near_miss"


class _DetectorState(str, Enum):
    IDLE = "idle"
    WATCHING = "watching"
    CANDIDATE = "candidate"


@dataclass(slots=True)
class PottyCandidate:
    """Camera/time-window-centric generic potty candidate.

    ``tracks`` are the contributing track histories clipped to this event window;
    ``detections`` contains every in-window dog detection that survived config
    filtering. ``ambiguous`` is true when more than one dog/track contributed or
    boxes overlapped enough that ID swaps are plausible.
    """

    camera_id: str
    primary_track_id: str
    start_ts: datetime
    end_ts: datetime
    tracks: list[Track]
    detections: list[Detection]
    trigger_reason: TriggerReason
    multi_dog: bool
    ambiguous: bool
    lifecycle: PottyLifecycle
    stationary_duration_s: float
    posture_summary: dict[str, Any] = field(default_factory=dict)
    near_miss: bool = False
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.start_ts = _ensure_aware_utc(self.start_ts)
        self.end_ts = _ensure_aware_utc(self.end_ts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for recorder metadata."""

        return _jsonify(self)


@dataclass(slots=True)
class _PostureStats:
    track_id: str
    stationary_duration_s: float
    max_centroid_motion_px: float
    centroid_motion_threshold_px: float
    is_stationary: bool
    window_start_mono: float
    window_end_mono: float
    pose_summary: dict[str, Any] | None = None


class PottyEventDetector:
    """State machine that emits generic potty candidates for one camera.

    Trigger: a non-suppressed track that reads stationary continuously for at least
    ``CameraConfig.dwell_trigger_s`` becomes a candidate (a viewpoint-invariant
    sustained-dwell cue). Stationary metric: over the trailing
    ``stationary_threshold_s`` window the dog's detections must cover most of that
    window (within a per-camera sampling tolerance, so intermittent night-time
    detections still qualify) and the maximum center displacement from the first
    center must be no more than ``max(10px, 20% of the median bbox diagonal)``.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        tracker: Tracker | None = None,
        emit_near_misses: bool = False,
        pose_gate: PoseGate | None = None,
    ) -> None:
        self.camera_config = camera_config
        self.camera_id = camera_config.id
        max_age_frames = max(1, round(camera_config.sample_rate_fps))
        self.tracker = tracker or Tracker(max_age_frames=max_age_frames)
        self.emit_near_misses = emit_near_misses
        # Optional pose-aware gate (Phase 4). None → bbox-only (default, unchanged).
        self.pose_gate = pose_gate
        self._state = _DetectorState.IDLE
        self._window_start_mono: float | None = None
        self._window_start_ts: datetime | None = None
        self._window_detections: list[Detection] = []
        self._window_track_ids: set[str] = set()
        self._multi_dog = False
        self._ambiguous = False
        self._primary_track_id: str | None = None
        self._candidate_start_mono: float | None = None
        self._event_due_mono: float | None = None
        self._best_stats: _PostureStats | None = None
        self._near_miss_stats: _PostureStats | None = None
        # Per-track continuous-stationary run start (mono_ts); drives the dwell
        # trigger. Cleared when a track stops being stationary or the window resets.
        # Brief detection gaps are tolerated on purpose: ``is_stationary`` itself is
        # computed over a gap-forgiving trailing window (so intermittent night-time
        # detections still qualify), and dwell credit only survives while the track
        # stays alive in the tracker (bounded by ``max_age_frames``).
        self._stationary_since: dict[str, float] = {}
        # Continuous dwell (s) of the current candidate's primary track, captured for
        # confidence scaling and review metadata.
        self._best_dwell_s: float = 0.0
        self._suppressed_track_ids: set[str] = set()
        self._suppressed_until_mono: dict[str, float] = {}
        self._last_frame: Frame | None = None
        self._last_trigger_reason = TriggerReason.YOLO

    def process(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        trigger_reason: TriggerReason = TriggerReason.YOLO,
    ) -> list[PottyCandidate]:
        """Consume one frame's detections and return newly completed events."""

        self._last_frame = self._lightweight_frame(frame)
        self._last_trigger_reason = trigger_reason
        filtered = self._filter_detections(detections, frame)
        if self.pose_gate is not None:
            self.pose_gate.observe(frame, filtered)
        active_tracks = self.tracker.update(list(filtered))
        self._expire_suppressed(frame.mono_ts)
        current_tracks = [
            track
            for track in active_tracks
            if track.detections and track.detections[-1].frame_idx == frame.frame_idx
        ]
        if self.pose_gate is not None:
            self.pose_gate.prune(self._pose_keep_ids(active_tracks, frame.mono_ts))
        emitted: list[PottyCandidate] = []

        eligible_tracks = [
            track
            for track in current_tracks
            if track.track_id not in self._suppressed_track_ids
        ]
        if eligible_tracks:
            if self._state == _DetectorState.IDLE:
                self._start_window(frame)
            self._append_window(filtered, current_tracks)
            self._update_ambiguity(filtered, current_tracks)
            self._update_posture_state(frame, current_tracks)
        elif self._state == _DetectorState.WATCHING:
            self._reset_window()

        if self._state == _DetectorState.CANDIDATE and self._event_due_mono is not None:
            if frame.mono_ts >= self._event_due_mono:
                emitted.append(self._finish_event(frame, near_miss=False))
                self._suppress_tracks(emitted[-1], frame.mono_ts)
                self._reset_window()

        if not active_tracks:
            if self._state == _DetectorState.CANDIDATE:
                emitted.append(self._finish_event(frame, near_miss=False))
                self._suppress_tracks(emitted[-1], frame.mono_ts)
            elif self._state == _DetectorState.WATCHING and self.emit_near_misses:
                if self._near_miss_stats is not None:
                    emitted.append(self._finish_event(frame, near_miss=True))
            self._reset_window(clear_suppressed=True)

        return emitted

    def flush(self) -> list[PottyCandidate]:
        """Emit any open candidate at stream end, including optional near-miss."""

        if self._last_frame is None:
            return []
        emitted: list[PottyCandidate] = []
        if self._state == _DetectorState.CANDIDATE:
            emitted.append(self._finish_event(self._last_frame, near_miss=False))
            self._suppress_tracks(emitted[-1], self._last_frame.mono_ts)
        elif self._state == _DetectorState.WATCHING and self.emit_near_misses:
            if self._near_miss_stats is not None:
                emitted.append(self._finish_event(self._last_frame, near_miss=True))
        self._reset_window(clear_suppressed=True)
        return emitted

    def _filter_detections(
        self, detections: Sequence[Detection], frame: Frame
    ) -> list[Detection]:
        return [
            detection
            for detection in detections
            if detection.confidence >= self.camera_config.detection_conf_threshold
            and self._point_allowed(detection.bbox.center, frame.width, frame.height)
        ]

    def _point_allowed(self, point: tuple[float, float], width: int, height: int) -> bool:
        roi_zones = [zone for zone in self.camera_config.roi if len(zone.points) >= 3]
        ignore_zones = [
            zone for zone in self.camera_config.ignore_zones if len(zone.points) >= 3
        ]
        if not roi_zones and not ignore_zones:
            return True
        if width <= 0 or height <= 0:
            # Zones are configured but the frame size is unknown, so we cannot
            # normalize the detection center. Fail closed rather than letting an
            # unmeasurable frame bypass include/exclude filtering.
            return False
        # ``bbox.center`` is in original-resolution pixels; zone points are
        # normalized [0.0, 1.0], so normalize the center before testing.
        normalized = (point[0] / width, point[1] / height)
        if roi_zones and not any(_point_in_zone(normalized, zone) for zone in roi_zones):
            return False
        return not any(_point_in_zone(normalized, zone) for zone in ignore_zones)

    def _start_window(self, frame: Frame) -> None:
        self._state = _DetectorState.WATCHING
        self._window_start_mono = frame.mono_ts
        self._window_start_ts = frame.wall_ts
        self._window_detections = []
        self._window_track_ids = set()
        self._multi_dog = False
        self._ambiguous = False
        self._primary_track_id = None
        self._candidate_start_mono = None
        self._event_due_mono = None
        self._best_stats = None
        self._near_miss_stats = None
        self._stationary_since = {}
        self._best_dwell_s = 0.0

    def _append_window(
        self,
        detections: Sequence[Detection],
        current_tracks: Sequence[Track],
    ) -> None:
        self._window_detections.extend(detections)
        current_ids = {track.track_id for track in current_tracks}
        self._window_track_ids.update(current_ids)
        if len(current_ids) > 1 or len(self._window_track_ids) > 1:
            self._multi_dog = True
            self._ambiguous = True

    def _update_ambiguity(
        self,
        detections: Sequence[Detection],
        current_tracks: Sequence[Track],
    ) -> None:
        if len(current_tracks) > 1:
            self._multi_dog = True
            self._ambiguous = True
        for left_idx, left in enumerate(detections):
            for right in detections[left_idx + 1 :]:
                if iou(left.bbox, right.bbox) >= 0.2:
                    self._ambiguous = True

    def _update_posture_state(
        self,
        frame: Frame,
        current_tracks: Sequence[Track],
    ) -> None:
        stats = [self._posture_stats(track, frame.mono_ts) for track in current_tracks]
        stationary = [item for item in stats if item.is_stationary]
        for item in stationary:
            if self._near_miss_stats is None or _posture_rank(item) > _posture_rank(
                self._near_miss_stats,
            ):
                self._near_miss_stats = item

        # Maintain a per-track continuous-stationary accumulator so a dog that simply
        # holds still long enough triggers a candidate. Runs are dropped the moment a
        # track stops reading stationary.
        stationary_ids = {item.track_id for item in stationary}
        self._stationary_since = {
            track_id: since
            for track_id, since in self._stationary_since.items()
            if track_id in stationary_ids
        }
        dwell_by_id: dict[str, float] = {}
        for item in stationary:
            since = self._stationary_since.get(item.track_id)
            if since is None:
                # Credit the coverage already observed in the trailing window.
                since = item.window_start_mono
                self._stationary_since[item.track_id] = since
            dwell_by_id[item.track_id] = frame.mono_ts - since

        dwell_trigger_s = self.camera_config.dwell_trigger_s
        candidates = [
            item
            for item in stationary
            if item.track_id not in self._suppressed_track_ids
            and dwell_by_id[item.track_id] >= dwell_trigger_s
        ]
        if not candidates:
            return
        # Rank by the real continuous dwell first (the trigger signal), then by the
        # trailing-window span as a tiebreaker, so the strongest hold wins.
        best = max(
            candidates,
            key=lambda item: (dwell_by_id[item.track_id], item.stationary_duration_s),
        )
        best_dwell = dwell_by_id.get(best.track_id, 0.0)
        if self._state != _DetectorState.CANDIDATE:
            self._state = _DetectorState.CANDIDATE
            self._primary_track_id = best.track_id
            self._candidate_start_mono = best.window_start_mono
            self._event_due_mono = frame.mono_ts + self.camera_config.event_duration_s
            self._best_stats = best
            self._best_dwell_s = best_dwell
        elif self._primary_track_id == best.track_id:
            self._best_stats = best
            self._best_dwell_s = best_dwell

    def _posture_stats(self, track: Track, current_mono: float) -> _PostureStats:
        threshold_s = self.camera_config.stationary_threshold_s
        cutoff = current_mono - threshold_s
        recent = [detection for detection in track.detections if detection.mono_ts >= cutoff]
        if not recent and track.detections:
            recent = [track.detections[-1]]
        duration_s = max(0.0, recent[-1].mono_ts - recent[0].mono_ts) if recent else 0.0
        centers = [detection.bbox.center for detection in recent]
        first_center = centers[0] if centers else (0.0, 0.0)
        max_motion = max(
            (
                math.hypot(center[0] - first_center[0], center[1] - first_center[1])
                for center in centers
            ),
            default=math.inf,
        )
        diagonals = [
            math.hypot(detection.bbox.width, detection.bbox.height)
            for detection in recent
        ]
        median_diag = sorted(diagonals)[len(diagonals) // 2] if diagonals else 0.0
        motion_threshold = max(10.0, 0.2 * median_diag)
        # The trailing window must *cover* most of ``stationary_threshold_s`` rather
        # than span it exactly. Allow it to fall short by the larger of ~15% of the
        # threshold or two sample intervals (capped at half the threshold): discrete
        # sampling and intermittent (e.g. night) detections leave the in-window span
        # structurally below the threshold even when the dog stood still for many
        # seconds. Requiring real coverage (not merely an old track) still rejects a
        # near-empty window, and the motion check enforces localization.
        sample_fps = self.camera_config.sample_rate_fps
        sample_interval_s = (1.0 / sample_fps) if sample_fps > 0 else 0.0
        coverage_tolerance_s = min(
            max(_DURATION_TOLERANCE_S, 0.15 * threshold_s, 2.0 * sample_interval_s),
            0.5 * threshold_s,
        )
        covered_long_enough = (
            duration_s >= threshold_s - coverage_tolerance_s and len(recent) >= 2
        )

        bbox_motion_ok = max_motion <= motion_threshold
        bbox_is_stationary = covered_long_enough and bbox_motion_ok

        # Pose is additive: it may relax the motion-jitter check, but it never removes
        # the bbox coverage requirement, so the candidate-window timing structure (and
        # the dwell recall fix) is preserved. When the gate is off or pose is too
        # sparse/low-quality, this leaves the bbox result and posture_summary
        # byte-for-byte unchanged.
        is_stationary = bbox_is_stationary
        pose_summary: dict[str, Any] | None = None
        if self.pose_gate is not None:
            gate_result = self.pose_gate.posture(recent)
            if gate_result is not None:
                # The detection trigger is dwell-only, so the gate's squat signal is
                # not consumed and must not leak into review metadata.
                pose_summary = {
                    key: value
                    for key, value in gate_result.summary().items()
                    if key != "pose_squat"
                }
                if gate_result.pose_stationary:
                    is_stationary = covered_long_enough and (
                        bbox_motion_ok or gate_result.pose_stationary
                    )

        return _PostureStats(
            track_id=track.track_id,
            stationary_duration_s=duration_s,
            max_centroid_motion_px=max_motion,
            centroid_motion_threshold_px=motion_threshold,
            is_stationary=is_stationary,
            window_start_mono=recent[0].mono_ts if recent else current_mono,
            window_end_mono=recent[-1].mono_ts if recent else current_mono,
            pose_summary=pose_summary,
        )

    def _pose_keep_ids(
        self,
        active_tracks: Sequence[Track],
        current_mono: float,
    ) -> set[int]:
        """Identities of detections still inside any trailing posture window.

        Bounds the pose cache to the windows that ``_posture_stats`` can read
        (detections newer than ``current - stationary_threshold_s``, plus each
        track's latest detection as that is the single-frame fallback window).
        """

        cutoff = current_mono - self.camera_config.stationary_threshold_s
        keep: set[int] = set()
        for track in active_tracks:
            if not track.detections:
                continue
            keep.add(id(track.detections[-1]))
            for detection in track.detections:
                if detection.mono_ts >= cutoff:
                    keep.add(id(detection))
        return keep

    def _finish_event(self, frame: Frame, near_miss: bool) -> PottyCandidate:
        if self._window_start_mono is None or self._window_start_ts is None:
            self._start_window(frame)
        assert self._window_start_mono is not None
        assert self._window_start_ts is not None
        start_mono = self._window_start_mono
        end_mono = frame.mono_ts
        tracks = self._window_tracks(start_mono, end_mono)
        primary_track_id = self._primary_track_id or _first_track_id(tracks)
        if primary_track_id is None:
            primary_track_id = "unknown"
        stats = self._near_miss_stats if near_miss else self._best_stats
        posture_summary = self._posture_summary(stats)
        dwell_duration_s = 0.0 if near_miss else self._best_dwell_s
        posture_summary["dwell_trigger_s"] = self.camera_config.dwell_trigger_s
        posture_summary["dwell_duration_s"] = dwell_duration_s
        detections = [
            detection
            for detection in self._window_detections
            if start_mono <= detection.mono_ts <= end_mono
        ]
        multi_dog = self._multi_dog or len(tracks) > 1
        ambiguous = self._ambiguous or multi_dog
        lifecycle = PottyLifecycle.NEAR_MISS if near_miss else PottyLifecycle.EMITTED
        confidence = self._confidence(stats, near_miss, dwell_duration_s)
        return PottyCandidate(
            camera_id=self.camera_id,
            primary_track_id=primary_track_id,
            start_ts=self._window_start_ts,
            end_ts=frame.wall_ts,
            tracks=tracks,
            detections=detections,
            trigger_reason=self._last_trigger_reason,
            multi_dog=multi_dog,
            ambiguous=ambiguous,
            lifecycle=lifecycle,
            stationary_duration_s=stats.stationary_duration_s if stats else 0.0,
            posture_summary=posture_summary,
            near_miss=near_miss,
            confidence=confidence,
        )

    def _window_tracks(self, start_mono: float, end_mono: float) -> list[Track]:
        tracks: list[Track] = []
        for track_id in sorted(self._window_track_ids, key=_track_sort_key):
            history = self.tracker.get_track(track_id)
            if history is None:
                continue
            detections = [
                detection
                for detection in history.detections
                if start_mono <= detection.mono_ts <= end_mono
            ]
            if detections:
                tracks.append(Track(track_id=track_id, detections=detections))
        return tracks

    def _posture_summary(self, stats: _PostureStats | None) -> dict[str, Any]:
        if stats is None:
            return {
                "stationary_threshold_s": self.camera_config.stationary_threshold_s,
            }
        summary = {
            "stationary_threshold_s": self.camera_config.stationary_threshold_s,
            "stationary_duration_s": stats.stationary_duration_s,
            "max_centroid_motion_px": stats.max_centroid_motion_px,
            "centroid_motion_threshold_px": stats.centroid_motion_threshold_px,
            "posture_window_start_mono": stats.window_start_mono,
            "posture_window_end_mono": stats.window_end_mono,
        }
        # Additive: pose keys appear only when the gate actually contributed, so
        # gate-off output is unchanged.
        if stats.pose_summary is not None:
            summary["pose"] = stats.pose_summary
        return summary

    @staticmethod
    def _confidence(
        stats: _PostureStats | None,
        near_miss: bool,
        dwell_duration_s: float = 0.0,
    ) -> float:
        if stats is None:
            return 0.25 if near_miss else 0.5
        if near_miss:
            return min(0.45, 0.25 + 0.05 * stats.stationary_duration_s)
        # Events are dwell-triggered: confidence grows with how long the dog actually
        # held still.
        return min(0.7, 0.4 + 0.03 * dwell_duration_s)

    def _suppress_tracks(self, candidate: PottyCandidate, current_mono: float) -> None:
        cooldown_s = max(
            self.camera_config.dwell_trigger_s,
            self.camera_config.stationary_threshold_s,
            self.camera_config.event_duration_s,
        )
        until = current_mono + cooldown_s
        for track in candidate.tracks:
            self._suppressed_track_ids.add(track.track_id)
            self._suppressed_until_mono[track.track_id] = until

    def _expire_suppressed(self, current_mono: float) -> None:
        for track_id, until in list(self._suppressed_until_mono.items()):
            if current_mono >= until:
                del self._suppressed_until_mono[track_id]
                self._suppressed_track_ids.discard(track_id)

    @staticmethod
    def _lightweight_frame(frame: Frame) -> Frame:
        return Frame(
            bgr=frame.bgr[:0, :0].copy(),
            frame_idx=frame.frame_idx,
            mono_ts=frame.mono_ts,
            wall_ts=frame.wall_ts,
            source_id=frame.source_id,
        )

    def _reset_window(self, clear_suppressed: bool = False) -> None:
        self._state = _DetectorState.IDLE
        self._window_start_mono = None
        self._window_start_ts = None
        self._window_detections = []
        self._window_track_ids = set()
        self._multi_dog = False
        self._ambiguous = False
        self._primary_track_id = None
        self._candidate_start_mono = None
        self._event_due_mono = None
        self._best_stats = None
        self._near_miss_stats = None
        self._stationary_since = {}
        self._best_dwell_s = 0.0
        if clear_suppressed:
            self._suppressed_track_ids = set()
            self._suppressed_until_mono = {}


def _posture_rank(stats: _PostureStats) -> tuple[float, float]:
    return (stats.stationary_duration_s, -stats.max_centroid_motion_px)


def _first_track_id(tracks: Sequence[Track]) -> str | None:
    return tracks[0].track_id if tracks else None


def _track_sort_key(track_id: str) -> tuple[int, str]:
    try:
        return (int(track_id), track_id)
    except ValueError:
        return (0, track_id)


def _point_in_zone(point: tuple[float, float], zone: ZoneConfig) -> bool:
    return _point_in_polygon(point, zone.points)


def _point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y)
        if intersects:
            slope_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < slope_x:
                inside = not inside
        previous = current
    return inside


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > 1e-9:
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= 1e-9


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _jsonify(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonify(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _ensure_aware_utc(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    return value
