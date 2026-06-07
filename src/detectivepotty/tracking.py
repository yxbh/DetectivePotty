"""Lightweight v0 dog tracking helpers.

The tracker below is intentionally small and dependency-free: it greedily assigns
new detections to existing tracks by bounding-box IoU. It is good enough for v0
potty-candidate windows, but it is not re-identification; IDs can swap during
occlusion/overlap. ByteTrack/Ultralytics can replace this module later behind the
same ``Tracker.update`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from detectivepotty.events import Detection, Track
from detectivepotty.geometry import BBox


def iou(left: BBox, right: BBox) -> float:
    """Return intersection-over-union for two pixel-space boxes."""

    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    union = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


# Window boxes whose center has drifted more than this many reference-diagonals
# away are treated as a moving/other dog and excluded, so a trailing union over a
# walking dog cannot elongate the crop along its path.
_UNION_MAX_CENTER_SHIFT_FRAC = 1.0
# A single anomalously large box must not poison the union into a giant crop: if
# the union would exceed this multiple of the reference area, keep the raw box.
_UNION_MAX_GROWTH_RATIO = 4.0


def temporal_box_union(
    detections: Sequence[Detection],
    reference: Detection,
    window_s: float,
    *,
    max_center_shift_frac: float = _UNION_MAX_CENTER_SHIFT_FRAC,
    max_growth_ratio: float = _UNION_MAX_GROWTH_RATIO,
) -> BBox:
    """Union ``reference``'s box with same-track boxes in a short trailing window.

    Recovers full dog extent and stabilizes tiny boxes when a single frame's
    detector under-segments (e.g. low-contrast IR that boxes only a bright
    sub-region), which otherwise feeds pose a partial crop. The window is the
    ``mono_ts`` interval ``[reference.mono_ts - window_s, reference.mono_ts]``
    (trailing, inclusive of the reference), so it is robust to variable night fps.

    Two guards keep the union from becoming a worse crop than the raw box: window
    boxes whose center drifts more than ``max_center_shift_frac`` reference
    diagonals away are skipped (a moving/other dog), and if the resulting union
    still exceeds ``max_growth_ratio`` times the reference area the raw box is
    returned unchanged. Returns ``reference.bbox`` unchanged when ``window_s <= 0``
    or no in-window neighbor survives the guards.
    """

    if window_s <= 0.0:
        return reference.bbox

    ref = reference.bbox
    ref_cx, ref_cy = ref.center
    ref_diag = math.hypot(ref.width, ref.height)
    lo = reference.mono_ts - window_s

    box = ref
    for detection in detections:
        if detection is reference:
            continue
        if not (lo <= detection.mono_ts <= reference.mono_ts):
            continue
        candidate = detection.bbox
        cx, cy = candidate.center
        if ref_diag > 0.0:
            shift = math.hypot(cx - ref_cx, cy - ref_cy)
            if shift > max_center_shift_frac * ref_diag:
                continue
        box = box.union(candidate)

    if ref.area > 0.0 and box.area > max_growth_ratio * ref.area:
        return ref
    return box


@dataclass(slots=True)
class _TrackState:
    track: Track
    last_detection: Detection
    last_frame_idx: int
    missed_frames: int = 0


class Tracker:
    """Greedy IoU multi-object tracker with string track IDs.

    ``update`` expects detections from one sampled frame and returns tracks still
    considered active. Histories are retained after death via ``histories`` or
    ``get_track`` so recorder/state-machine code can recover full windows.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_frames: int = 5,
        min_confidence: float = 0.0,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age_frames < 0:
            raise ValueError("max_age_frames must be non-negative")
        if min_confidence < 0.0:
            raise ValueError("min_confidence must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.min_confidence = min_confidence
        self._next_id = 1
        self._states: dict[str, _TrackState] = {}
        self._histories: dict[str, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate detections with existing tracks and return active tracks."""

        detections = [
            detection
            for detection in detections
            if detection.confidence >= self.min_confidence
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        current_frame_idx = max(
            (detection.frame_idx for detection in detections),
            default=None,
        )

        if not detections:
            self._age_unmatched(current_frame_idx)
            return self.active_tracks

        candidate_pairs: list[tuple[float, str, int]] = []
        for track_id, state in self._states.items():
            for detection_idx, detection in enumerate(detections):
                score = iou(state.last_detection.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidate_pairs.append((score, track_id, detection_idx))
        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_idx in candidate_pairs:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            detection = detections[detection_idx]
            state = self._states[track_id]
            state.track.detections.append(detection)
            state.last_detection = detection
            state.last_frame_idx = detection.frame_idx
            state.missed_frames = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        self._age_unmatched(current_frame_idx, excluding=matched_tracks)

        for detection_idx, detection in enumerate(detections):
            if detection_idx not in matched_detections:
                self._birth(detection)

        self._drop_expired()
        return self.active_tracks

    @property
    def active_tracks(self) -> list[Track]:
        """Tracks whose last detection is within ``max_age_frames`` updates."""

        return [
            self._states[track_id].track
            for track_id in sorted(self._states, key=_track_sort_key)
        ]

    @property
    def histories(self) -> list[Track]:
        """All tracks ever created, including inactive/dead histories."""

        return [
            self._histories[track_id]
            for track_id in sorted(self._histories, key=_track_sort_key)
        ]

    def get_track(self, track_id: str) -> Track | None:
        """Return a retained track history by ID, if known."""

        return self._histories.get(track_id)

    def _birth(self, detection: Detection) -> None:
        track_id = str(self._next_id)
        self._next_id += 1
        track = Track(track_id=track_id, detections=[detection])
        self._histories[track_id] = track
        self._states[track_id] = _TrackState(
            track=track,
            last_detection=detection,
            last_frame_idx=detection.frame_idx,
        )

    def _age_unmatched(
        self,
        current_frame_idx: int | None,
        excluding: set[str] | None = None,
    ) -> None:
        excluding = excluding or set()
        for track_id, state in list(self._states.items()):
            if track_id in excluding:
                continue
            gap = 1
            if current_frame_idx is not None and current_frame_idx > state.last_frame_idx:
                gap = max(1, current_frame_idx - state.last_frame_idx)
            state.missed_frames += gap
        self._drop_expired()

    def _drop_expired(self) -> None:
        for track_id, state in list(self._states.items()):
            if state.missed_frames > self.max_age_frames:
                del self._states[track_id]


def _track_sort_key(track_id: str) -> tuple[int, str]:
    try:
        return (int(track_id), track_id)
    except ValueError:
        return (0, track_id)
