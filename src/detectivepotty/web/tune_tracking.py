"""Ultralytics tracking adapter helpers for the Tune API."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml


# Detection floor for the in-browser tuner. The detector runs at this low
# confidence so borderline boxes are still returned; the client-side slider
# decides green-kept vs red-dropped without any re-inference.
TUNE_DETECTION_FLOOR = 0.05

# Tune "Track range" tracker backends. ``off``/``ours`` use the harvest IoU
# ``Tracker`` replay (every model incl. CoreML); the three native values map to
# Ultralytics built-in trackers (``.pt``-only — Ultralytics tracking won't run on
# a CoreML package). ``botsort_reid`` is BoT-SORT with appearance ReID enabled.
TUNE_TRACKERS = ("off", "ours", "bytetrack", "botsort", "botsort_reid")
ULTRALYTICS_TRACKERS = ("bytetrack", "botsort", "botsort_reid")


@dataclass(frozen=True, slots=True)
class TuneUltralyticsTrackerParams:
    """Per-run Ultralytics tracking knobs exposed by the Tune UI.

    ``conf`` is passed to ``YOLO.track``. The other fields are optional overrides
    for the bundled ByteTrack/BoT-SORT YAML; ``None`` means keep the YAML default.
    """

    conf: float = TUNE_DETECTION_FLOOR
    track_high_thresh: float | None = None
    track_low_thresh: float | None = None
    new_track_thresh: float | None = None
    track_buffer: int | None = None
    match_thresh: float | None = None
    proximity_thresh: float | None = None
    appearance_thresh: float | None = None

    def yaml_overrides(self, tracker: str) -> dict[str, float | int | bool]:
        overrides: dict[str, float | int | bool] = {}
        for key in (
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
        ):
            value = getattr(self, key)
            if value is not None:
                overrides[key] = value
        if tracker in ("botsort", "botsort_reid"):
            for key in ("proximity_thresh", "appearance_thresh"):
                value = getattr(self, key)
                if value is not None:
                    overrides[key] = value
        if tracker == "botsort_reid":
            overrides["with_reid"] = True
        return overrides

    def payload(self, tracker: str) -> dict[str, float | int | bool | None]:
        return {
            "conf": self.conf,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "proximity_thresh": (
                self.proximity_thresh if tracker in ("botsort", "botsort_reid") else None
            ),
            "appearance_thresh": (
                self.appearance_thresh if tracker in ("botsort", "botsort_reid") else None
            ),
            "with_reid": tracker == "botsort_reid",
        }


def ultralytics_tracking_available() -> bool:
    """True when Ultralytics native tracking can run (its ``lap`` dep imports)."""

    import importlib.util

    return importlib.util.find_spec("lap") is not None


def ultralytics_dog_class_indices(
    model: object, alias_classes: Iterable[str] = ()
) -> list[int]:
    """Class indices accepted as dogs for ``model`` (``dog`` + any aliases)."""

    accepted = {"dog", *(str(c).lower() for c in alias_classes)}
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        idxs = [int(i) for i, n in names.items() if str(n).lower() in accepted]
    else:  # pragma: no cover - list-style names are rare
        idxs = [i for i, n in enumerate(names) if str(n).lower() in accepted]
    return idxs or [16]


def ultralytics_tracker_yaml(
    trackers_dir: Path,
    tracker: str,
    params: TuneUltralyticsTrackerParams,
) -> tuple[str, Path | None]:
    """Return ``(tracker_yaml, temp_dir)`` for one Ultralytics tracking request."""

    if tracker == "bytetrack":
        base_yaml = trackers_dir / "bytetrack.yaml"
    elif tracker in ("botsort", "botsort_reid"):
        base_yaml = trackers_dir / "botsort.yaml"
    else:  # pragma: no cover - guarded by endpoints
        raise ValueError(f"unknown ultralytics tracker: {tracker}")

    overrides = params.yaml_overrides(tracker)
    if not overrides:
        return str(base_yaml), None

    with base_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid tracker yaml: {base_yaml}")
    data.update(overrides)

    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="dp_tracker_"))
    out = tmp_dir / f"{tracker}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(out), tmp_dir


def ultralytics_boxes(result: object) -> list[dict]:
    """Map one Ultralytics tracking ``Results`` to Tune track-box dicts."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    ids = boxes.id.tolist()
    out: list[dict] = []
    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        out.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(conf),
                "class_name": "dog",
                "track_id": str(int(tid)),
            }
        )
    return out
