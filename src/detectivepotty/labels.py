"""``labels.json`` schema for range-based labeling of harvested clips.

A labeler scrubs a harvested ``clip.mp4`` and marks frame ranges, each carrying a
behavior, a dog identity, and the dog track the label binds to. Ranges are stored
in a **dual time basis** (frame indices *and* seconds, plus a ``time_basis`` tag)
so the exporter never has to guess how the UI located the range. This module owns
the schema, enum validation, and atomic load/save; both the exporter and the
(future) label API depend on it.

Frame numbering is the **harvested clip's own** 0-based numbering (what the UI
scrubs), not the source recording's — the clip's ``metadata.json`` carries the
mapping back to the source.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any

LABELS_NAME = "labels.json"
SCHEMA_VERSION = "labels-1.0"
DEFAULT_TIME_BASIS = "clip_frames"


class Behavior(str, Enum):
    """What the dog is doing in the range. ``EXCLUDED`` = ambiguous/skip."""

    PEE = "pee"
    POOP = "poop"
    NOT_POTTY = "not_potty"
    EXCLUDED = "excluded"


class Dog(str, Enum):
    """Dog identity. ``UNKNOWN`` is open-set — excluded from dog-ID training."""

    GROMIT = "gromit"
    WALLE = "walle"
    APOLLO = "apollo"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "Dog | None":
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "").replace("_", "")
            for member in cls:
                if member.value.replace("-", "").replace("_", "") == normalized:
                    return member
        return None


# Behaviors that produce training crops (EXCLUDED never does).
TRAINABLE_BEHAVIORS = frozenset({Behavior.PEE, Behavior.POOP, Behavior.NOT_POTTY})


@dataclass(slots=True)
class LabelRange:
    """One labeled frame range bound to a single dog track.

    ``start_frame``/``end_frame`` are inclusive 0-based clip frames;
    ``start_s``/``end_s`` are the matching seconds. ``track_id`` ties the label to
    a specific dog (so multi-dog clips crop the right animal).
    """

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    behavior: Behavior
    dog: Dog = Dog.UNKNOWN
    track_id: str | None = None
    time_basis: str = DEFAULT_TIME_BASIS
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        self.behavior = Behavior(self.behavior)
        self.dog = Dog(self.dog)
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) < start_frame ({self.start_frame})"
            )
        if self.end_s + 1e-6 < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) < start_s ({self.start_s})")
        if self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")

    @property
    def is_trainable(self) -> bool:
        return self.behavior in TRAINABLE_BEHAVIORS

    def frames(self) -> range:
        return range(self.start_frame, self.end_frame + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "behavior": self.behavior.value,
            "dog": self.dog.value,
            "track_id": self.track_id,
            "time_basis": self.time_basis,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelRange":
        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            behavior=Behavior(data["behavior"]),
            dog=Dog(data.get("dog", Dog.UNKNOWN.value)),
            track_id=data.get("track_id"),
            time_basis=str(data.get("time_basis", DEFAULT_TIME_BASIS)),
            created_at=str(data.get("created_at"))
            if data.get("created_at")
            else datetime.now(timezone.utc).isoformat(),
        )


@dataclass(slots=True)
class ClipLabels:
    """All labeled ranges for one harvested clip."""

    clip: str = "clip.mp4"
    ranges: list[LabelRange] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "clip": self.clip,
            "ranges": [item.to_dict() for item in self.ranges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipLabels":
        return cls(
            clip=str(data.get("clip", "clip.mp4")),
            ranges=[LabelRange.from_dict(item) for item in data.get("ranges", [])],
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

    def trainable_ranges(self) -> Iterable[LabelRange]:
        return (item for item in self.ranges if item.is_trainable)


def load_labels(path: str | Path) -> ClipLabels:
    """Load a ``labels.json`` (file or its containing directory)."""

    target = Path(path)
    if target.is_dir():
        target = target / LABELS_NAME
    with target.open("r", encoding="utf-8") as fh:
        return ClipLabels.from_dict(json.load(fh))


def save_labels(labels: ClipLabels, path: str | Path) -> Path:
    """Atomically write ``labels.json`` (file or its containing directory)."""

    target = Path(path)
    if target.is_dir() or target.suffix.lower() != ".json":
        target = target / LABELS_NAME if target.suffix == "" else target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".labels.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(labels.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target
