"""Rerun reconciliation for the event recorder.

When detection is re-run over the same source, each candidate would otherwise be
written as a brand-new event (fresh ``uuid4`` ``event_id``) next to the previous
run's events, and any human labels applied in the review portal would be lost.

This module matches a freshly-detected event against prior on-disk events for the
same camera + source and decides whether to carry forward the human label fields
and supersede (delete) the now-duplicate prior directories. The recorder owns the
filesystem write; the pure decision logic lives here so it is easy to test.

Safety properties (see the recorder for how they are wired):
- Matching runs against a per-run snapshot, so a run never matches events it just
  wrote (no self-deletion).
- Disagreeing human labels are treated as a conflict: nothing is carried and
  nothing is deleted.
- ``protect_recording.mp4`` is carried forward before a prior dir is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

from detectivepotty.events import LabelStatus

PROTECT_RECORDING_NAME = "protect_recording.mp4"


@dataclass
class PriorEvent:
    """A previously recorded event considered as a rerun match candidate."""

    dir_path: Path
    metadata: dict[str, Any]

    @property
    def event_id(self) -> str:
        return str(self.metadata.get("event_id") or self.dir_path.name)

    @property
    def start_ts(self) -> datetime | None:
        return parse_metadata_ts(self.metadata.get("utc_ts"))

    @property
    def end_ts(self) -> datetime | None:
        return parse_metadata_ts(self.metadata.get("end_ts"))

    @property
    def source_start_s(self) -> float | None:
        return _as_float(self.metadata.get("source_start_s"))

    @property
    def source_end_s(self) -> float | None:
        return _as_float(self.metadata.get("source_end_s"))

    @property
    def protect_event_id(self) -> str | None:
        value = self.metadata.get("protect_event_id")
        return str(value) if value else None

    @property
    def label_status(self) -> str | None:
        value = self.metadata.get("label_status")
        return str(value) if value else None

    @property
    def is_labeled(self) -> bool:
        status = self.label_status
        return bool(status) and status != LabelStatus.UNLABELED.value

    @property
    def label_note(self) -> str | None:
        extra = self.metadata.get("extra")
        if isinstance(extra, dict):
            note = extra.get("label_note")
            return str(note) if note is not None else None
        return None

    @property
    def labeled_at(self) -> str | None:
        extra = self.metadata.get("extra")
        if isinstance(extra, dict):
            value = extra.get("labeled_at")
            return str(value) if value is not None else None
        return None

    def _human_key(self) -> tuple[Any, Any, Any]:
        return (
            self.metadata.get("label"),
            self.metadata.get("dog"),
            self.label_note,
        )


@dataclass
class ReconcileResult:
    """Outcome of reconciling a new event against prior events."""

    event_id: str | None = None
    carried: dict[str, Any] = field(default_factory=dict)
    superseded: list[PriorEvent] = field(default_factory=list)
    conflict: bool = False


def parse_metadata_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def snapshot_prior_events(
    camera_dir: Path,
    camera_id: str,
    source_id: str,
) -> list[PriorEvent]:
    """Load prior events for one camera + source via a depth-bounded glob.

    The dataset layout is ``<camera>/<date>/events/<event>/metadata.json`` so the
    glob never descends into the large ``frames``/``crops`` directories.
    """

    if not camera_dir.is_dir() or camera_dir.is_symlink():
        return []
    priors: list[PriorEvent] = []
    for metadata_path in camera_dir.glob("*/events/*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if metadata.get("camera_id") != camera_id:
            continue
        if metadata.get("sanitized_source_id") != source_id:
            continue
        priors.append(PriorEvent(dir_path=metadata_path.parent, metadata=metadata))
    return priors


def match_priors(
    snapshot: list[PriorEvent],
    *,
    start_ts: datetime,
    end_ts: datetime | None,
    protect_event_id: str | None,
    tolerance_s: float,
    source_start_s: float | None = None,
    source_end_s: float | None = None,
) -> list[PriorEvent]:
    """Return priors that represent the same real-world event, closest first.

    Priority: exact ``protect_event_id`` > source-relative interval overlap (when
    both events carry in-clip offsets) > wall-clock interval overlap (when both
    have an ``end_ts``) > start within ``tolerance_s``. A fuzzy match never
    crosses a *different* non-empty ``protect_event_id``.

    Source-relative overlap is anchor-independent: it matches the same in-clip
    moment across reruns even if the wall-clock anchor differed between runs.
    """

    if protect_event_id:
        exact = [p for p in snapshot if p.protect_event_id == protect_event_id]
        if exact:
            return _sort_by_closeness(exact, start_ts)

    matched: list[PriorEvent] = []
    for prior in snapshot:
        if prior.protect_event_id and prior.protect_event_id != protect_event_id:
            continue
        if (
            _source_overlaps(source_start_s, source_end_s, prior)
            or _overlaps(start_ts, end_ts, prior)
            or _within_tolerance(start_ts, prior, tolerance_s)
        ):
            matched.append(prior)
    return _sort_by_closeness(matched, start_ts)


def decide_carry(matched: list[PriorEvent]) -> ReconcileResult:
    """Decide which label fields to carry and which priors to supersede.

    Conservative: if labeled priors disagree on any human field, carry nothing and
    delete nothing (keep all priors).
    """

    if not matched:
        return ReconcileResult()

    labeled = [p for p in matched if p.is_labeled]
    if not labeled:
        # All matches are unlabeled duplicates: collapse onto the closest id.
        return ReconcileResult(event_id=matched[0].event_id, superseded=list(matched))

    if len({p._human_key() for p in labeled}) > 1:
        return ReconcileResult(conflict=True)

    representative = max(labeled, key=lambda p: p.labeled_at or "")
    carried = {
        "label": representative.metadata.get("label"),
        "label_status": representative.metadata.get("label_status"),
        "dog": representative.metadata.get("dog"),
        "label_note": representative.label_note,
        "labeled_at": representative.labeled_at,
    }
    return ReconcileResult(
        event_id=representative.event_id,
        carried=carried,
        superseded=list(matched),
    )


def preserve_protect_recording(old_dir: Path, new_dir: Path) -> bool:
    """Move a prior ``protect_recording.mp4`` into ``new_dir`` if it lacks one.

    Returns ``True`` when there is nothing to preserve or the move succeeds, and
    ``False`` when a recording exists but could not be carried forward (so a
    caller can choose to keep the source dir instead of deleting it).
    """

    src = old_dir / PROTECT_RECORDING_NAME
    dst = new_dir / PROTECT_RECORDING_NAME
    try:
        if not src.is_file() or src.is_symlink() or dst.exists():
            return True
        shutil.move(str(src), str(dst))
        return True
    except OSError:
        return False


def remove_event_dir(path: Path) -> None:
    """Recursively remove an event directory, ignoring symlinks."""

    if path.is_symlink() or not path.is_dir():
        return
    shutil.rmtree(path, ignore_errors=True)


def _overlaps(start_ts: datetime, end_ts: datetime | None, prior: PriorEvent) -> bool:
    prior_start = prior.start_ts
    prior_end = prior.end_ts
    if end_ts is None or prior_start is None or prior_end is None:
        return False
    return start_ts <= prior_end and prior_start <= end_ts


def _source_overlaps(
    source_start_s: float | None,
    source_end_s: float | None,
    prior: PriorEvent,
) -> bool:
    """Overlap of in-clip [start, end] offsets, independent of the wall anchor."""

    prior_start = prior.source_start_s
    prior_end = prior.source_end_s
    if (
        source_start_s is None
        or source_end_s is None
        or prior_start is None
        or prior_end is None
    ):
        return False
    return source_start_s <= prior_end and prior_start <= source_end_s


def _within_tolerance(start_ts: datetime, prior: PriorEvent, tolerance_s: float) -> bool:
    prior_start = prior.start_ts
    if prior_start is None:
        return False
    return abs((start_ts - prior_start).total_seconds()) <= tolerance_s


def _sort_by_closeness(
    priors: list[PriorEvent],
    start_ts: datetime,
) -> list[PriorEvent]:
    def key(prior: PriorEvent) -> tuple[float, str]:
        prior_start = prior.start_ts
        delta = (
            abs((start_ts - prior_start).total_seconds())
            if prior_start is not None
            else float("inf")
        )
        return (delta, prior.dir_path.name)

    return sorted(priors, key=key)


def paths_equal(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(os.path.realpath(left)) == os.path.normcase(
            os.path.realpath(right)
        )
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# One-time dedupe of EXISTING on-disk duplicates (no re-detection)
# --------------------------------------------------------------------------- #


@dataclass
class DedupeAction:
    """A planned (or applied) collapse of one cluster of duplicate events."""

    keeper: Path | None
    removed: list[Path]
    carried: dict[str, Any]
    conflict: bool
    cluster: list[Path]


def cluster_events(priors: list[PriorEvent], tolerance_s: float) -> list[list[PriorEvent]]:
    """Group prior events that are reruns of the same underlying event.

    A single sweep over start-sorted events. An event joins the running cluster
    when EITHER its interval overlaps the cluster's covered window (truly the
    same continuous activity) OR its start is within ``tolerance_s`` of the
    cluster *anchor* (the first event's start). Anchoring the tolerance check to
    the anchor — not the running end — prevents a long train of events, each
    just within tolerance of the previous one, from chaining into one giant
    cluster and collapsing genuinely distinct events.
    """

    far_past = datetime.min.replace(tzinfo=timezone.utc)
    ordered = sorted(priors, key=lambda p: p.start_ts or far_past)
    clusters: list[list[PriorEvent]] = []
    current: list[PriorEvent] = []
    anchor_start: datetime | None = None
    running_end: datetime | None = None
    for prior in ordered:
        start = prior.start_ts
        end = prior.end_ts or start
        if not current:
            current = [prior]
            anchor_start = start
            running_end = end
            continue
        overlaps = (
            start is not None and running_end is not None and start <= running_end
        )
        near_anchor = (
            start is not None
            and anchor_start is not None
            and (start - anchor_start).total_seconds() <= tolerance_s
        )
        if overlaps or near_anchor:
            current.append(prior)
            if end is not None and (running_end is None or end > running_end):
                running_end = end
        else:
            clusters.append(current)
            current = [prior]
            anchor_start = start
            running_end = end
    if current:
        clusters.append(current)
    return clusters


def dedupe_dataset(
    dataset_dir: str | Path,
    *,
    tolerance_s: float,
    dry_run: bool,
) -> list[DedupeAction]:
    """Collapse existing duplicate events per (camera, source) cluster.

    Keeps the newest-media copy of each cluster, carries human labels forward with
    the same conflict guard as live reruns, and removes the rest. Conflicting
    human labels leave the whole cluster untouched.
    """

    root = Path(dataset_dir)
    actions: list[DedupeAction] = []
    if not root.is_dir():
        return actions

    groups: dict[tuple[str, str], list[PriorEvent]] = {}
    for metadata_path in root.glob("*/*/events/*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        camera_id = metadata.get("camera_id")
        source_id = metadata.get("sanitized_source_id")
        # Skip records that cannot be reliably grouped: missing identity would
        # otherwise lump unrelated events into a shared "None" bucket and risk
        # deleting events that are not actually duplicates.
        if not isinstance(camera_id, str) or not camera_id:
            continue
        if not isinstance(source_id, str) or not source_id:
            continue
        groups.setdefault((camera_id, source_id), []).append(
            PriorEvent(dir_path=metadata_path.parent, metadata=metadata)
        )

    for priors in groups.values():
        for cluster in cluster_events(priors, tolerance_s):
            if len(cluster) < 2:
                continue
            action = _dedupe_cluster(cluster, dry_run=dry_run)
            if action is not None:
                actions.append(action)
    return actions


def _dedupe_cluster(cluster: list[PriorEvent], *, dry_run: bool) -> DedupeAction | None:
    cluster_paths = [p.dir_path for p in cluster]
    decision = decide_carry(cluster)
    if decision.conflict:
        return DedupeAction(
            keeper=None,
            removed=[],
            carried={},
            conflict=True,
            cluster=cluster_paths,
        )

    keeper = max(cluster, key=_media_recency)
    removable = [p for p in cluster if p.dir_path != keeper.dir_path]
    if not removable:
        return None

    if dry_run:
        return DedupeAction(
            keeper=keeper.dir_path,
            removed=[p.dir_path for p in removable],
            carried=decision.carried,
            conflict=False,
            cluster=cluster_paths,
        )

    # Fail-closed: never delete a duplicate until the human labels are safely
    # persisted on the keeper. If the carry-forward write fails, keep everything
    # and report the cluster as a conflict so the user can resolve it manually.
    if decision.carried and not _apply_carried_to_metadata_file(
        keeper.dir_path, decision.carried
    ):
        return DedupeAction(
            keeper=keeper.dir_path,
            removed=[],
            carried={},
            conflict=True,
            cluster=cluster_paths,
        )

    removed: list[Path] = []
    for prior in removable:
        # Also fail-closed on protect-recording preservation: if a recording
        # exists but cannot be carried forward, keep the source dir.
        if not preserve_protect_recording(prior.dir_path, keeper.dir_path):
            continue
        remove_event_dir(prior.dir_path)
        removed.append(prior.dir_path)

    return DedupeAction(
        keeper=keeper.dir_path,
        removed=removed,
        carried=decision.carried,
        conflict=False,
        cluster=cluster_paths,
    )


def _media_recency(prior: PriorEvent) -> datetime:
    recorded_at = parse_metadata_ts(prior.metadata.get("recorded_at"))
    if recorded_at is not None:
        return recorded_at
    try:
        mtime = (prior.dir_path / "metadata.json").stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _apply_carried_to_metadata_file(event_dir: Path, carried: dict[str, Any]) -> bool:
    """Write carried human labels into an existing event's metadata.json.

    Returns ``True`` on a successful atomic write and ``False`` on any read or
    write failure, so the caller can avoid deleting duplicates whose labels were
    never persisted onto the keeper.
    """

    metadata_path = event_dir / "metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    for key in ("label", "label_status", "dog"):
        if carried.get(key) is not None or key == "dog":
            metadata[key] = carried.get(key)
    extra = metadata.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        metadata["extra"] = extra
    if carried.get("label_note") is not None:
        extra["label_note"] = carried["label_note"]
    if carried.get("labeled_at") is not None:
        extra["labeled_at"] = carried["labeled_at"]

    tmp_path = event_dir / f".metadata.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, metadata_path)
        return True
    except OSError:
        tmp_path.unlink(missing_ok=True)
        return False
