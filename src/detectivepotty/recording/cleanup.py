"""One-time, label-preserving cleanup of legacy duplicate events.

Early runs anchored the file timeline to ``datetime.now()`` per run, so the same
clip moment was re-detected with a fresh timestamp every run and piled up as
brand-new event directories that the rerun-dedupe could never match. Once the
timeline is deterministic, the clean fix is to re-run detection — but the dataset
still carries the old duplicate noise.

This module removes only that noise, conservatively:

- It never touches an event with any human signal (label/status/dog/note).
- It never touches a deterministic-era event (one written with ``end_ts`` and
  ``recorded_at``), so freshly recorded events are always kept.
- It only removes a legacy, unlabeled event when its **source video still exists**
  so a clean re-run can regenerate an equivalent event.
- Removals are **quarantined** into ``<dataset>/.trash/<timestamp>/`` (a move, not
  a delete) so the operation is fully reversible.

The CLI runs in dry-run mode by default and reports exactly what it would keep,
remove, and skip before anything is moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import unquote, urlsplit

from detectivepotty.events import Label, LabelStatus

# Event classifications, in keep/remove terms.
KEEP_LABELED = "labeled"
KEEP_DETERMINISTIC = "deterministic"
KEEP_SOURCE_MISSING = "source_missing"
REMOVE_LEGACY = "removable"


@dataclass
class CleanupItem:
    """One classified event and where it would go (or went)."""

    event_dir: Path
    event_id: str
    classification: str
    source_id: str | None
    moved_to: Path | None = None


@dataclass
class CleanupReport:
    items: list[CleanupItem] = field(default_factory=list)
    trash_dir: Path | None = None
    applied: bool = False

    def by_class(self, classification: str) -> list[CleanupItem]:
        return [item for item in self.items if item.classification == classification]

    @property
    def removable(self) -> list[CleanupItem]:
        return self.by_class(REMOVE_LEGACY)

    @property
    def kept_labeled(self) -> list[CleanupItem]:
        return self.by_class(KEEP_LABELED)

    @property
    def kept_deterministic(self) -> list[CleanupItem]:
        return self.by_class(KEEP_DETERMINISTIC)

    @property
    def skipped_source_missing(self) -> list[CleanupItem]:
        return self.by_class(KEEP_SOURCE_MISSING)


def has_human_signal(metadata: dict[str, Any]) -> bool:
    """True when an event carries any human review signal worth preserving."""

    status = metadata.get("label_status")
    if status and str(status) != LabelStatus.UNLABELED.value:
        return True
    label = metadata.get("label")
    if label and str(label) != Label.UNKNOWN.value:
        return True
    if metadata.get("dog"):
        return True
    extra = metadata.get("extra")
    if isinstance(extra, dict):
        if extra.get("label_note"):
            return True
        if extra.get("labeled_at"):
            return True
    return False


def is_legacy(metadata: dict[str, Any]) -> bool:
    """True for pre-determinism events lacking both end_ts and recorded_at."""

    return not metadata.get("end_ts") and not metadata.get("recorded_at")


def _source_exists(metadata: dict[str, Any]) -> bool:
    for key in ("source_path", "source_id", "sanitized_source_id"):
        if _source_value_exists(metadata.get(key)):
            return True
    return False


def _source_value_exists(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    for candidate in _source_path_candidates(value):
        try:
            if candidate.is_file():
                return True
        except OSError:
            continue
    return False


def _source_path_candidates(value: str) -> list[Path]:
    candidates = [Path(value)]
    parts = urlsplit(value)
    if parts.scheme == "file":
        file_path = unquote(parts.path or parts.netloc)
        if file_path:
            candidates.append(Path(file_path))
    if value.startswith("file:"):
        file_path = unquote(value.removeprefix("file:"))
        if file_path:
            candidates.append(Path(file_path))
    return candidates


def _classify(metadata: dict[str, Any]) -> str:
    if has_human_signal(metadata):
        return KEEP_LABELED
    if not is_legacy(metadata):
        return KEEP_DETERMINISTIC
    if not _source_exists(metadata):
        return KEEP_SOURCE_MISSING
    return REMOVE_LEGACY


def plan_cleanup(dataset_dir: str | Path) -> CleanupReport:
    """Classify every event without touching disk."""

    root = Path(dataset_dir)
    report = CleanupReport()
    if not root.is_dir():
        return report
    for metadata_path in sorted(root.glob("*/*/events/*/metadata.json")):
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        event_dir = metadata_path.parent
        report.items.append(
            CleanupItem(
                event_dir=event_dir,
                event_id=str(metadata.get("event_id") or event_dir.name),
                classification=_classify(metadata),
                source_id=_optional_source_id(metadata),
            )
        )
    return report


def cleanup_legacy_events(
    dataset_dir: str | Path,
    *,
    dry_run: bool = True,
) -> CleanupReport:
    """Quarantine legacy unlabeled duplicates, preserving all reviewed work.

    Returns the classification report. When ``dry_run`` is False, removable
    events are moved into ``<dataset>/.trash/<timestamp>/`` (reversible) and the
    report records each destination. A belt-and-suspenders check guarantees no
    event with human signal is ever moved.
    """

    root = Path(dataset_dir)
    report = plan_cleanup(root)
    if dry_run:
        return report

    removable = report.removable
    if not removable:
        return report

    # Belt-and-suspenders: nothing with a human signal may be in the move set.
    protected = {item.event_dir.resolve() for item in report.kept_labeled}
    safe_removable = [
        item for item in removable if item.event_dir.resolve() not in protected
    ]

    trash_dir = root / ".trash" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report.trash_dir = trash_dir
    report.applied = True
    for item in safe_removable:
        try:
            relative = item.event_dir.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        dest = trash_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(item.event_dir), str(dest))
        except OSError:
            continue
        item.moved_to = dest

    # Post-condition: every labeled event must still be on disk, untouched.
    for item in report.kept_labeled:
        if not item.event_dir.exists():
            raise RuntimeError(
                f"cleanup invariant violated: labeled event {item.event_id} "
                f"was removed from {item.event_dir}"
            )
    return report


def _optional_source_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("sanitized_source_id")
    return str(value) if value else None
