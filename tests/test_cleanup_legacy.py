from __future__ import annotations

import json
from pathlib import Path

from detectivepotty.recording.cleanup import (
    KEEP_DETERMINISTIC,
    KEEP_LABELED,
    KEEP_SOURCE_MISSING,
    REMOVE_LEGACY,
    cleanup_legacy_events,
    plan_cleanup,
)


def write_event(
    dataset: Path,
    *,
    event_id: str,
    source_path: str,
    camera: str = "backyard",
    date: str = "2026-06-06",
    **metadata: object,
) -> Path:
    event_dir = dataset / camera / date / "events" / f"evt_{event_id}"
    event_dir.mkdir(parents=True)
    base = {
        "event_id": event_id,
        "camera_id": camera,
        "sanitized_source_id": source_path,
        "utc_ts": "2026-06-06T00:00:00+00:00",
        "label": "unknown",
        "label_status": "unlabeled",
    }
    base.update(metadata)
    (event_dir / "metadata.json").write_text(
        json.dumps(base, indent=2, sort_keys=True), encoding="utf-8"
    )
    (event_dir / "clip.mp4").write_bytes(b"clip")
    return event_dir


def make_source(tmp_path: Path, name: str = "clip.mp4") -> str:
    source = tmp_path / "videos" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")
    return str(source)


def test_plan_classifies_each_event(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)

    legacy = write_event(dataset, event_id="legacy", source_path=src)
    labeled = write_event(
        dataset,
        event_id="labeled",
        source_path=src,
        label="pee",
        label_status="labeled",
        dog="Apollo",
    )
    deterministic = write_event(
        dataset,
        event_id="fresh",
        source_path=src,
        end_ts="2026-06-06T00:00:05+00:00",
        recorded_at="2026-06-06T01:00:00+00:00",
    )
    missing = write_event(
        dataset, event_id="orphan", source_path=str(tmp_path / "gone.mp4")
    )

    report = plan_cleanup(dataset)
    by_dir = {item.event_dir: item.classification for item in report.items}

    assert by_dir[legacy] == REMOVE_LEGACY
    assert by_dir[labeled] == KEEP_LABELED
    assert by_dir[deterministic] == KEEP_DETERMINISTIC
    assert by_dir[missing] == KEEP_SOURCE_MISSING


def test_plan_checks_source_path_when_sanitized_id_is_not_a_path(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)
    legacy = write_event(dataset, event_id="legacy", source_path="file:backyard")
    metadata_path = legacy / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_path"] = src
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = plan_cleanup(dataset)

    assert report.items[0].classification == REMOVE_LEGACY


def test_plan_accepts_file_url_source_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)
    legacy = write_event(dataset, event_id="legacy", source_path=f"file:{src}")

    report = plan_cleanup(dataset)
    by_dir = {item.event_dir: item.classification for item in report.items}

    assert by_dir[legacy] == REMOVE_LEGACY


def test_dry_run_moves_nothing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)
    legacy = write_event(dataset, event_id="legacy", source_path=src)

    report = cleanup_legacy_events(dataset, dry_run=True)

    assert report.applied is False
    assert legacy.exists()
    assert len(report.removable) == 1
    assert report.removable[0].moved_to is None


def test_apply_quarantines_and_is_reversible(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)
    legacy = write_event(dataset, event_id="legacy", source_path=src)
    labeled = write_event(
        dataset,
        event_id="labeled",
        source_path=src,
        label="poop",
        label_status="labeled",
    )

    report = cleanup_legacy_events(dataset, dry_run=False)

    assert report.applied is True
    # Removable event was moved out of its original location...
    assert not legacy.exists()
    moved = report.removable[0].moved_to
    assert moved is not None and moved.exists()
    assert (moved / "clip.mp4").exists()
    # ...into the dataset trash dir (reversible), and the labeled event stayed.
    assert report.trash_dir is not None
    assert report.trash_dir in moved.parents
    assert labeled.exists()


def test_apply_with_no_removable_keeps_everything(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    src = make_source(tmp_path)
    labeled = write_event(
        dataset,
        event_id="labeled",
        source_path=src,
        label="pee",
        label_status="labeled",
    )

    report = cleanup_legacy_events(dataset, dry_run=False)

    assert report.removable == []
    assert labeled.exists()
