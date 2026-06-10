from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from detectivepotty.config import CameraConfig, Config, GlobalSettings
from detectivepotty.events import Detection, Track, TriggerReason
from detectivepotty.geometry import BBox
from detectivepotty.potty_event import PottyCandidate, PottyLifecycle
from detectivepotty.recording.recorder import EventRecorder
from detectivepotty.recording.reconcile import (
    PriorEvent,
    decide_carry,
    match_priors,
    parse_metadata_ts,
)
from detectivepotty.sources.base import Frame

BASE_TS = datetime(2026, 6, 6, 9, 10, 47, tzinfo=timezone.utc)
SOURCE_ID = "rtsp://user:pass@cam.local/stream?token=secret&keep=1"


# --------------------------------------------------------------------------- #
# Pure logic: matching + carry decisions
# --------------------------------------------------------------------------- #


def make_prior(
    name: str,
    start_offset: float,
    *,
    end_offset: float | None = None,
    label: str = "unknown",
    label_status: str = "unlabeled",
    dog: str | None = None,
    note: str | None = None,
    protect_id: str | None = None,
    labeled_at: str | None = None,
) -> PriorEvent:
    start = BASE_TS + timedelta(seconds=start_offset)
    metadata: dict = {
        "event_id": name,
        "camera_id": "cam-1",
        "sanitized_source_id": "src",
        "utc_ts": start.isoformat(),
        "label": label,
        "label_status": label_status,
        "dog": dog,
    }
    if end_offset is not None:
        metadata["end_ts"] = (BASE_TS + timedelta(seconds=end_offset)).isoformat()
    if protect_id is not None:
        metadata["protect_event_id"] = protect_id
    extra: dict = {}
    if note is not None:
        extra["label_note"] = note
    if labeled_at is not None:
        extra["labeled_at"] = labeled_at
    metadata["extra"] = extra
    return PriorEvent(dir_path=Path(name), metadata=metadata)


def test_parse_metadata_ts_handles_z_suffix_and_naive() -> None:
    assert parse_metadata_ts("2026-06-06T09:10:47Z") == BASE_TS
    assert parse_metadata_ts("2026-06-06T09:10:47") == BASE_TS
    assert parse_metadata_ts(None) is None
    assert parse_metadata_ts("not-a-date") is None


def test_match_priors_interval_overlap() -> None:
    snapshot = [make_prior("a", 0.0, end_offset=2.0)]
    matched = match_priors(
        snapshot,
        start_ts=BASE_TS + timedelta(seconds=1),
        end_ts=BASE_TS + timedelta(seconds=3),
        protect_event_id=None,
        tolerance_s=0.0,
    )
    assert [p.event_id for p in matched] == ["a"]


def test_match_priors_source_offsets_match_across_different_anchors() -> None:
    # Prior was recorded against a totally different wall-clock anchor (its utc_ts
    # is years away), so wall-clock overlap and tolerance can never match it. The
    # in-clip source offsets are identical, so source-relative overlap must.
    prior = make_prior("a", 0.0, end_offset=2.0)
    prior.metadata["utc_ts"] = "2001-01-01T00:00:00+00:00"
    prior.metadata.pop("end_ts", None)
    prior.metadata["source_start_s"] = 10.0
    prior.metadata["source_end_s"] = 12.0

    matched = match_priors(
        [prior],
        start_ts=BASE_TS,
        end_ts=BASE_TS + timedelta(seconds=2),
        protect_event_id=None,
        tolerance_s=0.0,
        source_start_s=11.0,
        source_end_s=13.0,
    )
    assert [p.event_id for p in matched] == ["a"]


def test_match_priors_source_offsets_no_overlap() -> None:
    prior = make_prior("a", 0.0, end_offset=2.0)
    prior.metadata["source_start_s"] = 10.0
    prior.metadata["source_end_s"] = 12.0
    matched = match_priors(
        [prior],
        start_ts=BASE_TS + timedelta(seconds=100),
        end_ts=BASE_TS + timedelta(seconds=102),
        protect_event_id=None,
        tolerance_s=0.0,
        source_start_s=40.0,
        source_end_s=42.0,
    )
    assert matched == []


def test_match_priors_start_tolerance_fallback_without_end_ts() -> None:
    snapshot = [make_prior("a", 0.0)]  # no end_ts persisted
    within = match_priors(
        snapshot,
        start_ts=BASE_TS + timedelta(seconds=4),
        end_ts=None,
        protect_event_id=None,
        tolerance_s=5.0,
    )
    assert [p.event_id for p in within] == ["a"]

    outside = match_priors(
        snapshot,
        start_ts=BASE_TS + timedelta(seconds=20),
        end_ts=None,
        protect_event_id=None,
        tolerance_s=5.0,
    )
    assert outside == []


def test_match_priors_protect_id_takes_priority() -> None:
    snapshot = [make_prior("a", 999.0, protect_id="p1")]
    matched = match_priors(
        snapshot,
        start_ts=BASE_TS,
        end_ts=None,
        protect_event_id="p1",
        tolerance_s=5.0,
    )
    assert [p.event_id for p in matched] == ["a"]


def test_match_priors_does_not_cross_different_protect_id() -> None:
    snapshot = [make_prior("a", 0.0, protect_id="p1")]
    matched = match_priors(
        snapshot,
        start_ts=BASE_TS,
        end_ts=None,
        protect_event_id="p2",
        tolerance_s=5.0,
    )
    assert matched == []


def test_decide_carry_empty() -> None:
    result = decide_carry([])
    assert result.event_id is None
    assert result.carried == {}
    assert result.superseded == []
    assert result.conflict is False


def test_decide_carry_unlabeled_only_collapses() -> None:
    snapshot = [make_prior("a", 0.0), make_prior("b", 1.0)]
    result = decide_carry(snapshot)
    assert result.conflict is False
    assert result.carried == {}
    assert {p.event_id for p in result.superseded} == {"a", "b"}
    assert result.event_id in {"a", "b"}


def test_decide_carry_single_labeled_carries_fields() -> None:
    snapshot = [
        make_prior("a", 0.0),
        make_prior(
            "b",
            1.0,
            label="poop",
            label_status="labeled",
            dog="Apollo",
            note="solid",
            labeled_at="2026-06-06T10:00:00+00:00",
        ),
    ]
    result = decide_carry(snapshot)
    assert result.conflict is False
    assert result.event_id == "b"
    assert result.carried["label"] == "poop"
    assert result.carried["label_status"] == "labeled"
    assert result.carried["dog"] == "Apollo"
    assert result.carried["label_note"] == "solid"
    assert {p.event_id for p in result.superseded} == {"a", "b"}


def test_decide_carry_conflicting_labels_keeps_everything() -> None:
    snapshot = [
        make_prior("a", 0.0, label="pee", label_status="labeled"),
        make_prior("b", 1.0, label="poop", label_status="labeled"),
    ]
    result = decide_carry(snapshot)
    assert result.conflict is True
    assert result.carried == {}
    assert result.superseded == []
    assert result.event_id is None


# --------------------------------------------------------------------------- #
# End-to-end recorder rerun behavior
# --------------------------------------------------------------------------- #


def camera_config(**overrides: object) -> CameraConfig:
    values = {
        "id": "cam-1",
        "name": "Backyard Grass",
        "pre_roll_s": 2.0,
        "post_roll_s": 3.0,
        "sample_rate_fps": 1.0,
    }
    values.update(overrides)
    return CameraConfig(**values)


def make_config(tmp_path: Path, camera: CameraConfig, **global_overrides: object) -> Config:
    settings = {
        "dataset_dir": tmp_path,
        "model_name": "test-model.pt",
        "dogs": ["Apollo", "Gromit"],
    }
    settings.update(global_overrides)
    return Config(global_settings=GlobalSettings(**settings), cameras=[camera])


def make_frame(frame_idx: int) -> Frame:
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[8:32, 18:42] = (30, 150, 240)
    return Frame(
        bgr=bgr,
        frame_idx=frame_idx,
        mono_ts=float(frame_idx),
        wall_ts=BASE_TS + timedelta(seconds=frame_idx),
        source_id=SOURCE_ID,
    )


def make_candidate(start_offset: float = 1.0, end_offset: float = 2.0) -> PottyCandidate:
    detections = [Detection(BBox(18, 8, 42, 32), 0.88, "dog", 1, 1.0, BASE_TS)]
    track = Track(track_id="dog-1", detections=detections)
    return PottyCandidate(
        camera_id="cam-1",
        primary_track_id="dog-1",
        start_ts=BASE_TS + timedelta(seconds=start_offset),
        end_ts=BASE_TS + timedelta(seconds=end_offset),
        tracks=[track],
        detections=detections,
        trigger_reason=TriggerReason.YOLO,
        multi_dog=False,
        ambiguous=False,
        lifecycle=PottyLifecycle.EMITTED,
        stationary_duration_s=2.5,
        posture_summary={"dwell_duration_s": 6.0},
        near_miss=False,
        confidence=0.77,
    )


def event_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in dataset_dir.glob("*/*/events/*")
        if p.is_dir() and not p.name.endswith(".rerun-bak")
    )


def label_on_disk(
    event_dir: Path,
    *,
    label: str = "poop",
    status: str = "labeled",
    dog: str = "Apollo",
    note: str = "solid pile",
) -> None:
    meta_path = event_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["label"] = label
    metadata["label_status"] = status
    metadata["dog"] = dog
    extra = metadata.setdefault("extra", {})
    extra["label_note"] = note
    extra["labeled_at"] = "2026-06-06T10:00:00+00:00"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def read_metadata(event_dir: Path) -> dict:
    return json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))


def test_rerun_with_drift_dedupes_and_carries_label(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera)
    frames = [make_frame(idx) for idx in range(5)]

    first_dir = EventRecorder(cfg).record(make_candidate(), frames, camera)
    label_on_disk(first_dir)
    first_id = read_metadata(first_dir)["event_id"]

    # A fresh recorder == a new run. Shift start by 1s to simulate window drift,
    # so the rerun lands in a new directory and must delete the old one.
    second_dir = EventRecorder(cfg).record(
        make_candidate(start_offset=2.0, end_offset=3.0), frames, camera
    )

    dirs = event_dirs(tmp_path)
    assert dirs == [second_dir]
    assert not first_dir.exists()

    meta = read_metadata(second_dir)
    assert meta["event_id"] == first_id
    assert meta["label"] == "poop"
    assert meta["label_status"] == "labeled"
    assert meta["dog"] == "Apollo"
    assert meta["extra"]["label_note"] == "solid pile"
    assert (second_dir / "clip.mp4").exists()


def test_rerun_in_place_collision_backup_swap(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera)
    frames = [make_frame(idx) for idx in range(5)]

    first_dir = EventRecorder(cfg).record(make_candidate(), frames, camera)
    label_on_disk(first_dir, label="pee", dog="Gromit", note="quick wee")

    # Same candidate + carried event_id => same target path (in-place overwrite).
    second_dir = EventRecorder(cfg).record(make_candidate(), frames, camera)

    assert second_dir == first_dir
    assert event_dirs(tmp_path) == [second_dir]
    assert not any(p.name.endswith(".rerun-bak") for p in tmp_path.glob("*/*/events/*"))
    meta = read_metadata(second_dir)
    assert meta["label"] == "pee"
    assert meta["dog"] == "Gromit"
    assert (second_dir / "clip.mp4").exists()


def test_same_run_does_not_self_match(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera)
    frames = [make_frame(idx) for idx in range(5)]
    recorder = EventRecorder(cfg)

    # Two distinct events within tolerance in ONE run must both survive.
    recorder.record(make_candidate(start_offset=1.0, end_offset=2.0), frames, camera)
    recorder.record(make_candidate(start_offset=1.5, end_offset=2.5), frames, camera)

    assert len(event_dirs(tmp_path)) == 2


def test_protect_recording_carried_forward(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera)
    frames = [make_frame(idx) for idx in range(5)]

    first_dir = EventRecorder(cfg).record(make_candidate(), frames, camera)
    (first_dir / "protect_recording.mp4").write_bytes(b"protect-bytes")
    label_on_disk(first_dir)

    second_dir = EventRecorder(cfg).record(
        make_candidate(start_offset=2.0, end_offset=3.0), frames, camera
    )

    assert event_dirs(tmp_path) == [second_dir]
    carried = second_dir / "protect_recording.mp4"
    assert carried.exists()
    assert carried.read_bytes() == b"protect-bytes"


def test_dedupe_disabled_appends(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera, dedupe_reruns=False)
    frames = [make_frame(idx) for idx in range(5)]

    EventRecorder(cfg).record(make_candidate(), frames, camera)
    label_on_disk(event_dirs(tmp_path)[0])
    EventRecorder(cfg).record(
        make_candidate(start_offset=2.0, end_offset=3.0), frames, camera
    )

    assert len(event_dirs(tmp_path)) == 2


def test_conflicting_labeled_priors_are_kept(tmp_path: Path) -> None:
    camera = camera_config()
    cfg = make_config(tmp_path, camera)
    frames = [make_frame(idx) for idx in range(5)]

    # Two labeled events with DIFFERENT labels, recorded with dedupe off so both
    # persist within tolerance of each other.
    cfg_off = make_config(tmp_path, camera, dedupe_reruns=False)
    d1 = EventRecorder(cfg_off).record(
        make_candidate(start_offset=1.0, end_offset=2.0), frames, camera
    )
    label_on_disk(d1, label="pee")
    d2 = EventRecorder(cfg_off).record(
        make_candidate(start_offset=1.2, end_offset=2.2), frames, camera
    )
    label_on_disk(d2, label="poop")

    # A rerun matching both must keep everything (conflict guard) and still add
    # its own event rather than delete a human-labeled one.
    EventRecorder(cfg).record(
        make_candidate(start_offset=1.1, end_offset=2.1), frames, camera
    )

    assert d1.exists()
    assert d2.exists()
    assert len(event_dirs(tmp_path)) == 3


# --------------------------------------------------------------------------- #
# One-time dedupe of existing duplicates
# --------------------------------------------------------------------------- #


def _make_existing_duplicates(tmp_path: Path, offsets: list[float]) -> list[Path]:
    camera = camera_config()
    cfg_off = make_config(tmp_path, camera, dedupe_reruns=False)
    frames = [make_frame(idx) for idx in range(5)]
    dirs = []
    for offset in offsets:
        dirs.append(
            EventRecorder(cfg_off).record(
                make_candidate(start_offset=offset, end_offset=offset + 1.0),
                frames,
                camera,
            )
        )
    return dirs


def test_dedupe_dataset_collapses_and_carries_label(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import dedupe_dataset

    dirs = _make_existing_duplicates(tmp_path, [1.0, 1.2, 1.4])
    label_on_disk(dirs[1], label="poop", dog="Apollo")

    actions = dedupe_dataset(tmp_path, tolerance_s=5.0, dry_run=False)

    assert len(actions) == 1
    assert actions[0].conflict is False
    remaining = event_dirs(tmp_path)
    assert len(remaining) == 1
    meta = read_metadata(remaining[0])
    assert meta["label"] == "poop"
    assert meta["dog"] == "Apollo"


def test_dedupe_dataset_dry_run_changes_nothing(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import dedupe_dataset

    _make_existing_duplicates(tmp_path, [1.0, 1.2, 1.4])
    before = event_dirs(tmp_path)

    actions = dedupe_dataset(tmp_path, tolerance_s=5.0, dry_run=True)

    assert len(actions) == 1
    assert len(actions[0].removed) == 2
    assert event_dirs(tmp_path) == before


def test_dedupe_dataset_skips_conflicting_labels(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import dedupe_dataset

    dirs = _make_existing_duplicates(tmp_path, [1.0, 1.2])
    label_on_disk(dirs[0], label="pee")
    label_on_disk(dirs[1], label="poop")

    actions = dedupe_dataset(tmp_path, tolerance_s=5.0, dry_run=False)

    assert len(actions) == 1
    assert actions[0].conflict is True
    assert len(event_dirs(tmp_path)) == 2


def test_cluster_events_does_not_chain_unboundedly() -> None:
    from detectivepotty.recording.reconcile import cluster_events

    # Three events 6s apart with end+1; tolerance 5 must keep them separate
    # rather than chaining anchor -> running_end -> ... into one cluster.
    priors = [
        make_prior("a", 0.0, end_offset=1.0),
        make_prior("b", 6.0, end_offset=7.0),
        make_prior("c", 12.0, end_offset=13.0),
    ]
    clusters = cluster_events(priors, tolerance_s=5.0)

    assert sorted(len(c) for c in clusters) == [1, 1, 1]


def test_cluster_events_groups_overlapping_reruns() -> None:
    from detectivepotty.recording.reconcile import cluster_events

    priors = [
        make_prior("a", 0.0, end_offset=2.0),
        make_prior("b", 1.0, end_offset=3.0),
        make_prior("c", 2.0, end_offset=4.0),
    ]
    clusters = cluster_events(priors, tolerance_s=5.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_apply_carried_reports_read_failure(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import _apply_carried_to_metadata_file

    missing = tmp_path / "no-such-event"
    missing.mkdir()
    assert _apply_carried_to_metadata_file(missing, {"label": "poop"}) is False

    (missing / "metadata.json").write_text("{}", encoding="utf-8")
    assert _apply_carried_to_metadata_file(missing, {"label": "poop"}) is True


def test_preserve_protect_recording_returns_status(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import (
        PROTECT_RECORDING_NAME,
        preserve_protect_recording,
    )

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()

    # Nothing to move -> success.
    assert preserve_protect_recording(old, new) is True

    (old / PROTECT_RECORDING_NAME).write_bytes(b"data")
    assert preserve_protect_recording(old, new) is True
    assert (new / PROTECT_RECORDING_NAME).is_file()
    assert not (old / PROTECT_RECORDING_NAME).exists()


def test_dedupe_dataset_skips_records_missing_identity(tmp_path: Path) -> None:
    from detectivepotty.recording.reconcile import dedupe_dataset

    dirs = _make_existing_duplicates(tmp_path, [1.0, 1.2])
    # Strip the grouping identity from both events: they must not be grouped or
    # deleted as duplicates.
    for event_dir in dirs:
        meta = read_metadata(event_dir)
        meta["camera_id"] = ""
        (event_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    actions = dedupe_dataset(tmp_path, tolerance_s=5.0, dry_run=False)

    assert actions == []
    assert len(event_dirs(tmp_path)) == 2
