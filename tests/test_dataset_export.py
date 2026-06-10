from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from detectivepotty.dataset_export import (
    assign_split,
    export_dataset,
    sample_range_frames,
)
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.labels import Behavior, ClipLabels, Dog, LabelRange, save_labels


# --- pure helpers -----------------------------------------------------------


def test_sample_range_frames_strides_by_time() -> None:
    frames = sample_range_frames(0, 20, fps=10.0, stride_s=0.3, max_frames=40)
    assert frames == [0, 3, 6, 9, 12, 15, 18]


def test_sample_range_frames_caps_and_thins() -> None:
    frames = sample_range_frames(0, 100, fps=10.0, stride_s=0.1, max_frames=5)
    assert len(frames) == 5
    assert frames == sorted(set(frames))
    assert frames[0] == 0


def test_sample_range_frames_always_returns_first() -> None:
    assert sample_range_frames(7, 7, fps=10.0, stride_s=5.0, max_frames=40) == [7]


def test_assign_split_is_deterministic_and_partitions() -> None:
    key = "cam|2026-06-06"
    assert assign_split(key, 0.2) == assign_split(key, 0.2)
    assert assign_split(key, 0.0) == "train"
    assert assign_split(key, 1.0) == "val"


# --- exporter with fakes ----------------------------------------------------


class FakeExportCapture:
    def __init__(self, n_frames: int, size=(48, 64)) -> None:
        self.height, self.width = size
        self._remaining = deque(range(n_frames))

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._remaining:
            idx = self._remaining.popleft()
            return True, np.full((self.height, self.width, 3), idx % 255, np.uint8)
        return False, None

    def release(self) -> None:
        pass


class FixedBoxDetector:
    """Always returns one dog box at ``box`` (overlaps the reference)."""

    def __init__(self, box: tuple[float, float, float, float] = (10, 10, 30, 30)) -> None:
        self.box = box

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(*self.box),
                confidence=0.9,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=0.0,
                wall_ts=datetime.now(timezone.utc),
            )
        ]


def _write_clip(
    clip_dir: Path,
    *,
    track_id: str = "1",
    ref_box=(10, 10, 30, 30),
    fps: float = 10.0,
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.mp4").write_bytes(b"fake")
    detections = [
        {
            "clip_frame_idx": i,
            "source_frame_idx": i,
            "time_s": i / fps,
            "track_id": track_id,
            "bbox": {"x1": ref_box[0], "y1": ref_box[1], "x2": ref_box[2], "y2": ref_box[3]},
            "confidence": 0.9,
        }
        for i in (0, 5, 10, 15, 20)
    ]
    meta = {
        "fps": fps,
        "source_id": f"recording-{clip_dir.name}.mp4",
        "source_span_start_utc": "2026-06-06T09:00:00+00:00",
        "track_id": track_id,
        "detections": detections,
    }
    (clip_dir / "metadata.json").write_text(json.dumps(meta))


def test_export_dataset_writes_behavior_and_dog_trees(tmp_path: Path) -> None:
    clips_root = tmp_path / "harvest"
    clip_dir = clips_root / "span_a"
    _write_clip(clip_dir)
    save_labels(
        ClipLabels(
            ranges=[
                LabelRange(0, 18, 0.0, 1.8, behavior=Behavior.PEE, dog=Dog.GROMIT, track_id="1")
            ]
        ),
        clip_dir,
    )

    out_dir = tmp_path / "export"
    stats = export_dataset(
        clips_root,
        out_dir,
        detector=FixedBoxDetector(),
        sample_stride_s=0.3,
        max_frames_per_range=40,
        val_fraction=0.0,  # force train for a stable assertion
        capture_factory=lambda _p: FakeExportCapture(40),
    )

    assert stats.clips == 1
    assert stats.crops_written == 7  # frames 0,3,...,18
    assert stats.behavior_counts == {"pee": 7}
    assert stats.dog_counts == {"gromit": 7}
    behavior_crops = list((out_dir / "behavior" / "train" / "pee").glob("*.jpg"))
    dog_crops = list((out_dir / "dog" / "train" / "gromit").glob("*.jpg"))
    assert len(behavior_crops) == 7
    assert len(dog_crops) == 7
    manifest = (out_dir / "manifest.csv").read_text().strip().splitlines()
    assert manifest[0].startswith("crop_path,")
    assert len(manifest) == 1 + 7
    assert (out_dir / "export_stats.json").exists()


def test_export_dataset_excludes_excluded_and_unknown_dog(tmp_path: Path) -> None:
    clips_root = tmp_path / "harvest"
    clip_dir = clips_root / "span_b"
    _write_clip(clip_dir)
    save_labels(
        ClipLabels(
            ranges=[
                LabelRange(0, 6, 0.0, 0.6, behavior=Behavior.NOT_POTTY, dog=Dog.UNKNOWN, track_id="1"),
                LabelRange(10, 16, 1.0, 1.6, behavior=Behavior.EXCLUDED, track_id="1"),
            ]
        ),
        clip_dir,
    )

    out_dir = tmp_path / "export"
    stats = export_dataset(
        clips_root,
        out_dir,
        detector=FixedBoxDetector(),
        sample_stride_s=0.3,
        val_fraction=0.0,
        capture_factory=lambda _p: FakeExportCapture(40),
    )

    assert stats.excluded_ranges == 1
    # not_potty crops exist; unknown dog contributes no dog-tree crops.
    assert stats.behavior_counts.get("not_potty", 0) > 0
    assert stats.dog_counts == {}
    assert not (out_dir / "dog").exists()
    assert not (out_dir / "behavior" / "train" / "excluded").exists()


def test_export_dataset_drops_unmatched_track(tmp_path: Path) -> None:
    clips_root = tmp_path / "harvest"
    clip_dir = clips_root / "span_c"
    _write_clip(clip_dir, ref_box=(10, 10, 30, 30))
    save_labels(
        ClipLabels(
            ranges=[
                LabelRange(0, 9, 0.0, 0.9, behavior=Behavior.POOP, dog=Dog.WALLE, track_id="1")
            ]
        ),
        clip_dir,
    )

    out_dir = tmp_path / "export"
    # Detector box is far from the reference -> IoU 0 -> dropped.
    stats = export_dataset(
        clips_root,
        out_dir,
        detector=FixedBoxDetector(box=(200, 200, 240, 240)),
        sample_stride_s=0.3,
        min_iou=0.3,
        val_fraction=0.0,
        capture_factory=lambda _p: FakeExportCapture(40),
    )

    assert stats.crops_written == 0
    assert stats.dropped_unmatched > 0
