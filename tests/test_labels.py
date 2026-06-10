from __future__ import annotations

from pathlib import Path

import pytest

from detectivepotty.labels import (
    Behavior,
    ClipLabels,
    Dog,
    LabelRange,
    load_labels,
    save_labels,
)


def test_label_range_validates_and_round_trips() -> None:
    rng = LabelRange(
        start_frame=10,
        end_frame=40,
        start_s=1.0,
        end_s=4.0,
        behavior=Behavior.PEE,
        dog=Dog.GROMIT,
        track_id="2",
    )
    assert rng.is_trainable
    assert list(rng.frames()) == list(range(10, 41))
    restored = LabelRange.from_dict(rng.to_dict())
    assert restored.behavior is Behavior.PEE
    assert restored.dog is Dog.GROMIT
    assert restored.track_id == "2"
    assert restored.time_basis == "clip_frames"


def test_label_range_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        LabelRange(
            start_frame=40,
            end_frame=10,
            start_s=4.0,
            end_s=1.0,
            behavior=Behavior.POOP,
        )


def test_excluded_behavior_not_trainable() -> None:
    rng = LabelRange(0, 5, 0.0, 0.5, behavior=Behavior.EXCLUDED)
    assert not rng.is_trainable


def test_dog_enum_normalizes_aliases() -> None:
    assert Dog("WALL-E") is Dog.WALLE
    assert Dog("wall_e") is Dog.WALLE
    assert Dog("Gromit") is Dog.GROMIT


def test_clip_labels_save_load_roundtrip(tmp_path: Path) -> None:
    labels = ClipLabels(
        ranges=[
            LabelRange(0, 10, 0.0, 1.0, behavior=Behavior.PEE, dog=Dog.APOLLO, track_id="1"),
            LabelRange(20, 25, 2.0, 2.5, behavior=Behavior.EXCLUDED),
        ]
    )
    save_labels(labels, tmp_path)
    loaded = load_labels(tmp_path)
    assert loaded.clip == "clip.mp4"
    assert len(loaded.ranges) == 2
    assert list(loaded.trainable_ranges())[0].dog is Dog.APOLLO
    assert loaded.ranges[1].behavior is Behavior.EXCLUDED
