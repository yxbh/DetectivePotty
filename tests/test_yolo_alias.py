"""Tests for dog-alias class acceptance + class-agnostic NMS (feature D).

All offline: fake YOLO results, no model/GPU/network. Covers the pure
``normalize_alias_classes`` / ``nms_dog_aliases`` helpers and the
``DogDetector._result_to_detections`` filter+NMS path, including the invariant
that an empty alias set reproduces the legacy dog-only behavior byte-for-byte.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

import detectivepotty.detect.yolo as yolo_mod
from detectivepotty.detect.yolo import (
    DogDetector,
    clear_model_cache,
    normalize_alias_classes,
    nms_dog_aliases,
)
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox


@pytest.fixture(autouse=True)
def _isolate_cache():
    clear_model_cache()
    yield
    clear_model_cache()


_WALL = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _det(x1, y1, x2, y2, conf, class_name):
    return Detection(
        bbox=BBox(x1, y1, x2, y2),
        confidence=conf,
        class_name=class_name,
        frame_idx=0,
        mono_ts=0.0,
        wall_ts=_WALL,
    )


# --- normalize_alias_classes -------------------------------------------------


def test_normalize_alias_classes_lowercases_dedupes_drops_dog_and_blanks():
    out = normalize_alias_classes(["Sheep", "sheep", " COW ", "dog", "", "Zebra"])
    assert out == ("sheep", "cow", "zebra")


def test_normalize_alias_classes_empty():
    assert normalize_alias_classes([]) == ()


# --- nms_dog_aliases (pure) --------------------------------------------------


def test_nms_collapses_same_animal_dog_wins_tiebreak():
    # dog + alias on (near) the same box: the higher-confidence one survives, and on
    # near-equal confidence the dog-class read wins via the priority boost.
    dog = _det(0, 0, 10, 10, 0.60, "dog")
    sheep = _det(0, 0, 10, 10, 0.58, "sheep")
    kept = nms_dog_aliases([sheep, dog], 0.65)
    assert len(kept) == 1
    assert kept[0].class_name == "dog"


def test_nms_prefer_dog_on_exact_tie():
    dog = _det(0, 0, 10, 10, 0.50, "dog")
    bear = _det(0, 0, 10, 10, 0.50, "bear")
    kept = nms_dog_aliases([bear, dog], 0.65)
    assert len(kept) == 1
    assert kept[0].class_name == "dog"


def test_nms_keeps_distinct_animals():
    # Two boxes with no overlap (IoU ~ 0) are both kept — two dogs survive.
    a = _det(0, 0, 10, 10, 0.58, "dog")
    b = _det(100, 100, 110, 110, 0.50, "sheep")
    kept = nms_dog_aliases([a, b], 0.65)
    assert len(kept) == 2


def test_nms_two_missed_both_aliases_distinct():
    sheep = _det(0, 0, 10, 10, 0.50, "sheep")
    zebra = _det(100, 100, 110, 110, 0.40, "zebra")
    kept = nms_dog_aliases([sheep, zebra], 0.65)
    assert {d.class_name for d in kept} == {"sheep", "zebra"}


def test_nms_output_sorted_by_real_confidence_not_boosted():
    # A high-confidence alias outranks a lower-confidence dog in OUTPUT order even
    # though the dog gets a suppression-priority boost (boost only governs suppression).
    sheep = _det(0, 0, 10, 10, 0.90, "sheep")
    dog = _det(200, 200, 210, 210, 0.55, "dog")
    kept = nms_dog_aliases([dog, sheep], 0.65)
    assert [d.class_name for d in kept] == ["sheep", "dog"]


def test_nms_single_and_empty_passthrough():
    one = [_det(0, 0, 1, 1, 0.5, "dog")]
    assert nms_dog_aliases(one, 0.65) == one
    assert nms_dog_aliases([], 0.65) == []


# --- DogDetector._result_to_detections --------------------------------------


class _Arr:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arr(np.asarray(xyxy, dtype=float).reshape(-1, 4))
        self.conf = _Arr(np.asarray(conf, dtype=float))
        self.cls = _Arr(np.asarray(cls, dtype=float))

    def __len__(self):
        return len(self.conf.value)


# COCO-ish names with the dog-confusable aliases + clutter we care about.
_NAMES = {0: "dog", 16: "dog", 18: "sheep", 22: "zebra", 58: "potted plant"}


class _Result:
    names = _NAMES

    def __init__(self, boxes):
        self.boxes = boxes


def _make_detector(monkeypatch, *, alias_classes=(), conf=0.25):
    monkeypatch.setattr(yolo_mod, "resolve_device", lambda d: d)

    def fake_load(self, candidates):  # noqa: ANN001
        self.model_name = candidates[0]

        class _M:
            names = _NAMES

        return _M()

    monkeypatch.setattr(DogDetector, "_load_model", fake_load)
    return DogDetector(
        model_name="m.pt",
        long_edge=64,
        conf_threshold=conf,
        device="cpu",
        alias_classes=alias_classes,
    )


def _to_dets(det, result, w=320, h=240):
    return det._result_to_detections(result, w, h, 0, 0.0, _WALL)


def test_result_no_aliases_filters_to_dog_only_and_sorts(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=())
    result = _Result(
        _Boxes(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
            [0.4, 0.9, 0.7],
            [0, 18, 58],  # dog, sheep, potted plant
        )
    )
    out = _to_dets(det, result)
    # Only the dog survives; aliases off => sheep dropped, clutter dropped.
    assert [d.class_name for d in out] == ["dog"]
    assert out[0].confidence == pytest.approx(0.4)


def test_result_no_aliases_sorted_by_confidence_desc(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=())
    result = _Result(
        _Boxes(
            [[0, 0, 10, 10], [100, 100, 110, 110]],
            [0.3, 0.8],
            [0, 16],  # both dog
        )
    )
    out = _to_dets(det, result)
    assert [d.confidence for d in out] == [pytest.approx(0.8), pytest.approx(0.3)]


def test_result_alias_accepted_keeps_real_class_name(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=["sheep"])
    result = _Result(_Boxes([[20, 20, 30, 30]], [0.6], [18]))  # sheep, no dog
    out = _to_dets(det, result)
    assert len(out) == 1
    # class_name preserved for training-set audit even though it's accepted as a dog.
    assert out[0].class_name == "sheep"


def test_result_clutter_never_suppresses_dog(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=["sheep"])
    # potted plant overlaps the dog heavily but is NOT in the accepted set, so it's
    # filtered before NMS and can never suppress the dog.
    result = _Result(
        _Boxes(
            [[0, 0, 50, 50], [0, 0, 50, 50]],
            [0.99, 0.5],
            [58, 0],  # potted plant (clutter), dog
        )
    )
    out = _to_dets(det, result)
    assert [d.class_name for d in out] == ["dog"]


def test_result_dog_plus_alias_same_animal_collapses(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=["sheep"])
    result = _Result(
        _Boxes(
            [[0, 0, 20, 20], [0, 0, 20, 20]],
            [0.6, 0.55],
            [0, 18],  # dog + sheep, same box
        )
    )
    out = _to_dets(det, result)
    assert len(out) == 1
    assert out[0].class_name == "dog"


def test_result_two_dogs_one_missed_to_alias_keeps_both(monkeypatch):
    det = _make_detector(monkeypatch=monkeypatch, alias_classes=["sheep"])
    result = _Result(
        _Boxes(
            [[0, 0, 20, 20], [200, 200, 220, 220]],
            [0.58, 0.50],
            [0, 18],  # dog + a far-away sheep (the other dog)
        )
    )
    out = _to_dets(det, result)
    assert len(out) == 2
    assert {d.class_name for d in out} == {"dog", "sheep"}
