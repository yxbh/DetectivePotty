"""Tests for DogDetector.detect_batch and its fallback ladder."""

from __future__ import annotations

import numpy as np
import pytest

import detectivepotty.detect.yolo as yolo_mod
from detectivepotty.detect.yolo import DogDetector, FrameMeta, clear_model_cache


@pytest.fixture(autouse=True)
def _isolate_cache():
    clear_model_cache()
    yield
    clear_model_cache()


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arr(np.asarray(xyxy, dtype=float).reshape(-1, 4))
        self.conf = _Arr(np.asarray(conf, dtype=float))
        self.cls = _Arr(np.asarray(cls, dtype=float))

    def __len__(self):
        return len(self.conf.value)


class _Arr:
    """Mimics the tensor surface ``_iter_boxes`` touches (detach/cpu/numpy)."""

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Result:
    names = {0: "dog", 1: "cat"}

    def __init__(self, boxes):
        self.boxes = boxes


class _BatchModel:
    """Fake YOLO returning one ``_Result`` per input frame.

    ``per_frame`` maps a frame's first pixel value -> (xyxy, conf, cls) box list.
    """

    def __init__(self, per_frame, supports_batch=True):
        self.per_frame = per_frame
        self.supports_batch = supports_batch
        self.predict_calls: list[int] = []
        self.devices: list[str] = []

    def predict(self, frames, imgsz, conf, device, verbose):  # noqa: ANN001
        is_list = isinstance(frames, list)
        n = len(frames) if is_list else 1
        if is_list and n > 1 and not self.supports_batch:
            raise RuntimeError("fixed batch=1")
        self.predict_calls.append(n)
        self.devices.append(device)
        batch = frames if is_list else [frames]
        results = []
        for frame in batch:
            key = int(np.asarray(frame).flat[0])
            spec = self.per_frame.get(key, ([], [], []))
            results.append(_Result(_Boxes(*spec)))
        return results


def _make_detector(monkeypatch, model, device="mps"):
    monkeypatch.setattr(yolo_mod, "resolve_device", lambda d: d)

    def fake_load(self, candidates):  # noqa: ANN001
        self.model_name = candidates[0]
        return model

    monkeypatch.setattr(DogDetector, "_load_model", fake_load)
    return DogDetector(model_name="m.pt", long_edge=64, conf_threshold=0.25, device=device)


def _frame(value, h=20, w=30):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0, 0, 0] = value
    return arr


def test_detect_batch_matches_per_frame_detect(monkeypatch):
    per_frame = {
        1: ([[1, 2, 11, 12]], [0.9], [0]),  # one dog
        2: ([], [], []),  # nothing
        3: ([[0, 0, 5, 5], [2, 2, 8, 8]], [0.8, 0.5], [0, 0]),  # two dogs
    }
    model = _BatchModel(per_frame)
    det = _make_detector(monkeypatch, model)

    frames = [_frame(1), _frame(2), _frame(3)]
    metas = [FrameMeta(frame_idx=i) for i in range(3)]
    batched = det.detect_batch(frames, metas)

    # One predict call covering all three frames.
    assert model.predict_calls == [3]

    # Equivalent to calling detect() per frame.
    singles = [det.detect(f, frame_idx=i) for i, f in enumerate(frames)]
    assert [len(b) for b in batched] == [1, 0, 2]
    for b, s in zip(batched, singles):
        assert [d.bbox for d in b] == [d.bbox for d in s]
        assert [d.confidence for d in b] == [d.confidence for d in s]
    # frame_idx carried through from metas.
    assert batched[0][0].frame_idx == 0
    assert batched[2][0].frame_idx == 2


def test_detect_batch_filters_non_dog_and_low_conf(monkeypatch):
    per_frame = {
        5: ([[0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 7, 7]], [0.9, 0.1, 0.9], [0, 0, 1]),
    }
    det = _make_detector(monkeypatch, _BatchModel(per_frame))
    out = det.detect_batch([_frame(5)])
    # Drops the 0.1-conf dog and the cat; keeps the 0.9 dog.
    assert len(out[0]) == 1
    assert out[0][0].class_name == "dog"


def test_detect_batch_falls_back_per_frame_on_unsupported_batch(monkeypatch):
    per_frame = {1: ([[1, 1, 9, 9]], [0.7], [0]), 2: ([[2, 2, 8, 8]], [0.6], [0])}
    model = _BatchModel(per_frame, supports_batch=False)
    det = _make_detector(monkeypatch, model)

    out = det.detect_batch([_frame(1), _frame(2)])
    assert [len(o) for o in out] == [1, 1]
    # First multi-frame predict raised; fell back to two single-frame predicts on mps.
    assert model.predict_calls == [1, 1]
    assert det.device == "mps"
    assert det._batch_unsupported is True
    assert det.batch_stats.per_frame_fallback_calls == 1

    # Subsequent calls skip the doomed batched attempt entirely.
    model.predict_calls.clear()
    det.detect_batch([_frame(1), _frame(2)])
    assert model.predict_calls == [1, 1]


def test_detect_batch_empty_returns_empty(monkeypatch):
    det = _make_detector(monkeypatch, _BatchModel({}))
    assert det.detect_batch([]) == []


def test_detect_batch_meta_length_mismatch_raises(monkeypatch):
    det = _make_detector(monkeypatch, _BatchModel({}))
    with pytest.raises(ValueError):
        det.detect_batch([_frame(1)], [FrameMeta(), FrameMeta()])


def test_batch_stats_track_effective_batch(monkeypatch):
    per_frame = {1: ([], [], []), 2: ([], [], [])}
    det = _make_detector(monkeypatch, _BatchModel(per_frame))
    det.detect_batch([_frame(1), _frame(2)])
    assert det.batch_stats.calls == 1
    assert det.batch_stats.frames == 2
    assert det.batch_stats.batched_calls == 1
    assert det.batch_stats.max_effective_batch == 2
    assert det.batch_stats.mean_effective_batch == 2.0
