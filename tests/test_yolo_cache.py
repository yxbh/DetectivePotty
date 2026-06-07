"""Tests for DogDetector's shared model cache and CPU-fallback swap."""

from __future__ import annotations

import numpy as np
import pytest

import detectivepotty.detect.yolo as yolo_mod
from detectivepotty.detect.yolo import DogDetector, clear_model_cache


@pytest.fixture(autouse=True)
def _isolate_cache():
    clear_model_cache()
    yield
    clear_model_cache()


class _FakeModel:
    """Stand-in YOLO model that only succeeds on ``ok_device``."""

    def __init__(self, ok_device: str | None) -> None:
        self.ok_device = ok_device
        self.calls: list[str] = []

    def predict(self, frame, imgsz, conf, device, verbose):  # noqa: ANN001
        self.calls.append(device)
        if device != self.ok_device:
            raise RuntimeError("boom")
        return []


def _patch_loader(monkeypatch, models):
    iterator = iter(models)

    def fake_load(self, candidates):  # noqa: ANN001
        self.model_name = candidates[0]
        return next(iterator)

    monkeypatch.setattr(yolo_mod, "resolve_device", lambda device: device)
    monkeypatch.setattr(DogDetector, "_load_model", fake_load)


def test_shared_model_reused_per_key(monkeypatch):
    loads: list[object] = []

    def fake_load(self, candidates):  # noqa: ANN001
        self.model_name = candidates[0]
        obj = object()
        loads.append(obj)
        return obj

    monkeypatch.setattr(yolo_mod, "resolve_device", lambda device: device)
    monkeypatch.setattr(DogDetector, "_load_model", fake_load)

    d1 = DogDetector(model_name="m.pt", device="cpu")
    d2 = DogDetector(model_name="m.pt", device="cpu")
    assert d1.model is d2.model
    assert len(loads) == 1

    # A different device must not share the cached instance.
    d3 = DogDetector(model_name="m.pt", device="mps")
    assert d3.model is not d1.model
    assert len(loads) == 2

    # Clearing the cache forces a fresh load on next construction.
    clear_model_cache()
    d4 = DogDetector(model_name="m.pt", device="cpu")
    assert d4.model is not d1.model
    assert len(loads) == 3


def test_unshared_model_always_fresh(monkeypatch):
    loads: list[object] = []

    def fake_load(self, candidates):  # noqa: ANN001
        self.model_name = candidates[0]
        obj = object()
        loads.append(obj)
        return obj

    monkeypatch.setattr(yolo_mod, "resolve_device", lambda device: device)
    monkeypatch.setattr(DogDetector, "_load_model", fake_load)

    a = DogDetector(model_name="m.pt", device="cpu", use_shared_model=False)
    b = DogDetector(model_name="m.pt", device="cpu", use_shared_model=False)
    assert a.model is not b.model
    assert len(loads) == 2


def test_cpu_fallback_swaps_to_cpu_keyed_model(monkeypatch):
    accelerator_model = _FakeModel(ok_device=None)  # fails on every device
    cpu_model = _FakeModel(ok_device="cpu")
    _patch_loader(monkeypatch, [accelerator_model, cpu_model])

    det = DogDetector(model_name="m.pt", long_edge=64, device="mps")
    assert det.model is accelerator_model

    out = det.detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert out == []
    assert det.device == "cpu"
    assert det.model is cpu_model
    assert accelerator_model.calls == ["mps"]
    assert cpu_model.calls == ["cpu"]
