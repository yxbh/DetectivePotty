"""Tests for shared compute-device resolution."""

from __future__ import annotations

import pytest

from detectivepotty.device import resolve_device


@pytest.fixture
def patch_torch(monkeypatch):
    import torch

    def _set(*, cuda: bool, mps: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return _set


def test_cpu_always_cpu(patch_torch) -> None:
    patch_torch(cuda=True, mps=True)
    assert resolve_device("cpu") == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu(patch_torch) -> None:
    patch_torch(cuda=True, mps=True)
    assert resolve_device("auto") == "cuda"
    patch_torch(cuda=False, mps=True)
    assert resolve_device("auto") == "mps"
    patch_torch(cuda=False, mps=False)
    assert resolve_device("auto") == "cpu"


def test_explicit_cuda(patch_torch) -> None:
    patch_torch(cuda=True, mps=False)
    assert resolve_device("cuda") == "cuda"
    patch_torch(cuda=False, mps=True)
    # Explicit CUDA without a GPU falls back to CPU (not MPS).
    assert resolve_device("cuda") == "cpu"


def test_explicit_mps(patch_torch) -> None:
    patch_torch(cuda=True, mps=True)
    assert resolve_device("mps") == "mps"
    patch_torch(cuda=True, mps=False)
    # Explicit MPS without Apple acceleration falls back to CPU (not CUDA).
    assert resolve_device("mps") == "cpu"


def test_invalid_device() -> None:
    with pytest.raises(ValueError):
        resolve_device("gpu")
