"""Offline tests for the CoreML SDPA export helpers.

Only the pure-torch parts are exercised: the SDPA rewrite must be numerically
identical to Ultralytics' original ``Attention.forward``, and the patch must
restore the original on exit. The actual CoreML export (heavy, macOS-only) is
never run here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from detectivepotty.detect.coreml_export import (
    _max_batch_from_spec,
    patch_attention_sdpa,
)


def _make_attention(dim: int = 256, num_heads: int = 8):
    from ultralytics.nn.modules.block import Attention

    torch.manual_seed(0)
    return Attention(dim, num_heads=num_heads).eval()


def test_sdpa_rewrite_matches_original() -> None:
    attn = _make_attention()
    x = torch.randn(2, 256, 16, 16)
    with torch.no_grad():
        ref = attn(x)
        with patch_attention_sdpa():
            patched = attn(x)
    assert patched.shape == ref.shape
    assert torch.allclose(patched, ref, atol=1e-4, rtol=1e-4)


def test_sdpa_rewrite_handles_non_square_and_single_head() -> None:
    # Non-square H!=W exercises the (B,heads,head_dim,N) -> (B,C,H,W) reshape;
    # num_heads=1 exercises the degenerate head split.
    attn = _make_attention(dim=64, num_heads=1)
    x = torch.randn(1, 64, 12, 20)
    with torch.no_grad():
        ref = attn(x)
        with patch_attention_sdpa():
            patched = attn(x)
    assert patched.shape == ref.shape
    assert torch.allclose(patched, ref, atol=1e-4, rtol=1e-4)


def test_patch_restores_original_forward() -> None:
    from ultralytics.nn.modules.block import Attention

    original = Attention.forward
    with patch_attention_sdpa():
        assert Attention.forward is not original
    assert Attention.forward is original


def test_patch_restores_original_even_on_error() -> None:
    from ultralytics.nn.modules.block import Attention

    original = Attention.forward
    with pytest.raises(RuntimeError):
        with patch_attention_sdpa():
            raise RuntimeError("boom")
    assert Attention.forward is original


# --- Batched (dynamic) export wiring ----------------------------------------


class _FakeExportModel:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.export_kwargs: dict = {}

    def export(self, **kwargs):
        self.export_kwargs = kwargs
        return "models/yolo11m.mlpackage"


def _capture_export_kwargs(monkeypatch: pytest.MonkeyPatch, **export_args) -> dict:
    from detectivepotty.detect import coreml_export

    captured: dict = {}

    class _FakeYOLO(_FakeExportModel):
        def export(self, **kwargs):
            captured.update(kwargs)
            return super().export(**kwargs)

    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLO)
    coreml_export.export_coreml("models/yolo11m.pt", **export_args)
    return captured


def test_export_batch_one_stays_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _capture_export_kwargs(monkeypatch, batch=1)
    # batch==1 → a fixed single-image package (no dynamic flag baked).
    assert "batch" not in kwargs
    assert "dynamic" not in kwargs


def test_export_batch_gt_one_is_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _capture_export_kwargs(monkeypatch, batch=16)
    # batch>1 → dynamic flexible-shape package accepting batches 1..16.
    assert kwargs["batch"] == 16
    assert kwargs["dynamic"] is True


# --- Reading the baked batch back out of a spec ------------------------------


class _FakeType:
    def __init__(self, kind: str, multi_array=None) -> None:
        self._kind = kind
        self.multiArrayType = multi_array

    def WhichOneof(self, _field: str) -> str:
        return self._kind


class _FakeMultiArray:
    def __init__(self, shape=(), shape_range=None) -> None:
        self.shape = list(shape)
        self._shape_range = shape_range

    def HasField(self, field: str) -> bool:
        return field == "shapeRange" and self._shape_range is not None

    @property
    def shapeRange(self):
        return self._shape_range


class _FakeRange:
    def __init__(self, *upper_bounds: int) -> None:
        self.sizeRanges = [SimpleNamespace(upperBound=b) for b in upper_bounds]


def _spec_with(input_type) -> SimpleNamespace:
    inp = SimpleNamespace(type=input_type)
    return SimpleNamespace(description=SimpleNamespace(input=[inp]))


def test_max_batch_image_input_is_one() -> None:
    spec = _spec_with(_FakeType("imageType"))
    assert _max_batch_from_spec(spec) == 1


def test_max_batch_reads_dynamic_upper_bound() -> None:
    ma = _FakeMultiArray(shape=(1, 3, 640, 640), shape_range=_FakeRange(16, 3, 1280, 1280))
    spec = _spec_with(_FakeType("multiArrayType", ma))
    assert _max_batch_from_spec(spec) == 16


def test_max_batch_falls_back_to_fixed_shape() -> None:
    ma = _FakeMultiArray(shape=(8, 3, 640, 640), shape_range=None)
    spec = _spec_with(_FakeType("multiArrayType", ma))
    assert _max_batch_from_spec(spec) == 8


def test_max_batch_no_inputs_is_one() -> None:
    spec = SimpleNamespace(description=SimpleNamespace(input=[]))
    assert _max_batch_from_spec(spec) == 1
