"""Offline tests for the CoreML SDPA export helpers.

Only the pure-torch parts are exercised: the SDPA rewrite must be numerically
identical to Ultralytics' original ``Attention.forward``, and the patch must
restore the original on exit. The actual CoreML export (heavy, macOS-only) is
never run here.
"""

from __future__ import annotations

import pytest
import torch

from detectivepotty.detect.coreml_export import patch_attention_sdpa


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
