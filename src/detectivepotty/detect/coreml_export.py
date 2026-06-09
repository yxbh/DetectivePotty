"""Export an Ultralytics YOLO11 model to a GPU-safe CoreML ``mlprogram``.

Apple's (closed-source) MetalPerformanceShadersGraph GPU compiler aborts with
``MLIR pass manager failed`` when it tries to lower YOLO11's ``C2PSA`` attention
block — specifically the hand-written 4D batched matmul + softmax in
:meth:`ultralytics.nn.modules.block.Attention.forward`. The crash happens inside
Apple's framework at predict/load time, so no coremltools conversion flag can
avoid it; only the *emitted ops* can be changed.

The fix is to rewrite that block with :func:`torch.nn.functional.scaled_dot_product_attention`.
coremltools *decomposes* SDPA into a GPU-lowerable matmul ordering (at the
default deployment target — Ultralytics deliberately omits
``minimum_deployment_target``, so the iOS18 native-SDPA op that would re-trigger
the crash is never emitted). The exported ``mlprogram`` then runs on the GPU at
roughly half the latency of the ``.pt`` MPS path, numerically identical to the
original block (max abs diff ~5e-4, float rounding).

This is opt-in and tuner-only; the live pipeline still runs ``.pt`` on MPS via
:class:`detectivepotty.detect.yolo.DogDetector`. See ``plan.md`` (ROUND 7c/8) for
the full investigation.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator

import torch
import torch.nn.functional as F

__all__ = ["patch_attention_sdpa", "export_coreml", "DEFAULT_COREML_DIR"]

# Curated, committable exports live here (tracked via a .gitignore negation);
# ad-hoc exports default next to the weights and stay ignored.
DEFAULT_COREML_DIR = Path("models") / "coreml"


def _sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """GPU-safe drop-in for ``Attention.forward`` using SDPA.

    Mathematically identical to Ultralytics' original block. The only subtlety is
    the positional-encoding term ``self.pe``: it is applied to the *original*
    value tensor (reshaped to ``B,C,H,W``), NOT to the attention output — applying
    it to the output is a silent correctness bug (max abs diff jumps to ~128).
    """

    b, c, h, w = x.shape
    n = h * w
    qkv = self.qkv(x)
    q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )
    # q, k: (B, heads, key_dim, N); v: (B, heads, head_dim, N).
    # SDPA wants (B, heads, seq, feat); its default scale is 1/sqrt(key_dim),
    # which equals the original ``self.scale``, so pass no explicit scale.
    out = F.scaled_dot_product_attention(
        q.transpose(-2, -1),
        k.transpose(-2, -1),
        v.transpose(-2, -1),
    )  # (B, heads, N, head_dim)
    out = out.transpose(-2, -1).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(out)


@contextmanager
def patch_attention_sdpa() -> Iterator[None]:
    """Temporarily swap ``Attention.forward`` for the SDPA rewrite.

    Patches the class method, so every ``Attention`` instance in the model (and
    any copies Ultralytics makes during export) uses it. Only YOLO11's ``C2PSA``
    instantiates ``block.Attention``; for architectures that don't (e.g. YOLO26)
    this is a no-op, so the context manager is always safe to enter. The original
    method is restored on exit even if export raises.
    """

    from ultralytics.nn.modules.block import Attention

    original = Attention.forward
    Attention.forward = _sdpa_attention_forward
    try:
        yield
    finally:
        Attention.forward = original


def export_coreml(
    weights: str | Path,
    out_path: str | Path | None = None,
    imgsz: int = 640,
    half: bool = True,
) -> Path:
    """Export ``weights`` (a YOLO ``.pt``) to a GPU-safe CoreML ``.mlpackage``.

    Applies :func:`patch_attention_sdpa` for the duration of the export so the
    emitted ``mlprogram`` runs on the GPU. Returns the path to the resulting
    ``.mlpackage`` directory.

    ``out_path`` redirects/renames the export (e.g. into
    :data:`DEFAULT_COREML_DIR` for the committable set); when omitted, Ultralytics
    writes ``<weights stem>.mlpackage`` next to the weights. ``imgsz`` is baked
    into the model's fixed input shape, so the tuner must run inference at the
    same long edge. Heavy and macOS-only — never invoked by the offline test
    suite (callers inject a fake).
    """

    from ultralytics import YOLO

    weights = Path(weights)
    model = YOLO(str(weights))
    with patch_attention_sdpa():
        exported = Path(
            model.export(
                format="coreml",
                imgsz=imgsz,
                half=half,
                nms=False,
                verbose=False,
            )
        )

    if out_path is None:
        return exported

    out_path = Path(out_path)
    if out_path.resolve() == exported.resolve():
        return exported
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    shutil.move(str(exported), str(out_path))
    return out_path
