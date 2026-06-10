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
the crash is never emitted). The exported ``mlprogram`` then runs on the GPU,
numerically identical to the original block (max abs diff ~5e-4, float rounding).

Single-frame CoreML latency is roughly on par with the ``.pt`` MPS path once the
CPU letterbox/NMS at full camera resolution is included; the real throughput win
(~3×) comes from a **batched** export (``export_coreml(..., batch=N)``), which
lets :class:`~detectivepotty.detect.yolo.DogDetector` run a whole frame window in
one GPU forward instead of one frame at a time.

This is opt-in and tuner-only; the live pipeline still runs ``.pt`` on MPS via
:class:`detectivepotty.detect.yolo.DogDetector`. See ``plan.md`` (ROUND 7c/8) for
the full investigation.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Any, Iterator

import torch
import torch.nn.functional as F

__all__ = [
    "patch_attention_sdpa",
    "export_coreml",
    "coreml_max_batch",
    "DEFAULT_COREML_DIR",
]

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
    batch: int = 1,
) -> Path:
    """Export ``weights`` (a YOLO ``.pt``) to a GPU-safe CoreML ``.mlpackage``.

    Applies :func:`patch_attention_sdpa` for the duration of the export so the
    emitted ``mlprogram`` runs on the GPU. Returns the path to the resulting
    ``.mlpackage`` directory.

    ``out_path`` redirects/renames the export (e.g. into
    :data:`DEFAULT_COREML_DIR` for the committable set); when omitted, Ultralytics
    writes ``<weights stem>.mlpackage`` next to the weights. ``imgsz`` is baked
    into the model's input shape, so the tuner must run inference at the same long
    edge. Heavy and macOS-only — never invoked by the offline test suite (callers
    inject a fake).

    ``batch`` controls whether the package can run **batched** inference:

    * ``batch == 1`` (default) → a fixed single-image package (back-compat). It
      can only ever process one frame per ``predict`` call, so the detector falls
      back to serial per-frame inference and GPU utilisation stays low.
    * ``batch > 1`` → a **dynamic** flexible-shape package that accepts any batch
      in ``[1, batch]``. This is GPU-safe (verified on Apple MPS with the SDPA
      attention rewrite) and unlocks ~3× detection throughput by letting
      :class:`~detectivepotty.detect.yolo.DogDetector` submit whole frame windows
      in one GPU forward. Export ``batch`` to match the largest batch the caller
      submits (the tuner caps batches at ``tune_detection_batch_size``); larger
      batches than baked simply fall back to per-frame on the same accelerator.
    """

    from ultralytics import YOLO

    weights = Path(weights)
    model = YOLO(str(weights))
    export_kwargs: dict[str, Any] = dict(
        format="coreml",
        imgsz=imgsz,
        half=half,
        nms=False,
        verbose=False,
    )
    # CoreML rejects a fixed ``batch > 1`` ("not supported without 'dynamic=True'");
    # ``dynamic=True`` makes the input shape flexible so it accepts batches 1..N.
    if batch > 1:
        export_kwargs["batch"] = batch
        export_kwargs["dynamic"] = True
    with patch_attention_sdpa():
        exported = Path(model.export(**export_kwargs))

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


def _max_batch_from_spec(spec: Any) -> int:
    """Pull the max batch an exported CoreML ``spec`` accepts (1 if not batched).

    A fixed single-image export uses an ``imageType`` input (no batch dimension →
    1). A :func:`export_coreml` ``batch > 1`` export uses a ``multiArrayType``
    whose leading dimension is flexible; its upper bound is the largest batch the
    package can run. Pure (takes an already-loaded proto) so it is unit-testable
    without ``coremltools`` or a real ``.mlpackage``.
    """

    inputs = getattr(spec.description, "input", None)
    if not inputs:
        return 1
    inp = inputs[0]
    if inp.type.WhichOneof("Type") != "multiArrayType":
        return 1  # imageType (fixed single image) or other → no batching
    ma = inp.type.multiArrayType
    if ma.HasField("shapeRange") and len(ma.shapeRange.sizeRanges) > 0:
        return int(ma.shapeRange.sizeRanges[0].upperBound)
    if len(ma.shape) > 0:
        return int(ma.shape[0])
    return 1


_BATCH_CACHE: dict[tuple[str, int], int] = {}


def coreml_max_batch(path: str | Path) -> int:
    """Return the largest batch the ``.mlpackage`` at ``path`` accepts (``1`` if
    fixed single-image, unreadable, or ``coremltools`` is unavailable).

    Reads the model *spec* only (a cheap protobuf parse — no compile/load), and
    memoises the result per ``(path, mtime)`` so the model picker can label each
    CoreML option with its baked batch size without repeated I/O.
    """

    p = Path(path)
    try:
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        return 1
    cached = _BATCH_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import coremltools as ct

        value = _max_batch_from_spec(ct.utils.load_spec(str(p)))
    except Exception:  # noqa: BLE001 - any read failure → treat as un-batched
        value = 1
    _BATCH_CACHE[key] = value
    return value
