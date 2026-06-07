"""Compute-device resolution shared by the detector and the pose backend.

One helper keeps device policy identical everywhere. ``auto`` prefers CUDA (the
semi-permanent Windows + NVIDIA runtime), then MPS (Apple Silicon dev), then CPU.
An explicitly requested accelerator that is unavailable falls back to CPU with a
warning so a misconfiguration is visible rather than silently slow.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch always present in this project.
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover - torch always present in this project.
        return False


def resolve_device(requested: str) -> str:
    """Resolve a requested device string to a concrete backend name.

    ``auto`` resolves CUDA -> MPS -> CPU. An explicit ``cuda``/``mps`` that is
    unavailable warns once and returns ``cpu``. ``cpu`` is always honoured.
    """

    if requested not in VALID_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(VALID_DEVICES)}")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if _cuda_available():
            return "cuda"
        logger.warning("CUDA device requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "mps":
        if _mps_available():
            return "mps"
        logger.warning("MPS device requested but unavailable; falling back to CPU.")
        return "cpu"
    # auto
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"
