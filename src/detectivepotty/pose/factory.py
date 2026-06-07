"""Factory for constructing the configured pose estimator.

Centralizes the ``enabled``/``backend`` switch so callers ask for "the pose
estimator for this config" without importing backend modules or knowing which one
is active. Importing this module does not import ``deeplabcut``; the SuperAnimal
backend only imports it lazily on first inference.
"""

from __future__ import annotations

from detectivepotty.config import PoseConfig
from detectivepotty.pose.base import MockPoseEstimator, PoseEstimator
from detectivepotty.pose.superanimal import SuperAnimalPoseEstimator


def build_pose_estimator(config: PoseConfig) -> PoseEstimator | None:
    """Return the configured pose estimator, or ``None`` when pose is disabled."""

    if not config.enabled:
        return None
    if config.backend == "mock":
        return MockPoseEstimator()
    if config.backend == "superanimal":
        return SuperAnimalPoseEstimator(config)
    raise ValueError(f"Unknown pose backend: {config.backend!r}")
