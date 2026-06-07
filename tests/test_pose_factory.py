from __future__ import annotations

import sys

import pytest

from detectivepotty.config import PoseConfig
from detectivepotty.pose.base import MockPoseEstimator
from detectivepotty.pose.factory import build_pose_estimator
from detectivepotty.pose.superanimal import SuperAnimalPoseEstimator


def test_disabled_returns_none() -> None:
    assert build_pose_estimator(PoseConfig(enabled=False)) is None


def test_mock_backend_routing() -> None:
    est = build_pose_estimator(PoseConfig(enabled=True, backend="mock"))
    assert isinstance(est, MockPoseEstimator)


def test_superanimal_backend_routing() -> None:
    est = build_pose_estimator(
        PoseConfig(enabled=True, backend="superanimal", device="cpu")
    )
    assert isinstance(est, SuperAnimalPoseEstimator)


def test_building_superanimal_does_not_import_deeplabcut() -> None:
    build_pose_estimator(PoseConfig(enabled=True, backend="superanimal", device="cpu"))
    assert "deeplabcut" not in sys.modules


def test_unknown_backend_raises() -> None:
    config = PoseConfig(enabled=True, backend="mock")
    config.backend = "bogus"  # bypass literal validation to hit the guard
    with pytest.raises(ValueError):
        build_pose_estimator(config)
