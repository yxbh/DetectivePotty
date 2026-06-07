from __future__ import annotations

import logging

import numpy as np
import pytest

from detectivepotty.config import PoseConfig
from detectivepotty.geometry import BBox
from detectivepotty.pose.keypoints import QUADRUPED_KEYPOINTS, QUADRUPED_SCHEMA
from detectivepotty.pose.superanimal import (
    SuperAnimalPoseEstimator,
    normalize_pose_array,
    resolve_bodypart_order,
    resolve_pose_device,
)
from detectivepotty.pose.telemetry import (
    OUTCOME_BAD_SHAPE,
    OUTCOME_BUILD_FAILED,
    OUTCOME_INFER_ERROR,
    OUTCOME_SKIP_TINY_CROP,
    OUTCOME_SUCCESS,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
N_KPTS = len(QUADRUPED_KEYPOINTS)


class FakeInfer:
    """Records calls and returns a deterministic ``(n_kpts, 3)`` pose array."""

    def __init__(self, array: np.ndarray | None = None, raises: bool = False) -> None:
        self.array = array
        self.raises = raises
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, rgb_crop: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
        self.calls.append((rgb_crop, target_bbox))
        if self.raises:
            raise RuntimeError("boom")
        if self.array is not None:
            return self.array
        rows = np.zeros((N_KPTS, 3), dtype=float)
        for i in range(N_KPTS):
            rows[i] = (float(i), float(2 * i), 0.9)
        return rows


def _config(**overrides: object) -> PoseConfig:
    base: dict[str, object] = {"enabled": True, "backend": "superanimal", "device": "cpu"}
    base.update(overrides)
    return PoseConfig(**base)


def test_estimate_maps_keypoints_to_original_frame_coords() -> None:
    # margin 0.4 on a 100x100 box centered in the frame -> no clipping, integer box.
    bbox = BBox(200, 150, 300, 250)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=infer)

    pose = est.estimate(FRAME, bbox, frame_idx=7, mono_ts=1.5, source_id="cam:a")

    assert pose is not None
    # expanded box = (160, 110, 340, 290); crop origin = (160, 110).
    nose = pose.points["nose"]  # row 0 -> (0, 0)
    assert nose.x == pytest.approx(160.0)
    assert nose.y == pytest.approx(110.0)
    back_middle = pose.points["back_middle"]  # index 21 -> (21, 42)
    assert back_middle.x == pytest.approx(160.0 + 21)
    assert back_middle.y == pytest.approx(110.0 + 42)


def test_estimate_records_expanded_crop_and_provenance() -> None:
    bbox = BBox(200, 150, 300, 250)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=infer)

    pose = est.estimate(FRAME, bbox, frame_idx=2, source_id="cam:b")

    assert pose is not None
    assert pose.backend == "superanimal"
    assert pose.model_name == "hrnet_w32"
    assert pose.device == "cpu"
    assert pose.image_size == (640, 480)
    assert pose.crop_margin_frac == pytest.approx(0.4)
    assert pose.keypoint_schema == QUADRUPED_SCHEMA
    assert pose.latency_ms is not None and pose.latency_ms >= 0.0
    assert pose.source_id == "cam:b"
    # crop_bbox is the integer box actually sliced.
    assert (pose.crop_bbox.x1, pose.crop_bbox.y1) == (160.0, 110.0)
    assert (pose.crop_bbox.x2, pose.crop_bbox.y2) == (340.0, 290.0)
    # The crop handed to the model matches the expanded region; bbox covers it.
    rgb_crop, target_bbox = infer.calls[0]
    assert rgb_crop.shape[:2] == (180, 180)
    assert target_bbox.tolist() == [[0.0, 0.0, 180.0, 180.0]]


def test_estimate_converts_bgr_to_rgb_for_model() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill the crop region with a distinct BGR triple (B=10, G=20, R=30).
    frame[:, :] = (10, 20, 30)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.2), infer_fn=infer)

    est.estimate(frame, BBox(60, 60, 140, 140))

    rgb_crop, _ = infer.calls[0]
    # After BGR->RGB the first channel must be the R value (30), last the B (10).
    assert int(rgb_crop[0, 0, 0]) == 30
    assert int(rgb_crop[0, 0, 1]) == 20
    assert int(rgb_crop[0, 0, 2]) == 10


def test_estimate_clips_crop_to_frame_edge() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=infer)

    pose = est.estimate(frame, BBox(80, 80, 98, 98))

    assert pose is not None
    # Expanded box clips to the frame's bottom-right corner.
    assert (pose.crop_bbox.x2, pose.crop_bbox.y2) == (100.0, 100.0)


def test_estimate_uses_integer_origin_for_fractional_bbox() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=infer)

    pose = est.estimate(frame, BBox(50.3, 40.7, 90.3, 80.7))

    assert pose is not None
    # expanded floats (34.3, 24.7, 106.3, 96.7) -> int slice origin (34, 24).
    assert (pose.crop_bbox.x1, pose.crop_bbox.y1) == (34.0, 24.0)
    assert pose.points["nose"].x == pytest.approx(34.0)
    assert pose.points["nose"].y == pytest.approx(24.0)


def test_estimate_returns_none_for_tiny_crop() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    infer = FakeInfer()
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=infer)

    assert est.estimate(frame, BBox(10, 10, 12, 12)) is None
    assert infer.calls == []  # never bothered the model


def test_estimate_returns_none_when_inference_empty() -> None:
    infer = FakeInfer(array=np.zeros((0, N_KPTS, 3), dtype=float))
    est = SuperAnimalPoseEstimator(_config(), infer_fn=infer)
    assert est.estimate(FRAME, BBox(200, 150, 300, 250)) is None


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((N_KPTS - 1, 3), dtype=float),
        np.zeros((N_KPTS + 1, 3), dtype=float),
        np.zeros((N_KPTS, 2), dtype=float),
        np.zeros((0, N_KPTS, 3), dtype=float),
    ],
)
def test_estimate_returns_none_for_bad_shapes(bad: np.ndarray) -> None:
    est = SuperAnimalPoseEstimator(_config(), infer_fn=FakeInfer(array=bad))
    assert est.estimate(FRAME, BBox(200, 150, 300, 250)) is None


def test_estimate_drops_non_finite_keypoints() -> None:
    rows = np.zeros((N_KPTS, 3), dtype=float)
    for i in range(N_KPTS):
        rows[i] = (float(i), float(i), 0.9)
    rows[0] = (np.nan, 10.0, 0.9)  # nose: NaN x
    rows[1] = (10.0, np.inf, 0.9)  # upper_jaw: inf y
    rows[2] = (10.0, 10.0, np.nan)  # lower_jaw: NaN conf
    est = SuperAnimalPoseEstimator(_config(), infer_fn=FakeInfer(array=rows))

    pose = est.estimate(FRAME, BBox(200, 150, 300, 250))

    assert pose is not None
    assert "nose" not in pose.points
    assert "upper_jaw" not in pose.points
    assert "lower_jaw" not in pose.points
    assert "back_middle" in pose.points


def test_estimate_returns_none_when_all_keypoints_non_finite() -> None:
    rows = np.full((N_KPTS, 3), np.nan, dtype=float)
    est = SuperAnimalPoseEstimator(_config(), infer_fn=FakeInfer(array=rows))
    assert est.estimate(FRAME, BBox(200, 150, 300, 250)) is None


def test_inference_error_is_caught_and_does_not_poison(caplog) -> None:
    class FlakyInfer:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            rows = np.zeros((N_KPTS, 3), dtype=float)
            rows[:, 2] = 0.9
            return rows

    est = SuperAnimalPoseEstimator(_config(), infer_fn=FlakyInfer())
    with caplog.at_level(logging.WARNING):
        first = est.estimate(FRAME, BBox(200, 150, 300, 250))
    assert first is None  # caught, not raised
    second = est.estimate(FRAME, BBox(200, 150, 300, 250))
    assert second is not None  # future calls still work


def test_lazy_build_happens_once() -> None:
    rows = np.zeros((N_KPTS, 3), dtype=float)
    rows[:, 2] = 0.9
    infer = FakeInfer(array=rows)
    est = SuperAnimalPoseEstimator(_config())
    builds = {"n": 0}

    def fake_build() -> FakeInfer:
        builds["n"] += 1
        return infer

    est._build_dlc_infer_fn = fake_build  # type: ignore[method-assign]
    est.estimate(FRAME, BBox(200, 150, 300, 250))
    est.estimate(FRAME, BBox(200, 150, 300, 250))
    assert builds["n"] == 1
    assert len(infer.calls) == 2


def test_build_failure_disables_backend_without_retry(caplog) -> None:
    est = SuperAnimalPoseEstimator(_config())
    builds = {"n": 0}

    def failing_build() -> object:
        builds["n"] += 1
        raise ImportError("no deeplabcut")

    est._build_dlc_infer_fn = failing_build  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        assert est.estimate(FRAME, BBox(200, 150, 300, 250)) is None
    assert est.estimate(FRAME, BBox(200, 150, 300, 250)) is None
    assert builds["n"] == 1  # not retried every frame
    assert any("failed to initialize" in r.message for r in caplog.records)


def test_resolve_bodypart_order_exact_match() -> None:
    assert resolve_bodypart_order(list(QUADRUPED_KEYPOINTS)) == QUADRUPED_KEYPOINTS


def test_resolve_bodypart_order_same_set_reordered() -> None:
    reordered = (QUADRUPED_KEYPOINTS[-1],) + QUADRUPED_KEYPOINTS[:-1]
    assert resolve_bodypart_order(reordered) == reordered


@pytest.mark.parametrize(
    "bad",
    [
        QUADRUPED_KEYPOINTS[:-1],  # missing one
        QUADRUPED_KEYPOINTS + ("extra_part",),  # extra
        tuple(f"p{i}" for i in range(N_KPTS)),  # renamed
    ],
)
def test_resolve_bodypart_order_rejects_incompatible(bad: tuple[str, ...]) -> None:
    with pytest.raises(RuntimeError):
        resolve_bodypart_order(bad)


def test_normalize_pose_array_selects_first_individual() -> None:
    arr = np.arange(2 * N_KPTS * 3, dtype=float).reshape(2, N_KPTS, 3)
    out = normalize_pose_array(arr, N_KPTS)
    assert out is not None
    assert out.shape == (N_KPTS, 3)
    assert np.array_equal(out, arr[0])


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((0, N_KPTS, 3)),
        np.zeros((N_KPTS, 2)),
        np.zeros((N_KPTS + 1, 3)),
        "not-an-array-but-ragged",
    ],
)
def test_normalize_pose_array_rejects_bad(raw: object) -> None:
    assert normalize_pose_array(raw, N_KPTS) is None


def test_resolve_pose_device(monkeypatch) -> None:
    assert resolve_pose_device("cpu") == "cpu"

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_pose_device("auto") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_pose_device("auto") == "cpu"
    assert resolve_pose_device("mps") == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_pose_device("auto") == "cuda"
    assert resolve_pose_device("cuda") == "cuda"

    with pytest.raises(ValueError):
        resolve_pose_device("gpu")


# --- telemetry integration -------------------------------------------------


def test_telemetry_records_success_with_latency_and_health() -> None:
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=FakeInfer())
    est.estimate(FRAME, BBox(200, 150, 300, 250))

    snap = est.telemetry_snapshot()
    assert snap.total_calls == 1
    assert snap.outcomes[OUTCOME_SUCCESS] == 1
    assert snap.latency_ms_count == 1
    assert snap.latency_ms_mean is not None and snap.latency_ms_mean >= 0.0
    # FakeInfer returns conf 0.9 for all 39 keypoints.
    assert snap.mean_kpt_conf == pytest.approx(0.9)
    assert snap.mean_frac_conf_ge == pytest.approx(1.0)
    assert snap.mean_keypoints == pytest.approx(float(N_KPTS))
    # No build happened (infer_fn injected) so cold start is unset.
    assert snap.cold_start_ms is None


def test_telemetry_records_skip_tiny_crop_without_latency() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=FakeInfer())
    est.estimate(frame, BBox(10, 10, 12, 12))

    snap = est.telemetry_snapshot()
    assert snap.outcomes[OUTCOME_SKIP_TINY_CROP] == 1
    assert snap.latency_ms_count == 0


def test_telemetry_records_bad_shape_with_latency() -> None:
    bad = np.zeros((N_KPTS - 1, 3), dtype=float)
    est = SuperAnimalPoseEstimator(_config(), infer_fn=FakeInfer(array=bad))
    est.estimate(FRAME, BBox(200, 150, 300, 250))

    snap = est.telemetry_snapshot()
    assert snap.outcomes[OUTCOME_BAD_SHAPE] == 1
    assert snap.latency_ms_count == 1  # the model ran, just produced junk


def test_telemetry_records_infer_error_without_latency() -> None:
    est = SuperAnimalPoseEstimator(_config(), infer_fn=FakeInfer(raises=True))
    est.estimate(FRAME, BBox(200, 150, 300, 250))

    snap = est.telemetry_snapshot()
    assert snap.outcomes[OUTCOME_INFER_ERROR] == 1
    assert snap.latency_ms_count == 0


def test_telemetry_records_cold_start_on_lazy_build() -> None:
    rows = np.zeros((N_KPTS, 3), dtype=float)
    rows[:, 2] = 0.9
    est = SuperAnimalPoseEstimator(_config())
    est._build_dlc_infer_fn = lambda: FakeInfer(array=rows)  # type: ignore[method-assign]
    est.estimate(FRAME, BBox(200, 150, 300, 250))

    snap = est.telemetry_snapshot()
    assert snap.cold_start_ms is not None and snap.cold_start_ms >= 0.0
    assert snap.build_failed is False
    assert snap.outcomes[OUTCOME_SUCCESS] == 1


def test_telemetry_records_cold_start_failure_on_build_error() -> None:
    est = SuperAnimalPoseEstimator(_config())

    def boom() -> object:
        raise ImportError("no deeplabcut")

    est._build_dlc_infer_fn = boom  # type: ignore[method-assign]
    est.estimate(FRAME, BBox(200, 150, 300, 250))
    est.estimate(FRAME, BBox(200, 150, 300, 250))

    snap = est.telemetry_snapshot()
    assert snap.build_failed is True
    assert snap.cold_start_ms is not None
    assert snap.outcomes[OUTCOME_BUILD_FAILED] == 2  # both calls hit the disabled backend


def test_telemetry_outcomes_partition_total_calls() -> None:
    est = SuperAnimalPoseEstimator(_config(crop_margin_frac=0.4), infer_fn=FakeInfer())
    est.estimate(FRAME, BBox(200, 150, 300, 250))  # success
    est.estimate(np.zeros((40, 40, 3), dtype=np.uint8), BBox(10, 10, 12, 12))  # tiny

    snap = est.telemetry_snapshot()
    assert snap.total_calls == 2
    assert sum(snap.outcomes.values()) == snap.total_calls
