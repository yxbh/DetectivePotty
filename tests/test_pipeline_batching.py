"""Equivalence tests for batched detection in the file and live loops.

These guard the two batch-inference restructures:

* the file ``process_file_camera`` segment-replay loop, where ``file_detection_
  batch_size`` must not change recorded events (detections are per-image-
  independent, so batched == per-frame), and
* the ``_run_live_detection_loop`` accumulate/flush path, where batching defers
  ``state_machine.process`` but must keep frame->detection mapping, ordering, and
  the trailing flush intact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from detectivepotty.config import (
    CameraConfig,
    CameraInputConfig,
    Config,
    GlobalSettings,
)
from detectivepotty.events import Detection
from detectivepotty.geometry import BBox
from detectivepotty.pipeline import run_pipeline
from detectivepotty.sources.base import Frame, VideoSource

from test_pipeline import write_synthetic_video


# --------------------------------------------------------------------------- #
# File-loop equivalence
# --------------------------------------------------------------------------- #


def _file_config(
    dataset_dir: Path,
    video_path: Path,
    *,
    file_batch: int,
) -> Config:
    return Config(
        global_settings=GlobalSettings(
            dataset_dir=dataset_dir,
            model_name="fake.pt",
            device="cpu",
            file_detection_batch_size=file_batch,
        ),
        cameras=[
            CameraConfig(
                id="cam-1",
                name="Backyard",
                input=CameraInputConfig(kind="file", path=video_path),
                detection_conf_threshold=0.25,
                event_duration_s=1.0,
                stationary_threshold_s=1.0,
                dwell_trigger_s=2.0,
                sample_rate_fps=1.0,
                pre_roll_s=1.0,
                post_roll_s=1.0,
                retention_days=30,
            ),
        ],
    )


def _dets_for(frame_idx: int, mono_ts: float, wall_ts) -> list[Detection]:  # noqa: ANN001
    """A dog held still for frames 0..5, then gone — yields exactly one event."""

    if frame_idx <= 5:
        return [
            Detection(
                bbox=BBox(40, 20, 90, 100),
                confidence=0.9,
                class_name="dog",
                frame_idx=frame_idx,
                mono_ts=mono_ts,
                wall_ts=wall_ts,
            ),
        ]
    return []


class _RecordingBatchDetector:
    """Deterministic detector exposing both ``detect`` and ``detect_batch``.

    Records the size of every ``detect_batch`` call so a test can assert that a
    real multi-frame batch actually formed (vs silently degrading to batch-1).
    """

    device = "cpu"
    model_name = "fake.pt"
    last_inference = None

    def __init__(self) -> None:
        self.batch_calls: list[int] = []

    def detect(self, frame_bgr, *, frame_idx, mono_ts, wall_ts):  # noqa: ANN001
        del frame_bgr
        return _dets_for(frame_idx, mono_ts, wall_ts)

    def detect_batch(self, frames, metas):  # noqa: ANN001
        self.batch_calls.append(len(frames))
        return [_dets_for(m.frame_idx, m.mono_ts, m.wall_ts) for m in metas]


def _run_file(tmp_path: Path, *, file_batch: int) -> tuple[list[Path], _RecordingBatchDetector]:
    video_path = tmp_path / "sample.mp4"
    dataset_dir = tmp_path / "dataset"
    write_synthetic_video(video_path, frames=16)
    config = _file_config(dataset_dir, video_path, file_batch=file_batch)

    holder: list[_RecordingBatchDetector] = []

    def factory(_camera_config):  # noqa: ANN001
        detector = _RecordingBatchDetector()
        holder.append(detector)
        return detector

    event_dirs = run_pipeline(config, detector_factory=factory)
    return event_dirs, holder[0]


def _event_fingerprint(event_dirs: list[Path]) -> list[dict]:
    """Reduce recorded events to the fields a batch change could perturb."""

    fingerprint: list[dict] = []
    for event_dir in sorted(event_dirs):
        metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
        fingerprint.append(
            {
                "frames": sorted(r["frame_idx"] for r in metadata["frame_records"]),
                "crops": sorted(c["frame_idx"] for c in metadata["crop_boxes"]),
                "guess": metadata["classifier_guess"],
                "trigger": metadata["trigger_reason"],
            },
        )
    return fingerprint


def test_file_batch_matches_single_frame_events(tmp_path) -> None:
    single_dirs, single_detector = _run_file(tmp_path / "single", file_batch=1)
    batch_dirs, batch_detector = _run_file(tmp_path / "batched", file_batch=8)

    # The restructure must not change what gets recorded.
    assert len(single_dirs) == 1
    assert _event_fingerprint(single_dirs) == _event_fingerprint(batch_dirs)

    # ...and the batched run must actually have batched (segment of >1 sampled
    # frame), while the batch=1 run only ever submits single frames.
    assert max(batch_detector.batch_calls) > 1
    assert set(single_detector.batch_calls) == {1}


# --------------------------------------------------------------------------- #
# Live-loop equivalence
# --------------------------------------------------------------------------- #


def _live_dets_for(frame_idx: int, mono_ts: float, wall_ts) -> list[Detection]:  # noqa: ANN001
    """Varied per-frame output (some empty) to exercise mapping integrity."""

    if frame_idx % 3 == 2:
        return []
    return [
        Detection(
            bbox=BBox(10 + frame_idx, 20, 60 + frame_idx, 90),
            confidence=0.8,
            class_name="dog",
            frame_idx=frame_idx,
            mono_ts=mono_ts,
            wall_ts=wall_ts,
        ),
    ]


class _KeyedDetector:
    device = "cpu"
    model_name = "fake.pt"
    last_inference = None

    def detect(self, frame_bgr, *, frame_idx, mono_ts, wall_ts):  # noqa: ANN001
        del frame_bgr
        return _live_dets_for(frame_idx, mono_ts, wall_ts)

    def detect_batch(self, frames, metas):  # noqa: ANN001
        del frames
        return [_live_dets_for(m.frame_idx, m.mono_ts, m.wall_ts) for m in metas]


class _PacedLiveSource(VideoSource):
    """Synthetic live source that paces reads so the consumer keeps up."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._idx = 0

    def open(self):
        return self

    def read(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        frame = Frame(
            bgr=image,
            frame_idx=self._idx,
            mono_ts=float(self._idx),
            wall_ts=datetime.now(timezone.utc),
            source_id="pool-rtsp",
        )
        self._idx += 1
        time.sleep(0.02)
        return frame

    def close(self) -> None:
        return None

    @property
    def fps(self):
        return 10.0

    @property
    def resolution(self):
        return (160, 120)

    @property
    def is_live(self):
        return True


class _RecordingStateMachine:
    """Captures every ``process`` call without emitting events (no timing flake)."""

    pose_gate = None

    def __init__(self, _camera_config) -> None:  # noqa: ANN001
        self.calls: list[tuple[int, tuple]] = []

    def process(self, frame, detections, trigger_reason=None):  # noqa: ANN001
        del trigger_reason
        shaped = tuple(
            (d.frame_idx, d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, round(d.confidence, 4))
            for d in detections
        )
        self.calls.append((frame.frame_idx, shaped))
        return []

    def flush(self):
        return []


def _live_config(dataset_dir: Path, *, live_batch: int, max_wait_s: float) -> Config:
    return Config(
        global_settings=GlobalSettings(
            dataset_dir=dataset_dir,
            model_name="fake.pt",
            device="cpu",
            live_detection_batch_size=live_batch,
            max_batch_wait_s=max_wait_s,
        ),
        cameras=[
            CameraConfig(
                id="cam-pool",
                name="Pool",
                input=CameraInputConfig(kind="rtsp", url_env="POOL_RTSP_URL"),
                sample_rate_fps=1.0,
            ),
        ],
    )


def _run_live(
    tmp_path: Path,
    monkeypatch,
    *,
    live_batch: int,
    max_wait_s: float,
) -> _RecordingStateMachine:
    monkeypatch.setenv("POOL_RTSP_URL", "rtsp://user:pass@10.0.0.1:554/cam")
    config = _live_config(tmp_path, live_batch=live_batch, max_wait_s=max_wait_s)
    holder: list[_RecordingStateMachine] = []

    def state_factory(camera_config):  # noqa: ANN001
        sm = _RecordingStateMachine(camera_config)
        holder.append(sm)
        return sm

    run_pipeline(
        config,
        detector_factory=lambda _c: _KeyedDetector(),
        state_machine_factory=state_factory,
        rtsp_source_factory=_PacedLiveSource,
        max_live_frames=8,
    )
    return holder[0]


def _assert_live_invariants(sm: _RecordingStateMachine) -> None:
    assert sm.calls, "expected at least one processed frame"
    indices = [frame_idx for frame_idx, _ in sm.calls]
    # Strictly increasing: ordering is preserved and no frame processed twice.
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    # Mapping integrity: each frame carries exactly the detections the detector
    # produces for that frame_idx — batching must not misalign frames/results.
    for frame_idx, shaped in sm.calls:
        expected = _live_dets_for(frame_idx, 0.0, None)
        expected_shaped = tuple(
            (d.frame_idx, d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, round(d.confidence, 4))
            for d in expected
        )
        assert shaped == expected_shaped


def test_live_batch_one_processes_each_sampled_frame(tmp_path, monkeypatch) -> None:
    sm = _run_live(tmp_path / "b1", monkeypatch, live_batch=1, max_wait_s=0.5)
    _assert_live_invariants(sm)


def test_live_batch_many_preserves_frame_detection_mapping(tmp_path, monkeypatch) -> None:
    # Large wait so flushing is driven by batch size, not the timeout.
    sm = _run_live(tmp_path / "b4", monkeypatch, live_batch=4, max_wait_s=1000.0)
    _assert_live_invariants(sm)
    # The trailing flush must drain the final partial batch: the last sampled
    # frame the loop saw has to have reached the state machine.
    assert sm.calls
