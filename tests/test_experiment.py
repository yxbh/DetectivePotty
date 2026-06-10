from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from detectivepotty.experiment import (
    FrameDetection,
    GroundTruth,
    SecondTimeline,
    aggregate_scores,
    detect_frames,
    find_chunk_videos,
    parse_packets,
    per_second_energy,
    run_bakeoff,
    run_bakeoff_dir,
    score_strategy,
    select_by_threshold,
)
from detectivepotty.experiment.timeline import BakeoffReport


# --------------------------------------------------------------------------- #
# SecondTimeline + scoring
# --------------------------------------------------------------------------- #


def test_second_timeline_clamps_and_fractions():
    tl = SecondTimeline.from_iterable([-1, 0, 5, 9, 10, 100], duration_s=10)
    assert tl.seconds == frozenset({0, 5, 9})
    assert tl.count == 3
    assert tl.selected_fraction == pytest.approx(0.3)


def test_full_and_empty_timelines():
    assert SecondTimeline.full(4).seconds == frozenset({0, 1, 2, 3})
    assert SecondTimeline.empty(4).seconds == frozenset()
    assert SecondTimeline.full(4).selected_fraction == 1.0
    assert SecondTimeline.empty(0).selected_fraction == 0.0


def test_dilate_grows_and_clamps():
    tl = SecondTimeline.from_iterable([5], duration_s=10)
    grown = tl.dilate(2)
    assert grown.seconds == frozenset({3, 4, 5, 6, 7})
    # clamp at edges
    edge = SecondTimeline.from_iterable([0, 9], duration_s=10).dilate(2)
    assert 0 in edge.seconds and 9 in edge.seconds
    assert max(edge.seconds) == 9 and min(edge.seconds) == 0


def test_dilate_zero_is_noop():
    tl = SecondTimeline.from_iterable([5], duration_s=10)
    assert tl.dilate(0) is tl


def test_union_takes_max_duration():
    a = SecondTimeline.from_iterable([1], 5)
    b = SecondTimeline.from_iterable([7], 10)
    u = a.union(b)
    assert u.seconds == frozenset({1, 7})
    assert u.duration_s == 10


def test_negative_duration_rejected():
    with pytest.raises(ValueError):
        SecondTimeline(seconds=frozenset(), duration_s=-1)


def test_score_strategy_perfect_recall_baseline():
    ground = SecondTimeline.from_iterable([2, 3, 4], duration_s=10)
    full = SecondTimeline.full(10)
    s = score_strategy("blind-scrub", full, ground)
    assert s.recall == 1.0
    assert s.missed_dog_seconds == 0
    assert s.selected_fraction == 1.0
    assert s.compute_saved == 0.0
    assert s.precision == pytest.approx(3 / 10)


def test_score_strategy_partial_recall_and_compute_saved():
    ground = SecondTimeline.from_iterable([2, 3, 4, 8], duration_s=10)
    selected = SecondTimeline.from_iterable([3, 4, 5], duration_s=10)
    s = score_strategy("motion", selected, ground)
    # hits = {3,4}; misses {2,8}
    assert s.hits == 2
    assert s.recall == pytest.approx(2 / 4)
    assert s.missed_dog_seconds == 2
    assert s.precision == pytest.approx(2 / 3)
    assert s.selected_fraction == pytest.approx(3 / 10)
    assert s.compute_saved == pytest.approx(7 / 10)


def test_score_strategy_no_dogs_precision_one():
    ground = SecondTimeline.empty(10)
    selected = SecondTimeline.empty(10)
    s = score_strategy("none", selected, ground)
    assert s.recall == 1.0
    assert s.precision == 1.0


def test_bakeoff_report_table_lists_strategies():
    report = BakeoffReport(source="clip.mp4", duration_s=10, ground_truth_dog_seconds=3)
    report.scores.append(
        score_strategy("blind-scrub", SecondTimeline.full(10),
                        SecondTimeline.from_iterable([2, 3, 4], 10))
    )
    table = report.format_table()
    assert "clip.mp4" in table
    assert "blind-scrub" in table
    assert "recall" in table


# --------------------------------------------------------------------------- #
# Compressed-domain motion
# --------------------------------------------------------------------------- #


def test_parse_packets_tolerates_strings_and_missing_pts():
    obj = {
        "packets": [
            {"pts_time": "0.0", "size": "1000", "flags": "K_"},
            {"dts_time": "1.0", "size": "200", "flags": "__"},  # no pts → uses dts
            {"size": "50", "flags": "__"},  # no time at all
            {"pts_time": "bad", "size": "bad", "flags": ""},  # garbage
        ]
    }
    packets = parse_packets(obj)
    assert len(packets) == 4
    assert packets[0].is_keyframe is True
    assert packets[0].size == 1000
    assert packets[1].pts_time == 1.0 and packets[1].is_keyframe is False
    assert packets[2].pts_time is None
    assert packets[3].pts_time is None and packets[3].size == 0


def test_per_second_energy_excludes_keyframes():
    packets = parse_packets(
        {
            "packets": [
                {"pts_time": "0.1", "size": "5000", "flags": "K_"},  # keyframe, excluded
                {"pts_time": "0.5", "size": "100", "flags": "__"},
                {"pts_time": "0.9", "size": "200", "flags": "__"},
                {"pts_time": "1.2", "size": "900", "flags": "__"},
            ]
        }
    )
    energy = per_second_energy(packets, duration_s=2)
    assert energy == [300.0, 900.0]


def test_per_second_energy_includes_keyframes_when_asked():
    packets = parse_packets(
        {"packets": [{"pts_time": "0.0", "size": "5000", "flags": "K_"}]}
    )
    energy = per_second_energy(packets, duration_s=1, exclude_keyframes=False)
    assert energy == [5000.0]


def test_select_by_threshold_relative_to_peak():
    energy = [10.0, 100.0, 30.0, 0.0]
    sel = select_by_threshold(energy, threshold_frac=0.25)
    # cutoff = 25; seconds >= 25 with e>0 → {1,2}
    assert sel.seconds == frozenset({1, 2})
    assert sel.duration_s == 4


def test_select_by_threshold_all_zero_selects_nothing():
    sel = select_by_threshold([0.0, 0.0, 0.0], threshold_frac=0.1)
    assert sel.seconds == frozenset()
    assert sel.duration_s == 3


def test_select_by_threshold_empty():
    sel = select_by_threshold([], threshold_frac=0.1)
    assert sel.count == 0


# --------------------------------------------------------------------------- #
# Ground truth (fake detector — no model/GPU/video)
# --------------------------------------------------------------------------- #


@dataclass
class _FakeDet:
    confidence: float


class _FakeDetector:
    """Returns a dog for frame indices in ``dog_frames`` (assigned in submit order)."""

    def __init__(self, dog_frames: set[int]):
        self.dog_frames = dog_frames
        self._submitted = 0
        self.batch_sizes: list[int] = []

    def detect_batch(self, frames, metas=None):
        self.batch_sizes.append(len(frames))
        out = []
        for _ in frames:
            idx = self._submitted
            self._submitted += 1
            out.append([_FakeDet(0.8)] if idx in self.dog_frames else [])
        return out


def _frames(n: int):
    return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]


def test_detect_frames_batches_and_marks_dogs():
    det = _FakeDetector(dog_frames={1, 2, 7})
    gt = detect_frames(_frames(10), det, fps=2.0, batch_size=4)
    assert gt.frame_count == 10
    assert det.batch_sizes == [4, 4, 2]
    assert gt.dog_frame_count == 3
    flagged = {d.frame_idx for d in gt.detections if d.has_dog}
    assert flagged == {1, 2, 7}


def test_detect_frames_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        detect_frames(_frames(2), _FakeDetector(set()), fps=1.0, batch_size=0)


def test_ground_truth_dog_seconds_threshold():
    # fps=2 → frames 0,1 → sec 0; frames 2,3 → sec 1; frames 4,5 → sec 2
    gt = GroundTruth(
        fps=2.0,
        detections=[
            FrameDetection(0, True, 0.9),
            FrameDetection(1, True, 0.9),  # sec 0 has 2 dog frames
            FrameDetection(2, True, 0.9),  # sec 1 has 1 dog frame
            FrameDetection(3, False, 0.0),
            FrameDetection(4, False, 0.0),  # sec 2 has 0 dog frames
            FrameDetection(5, False, 0.0),
        ],
    )
    assert gt.duration_s == 3
    assert gt.dog_seconds(min_dog_frames=1).seconds == frozenset({0, 1})
    assert gt.dog_seconds(min_dog_frames=2).seconds == frozenset({0})


def test_ground_truth_zero_fps_is_empty():
    gt = GroundTruth(fps=0.0, detections=[FrameDetection(0, True, 0.9)])
    assert gt.duration_s == 0
    assert gt.dog_seconds().count == 0


# --------------------------------------------------------------------------- #
# run_bakeoff end-to-end with fakes (patch ffprobe so it stays offline)
# --------------------------------------------------------------------------- #


def test_run_bakeoff_end_to_end(monkeypatch):
    # 30 frames @ 3 fps = 10 seconds; dogs in seconds 2,3 (frames 6..11).
    dog_frames = set(range(6, 12))
    det = _FakeDetector(dog_frames=dog_frames)

    monkeypatch.setattr(
        "detectivepotty.experiment.bakeoff.iter_video_frames",
        lambda path, every_n=1: _frames(30),
    )
    monkeypatch.setattr(
        "detectivepotty.experiment.bakeoff.video_fps", lambda path: 3.0
    )
    # high motion exactly on the dog seconds 2,3
    fake_packets = parse_packets(
        {
            "packets": [
                {"pts_time": f"{s}.5", "size": "5000" if s in (2, 3) else "10", "flags": "__"}
                for s in range(10)
            ]
        }
    )
    monkeypatch.setattr(
        "detectivepotty.experiment.bakeoff.probe_packets", lambda path: fake_packets
    )

    report = run_bakeoff(
        "fake.mp4", det, source="fake.mp4", thresholds=(0.5,), pad_s=0, batch_size=8
    )
    by_name = {s.name: s for s in report.scores}
    assert by_name["blind-scrub"].recall == 1.0
    assert by_name["blind-scrub"].compute_saved == 0.0
    motion = by_name["motion@0.50"]
    # motion seconds {2,3} exactly match the dog seconds → recall 1.0, saves 8/10
    assert motion.recall == 1.0
    assert motion.selected_seconds == 2
    assert motion.compute_saved == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Multi-chunk aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_scores_sums_counts_and_recomputes_ratios():
    # chunk A: 10s, dog {2,3,4} (3), selected {3,4,5} (3), hits {3,4} (2)
    a = score_strategy(
        "motion",
        SecondTimeline.from_iterable([3, 4, 5], 10),
        SecondTimeline.from_iterable([2, 3, 4], 10),
    )
    # chunk B: 20s, dog {0,1} (2), selected {0,1,2,3} (4), hits {0,1} (2)
    b = score_strategy(
        "motion",
        SecondTimeline.from_iterable([0, 1, 2, 3], 20),
        SecondTimeline.from_iterable([0, 1], 20),
    )
    agg = aggregate_scores("motion", [a, b])
    assert agg.duration_s == 30
    assert agg.true_dog_seconds == 5
    assert agg.selected_seconds == 7
    assert agg.hits == 4
    assert agg.recall == pytest.approx(4 / 5)
    assert agg.precision == pytest.approx(4 / 7)
    assert agg.selected_fraction == pytest.approx(7 / 30)
    assert agg.compute_saved == pytest.approx(1 - 7 / 30)
    assert agg.missed_dog_seconds == 1


def test_aggregate_scores_requires_input():
    with pytest.raises(ValueError):
        aggregate_scores("x", [])


def test_find_chunk_videos_sorted_and_filtered(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("nope")
    (tmp_path / "sub").mkdir()
    found = find_chunk_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.mp4"]


def test_run_bakeoff_dir_aggregates_two_chunks(monkeypatch):
    # Two chunks, each 30 frames @ 3 fps = 10s. Dogs differ per chunk.
    chunk_dogs = {
        "c0.mp4": set(range(6, 12)),   # seconds 2,3
        "c1.mp4": set(range(0, 3)),    # second 0 (frames 0,1,2)
    }

    class _MultiDetector:
        """Independent submit counters keyed by which chunk is being built."""

        def __init__(self):
            self.current = None
            self._counters = {}

        def detect_batch(self, frames, metas=None):
            out = []
            for _ in frames:
                idx = self._counters.get(self.current, 0)
                self._counters[self.current] = idx + 1
                dogs = chunk_dogs[self.current]
                out.append([_FakeDet(0.8)] if idx in dogs else [])
            return out

    det = _MultiDetector()

    def fake_iter(path, every_n=1):
        det.current = str(path).split("/")[-1]
        det._counters[det.current] = 0
        return _frames(30)

    monkeypatch.setattr(
        "detectivepotty.experiment.bakeoff.iter_video_frames", fake_iter
    )
    monkeypatch.setattr("detectivepotty.experiment.bakeoff.video_fps", lambda p: 3.0)

    def fake_probe(path):
        name = str(path).split("/")[-1]
        hot = {"c0.mp4": (2, 3), "c1.mp4": (0,)}[name]
        return parse_packets(
            {
                "packets": [
                    {"pts_time": f"{s}.5", "size": "5000" if s in hot else "10",
                     "flags": "__"}
                    for s in range(10)
                ]
            }
        )

    monkeypatch.setattr(
        "detectivepotty.experiment.bakeoff.probe_packets", fake_probe
    )

    report = run_bakeoff_dir(
        ["/tmp/c0.mp4", "/tmp/c1.mp4"],
        det,
        source="window",
        thresholds=(0.5,),
        pad_s=0,
        batch_size=8,
    )
    assert report.duration_s == 20
    # c0 dog seconds {2,3}=2, c1 {0}=1 → 3 total
    assert report.ground_truth_dog_seconds == 3
    by_name = {s.name: s for s in report.scores}
    assert by_name["blind-scrub"].recall == 1.0
    assert by_name["blind-scrub"].selected_seconds == 20
    motion = by_name["motion@0.50"]
    # motion selects exactly the dog seconds in each chunk → recall 1.0, 3/20 kept
    assert motion.recall == 1.0
    assert motion.selected_seconds == 3
    assert motion.compute_saved == pytest.approx(1 - 3 / 20)


def test_run_bakeoff_dir_requires_videos():
    with pytest.raises(ValueError):
        run_bakeoff_dir([], _FakeDetector(set()))
