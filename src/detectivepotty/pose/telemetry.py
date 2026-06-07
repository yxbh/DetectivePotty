"""Thread-safe perf/health telemetry for the pose backend.

Pure Python (no numpy/deeplabcut) so it is trivially unit-testable and cheap to
update on the hot path. One :class:`PoseTelemetry` lives on a pose estimator and is
updated once per :meth:`~detectivepotty.pose.base.PoseEstimator.estimate` call plus
on the one-time model build. ``snapshot()`` returns an immutable
:class:`PoseTelemetrySnapshot` for logging, the web app, or later monitoring.

Why this exists: pose is the heaviest per-frame component and we deliberately chose
the more accurate (slower) ``hrnet_w32`` head, so we want measured latency
percentiles and outcome/health rates to catch regressions or model drift rather
than guessing. All counters are process-lifetime unless noted; latency percentiles
are over a bounded recent window.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import threading
import time
from typing import Any

# Terminal outcomes: exactly one is recorded per *completed* estimate() call, so
# their counts partition ``total_calls``. (An invalid-input ValueError is a caller
# contract violation raised before any record and is intentionally not counted.)
OUTCOME_SUCCESS = "success"
OUTCOME_SKIP_TINY_CROP = "skip_tiny_crop"
OUTCOME_BUILD_FAILED = "build_failed"
OUTCOME_INFER_ERROR = "infer_error"
OUTCOME_INFER_NONE = "infer_none"
OUTCOME_BAD_SHAPE = "bad_shape"
OUTCOME_NO_FINITE_KEYPOINTS = "no_finite_keypoints"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_SUCCESS,
    OUTCOME_SKIP_TINY_CROP,
    OUTCOME_BUILD_FAILED,
    OUTCOME_INFER_ERROR,
    OUTCOME_INFER_NONE,
    OUTCOME_BAD_SHAPE,
    OUTCOME_NO_FINITE_KEYPOINTS,
)

_DEFAULT_LATENCY_WINDOW = 2048


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 100]``) of a sorted list."""

    if not sorted_vals:
        raise ValueError("empty input")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


@dataclass(frozen=True, slots=True)
class PoseTelemetrySnapshot:
    """Immutable point-in-time view of pose perf/health telemetry."""

    total_calls: int
    outcomes: dict[str, int]
    # Latency of the actual model call (excludes lock wait and one-time build).
    latency_ms_count: int
    latency_ms_mean: float | None
    latency_ms_min: float | None
    latency_ms_max: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_p99: float | None
    latency_window: int
    # One-time model build (cold start), NOT a per-frame cost: the runner is built
    # lazily on the first call and reused for the life of the estimator.
    cold_start_ms: float | None
    build_failed: bool
    # Health over successful poses (model-quality / drift signal).
    mean_kpt_conf: float | None
    mean_frac_conf_ge: float | None
    mean_keypoints: float | None
    conf_threshold: float
    started_at: float
    last_updated_at: float | None
    last_success_at: float | None
    snapshot_at: float

    @property
    def success_rate(self) -> float | None:
        """Fraction of completed calls that produced a pose."""

        if self.total_calls == 0:
            return None
        return self.outcomes.get(OUTCOME_SUCCESS, 0) / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "outcomes": dict(self.outcomes),
            "success_rate": self.success_rate,
            "latency_ms": {
                "count": self.latency_ms_count,
                "mean": self.latency_ms_mean,
                "min": self.latency_ms_min,
                "max": self.latency_ms_max,
                "p50": self.latency_ms_p50,
                "p95": self.latency_ms_p95,
                "p99": self.latency_ms_p99,
                "window": self.latency_window,
            },
            "cold_start": {
                "cold_start_ms": self.cold_start_ms,
                "failed": self.build_failed,
            },
            "health": {
                "mean_kpt_conf": self.mean_kpt_conf,
                "mean_frac_conf_ge": self.mean_frac_conf_ge,
                "mean_keypoints": self.mean_keypoints,
                "conf_threshold": self.conf_threshold,
            },
            "started_at": self.started_at,
            "last_updated_at": self.last_updated_at,
            "last_success_at": self.last_success_at,
            "snapshot_at": self.snapshot_at,
        }


class PoseTelemetry:
    """Accumulates pose perf/health counters under a single leaf lock.

    ``record``/``record_cold_start``/``snapshot``/``reset`` all take the same lock and do
    no I/O or callbacks while holding it, so it is safe to call from multiple camera
    threads (and from inside the estimator's own build lock — this lock is always
    acquired last and never re-enters the estimator).
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        latency_window: int = _DEFAULT_LATENCY_WINDOW,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._latency_window = latency_window
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        now = time.time()
        self._total_calls = 0
        self._outcomes: dict[str, int] = {name: 0 for name in OUTCOMES}
        self._lat_recent: deque[float] = deque(maxlen=self._latency_window)
        self._lat_count = 0
        self._lat_sum = 0.0
        self._lat_min: float | None = None
        self._lat_max: float | None = None
        self._cold_start_ms: float | None = None
        self._build_failed = False
        self._conf_sum = 0.0
        self._frac_sum = 0.0
        self._kpt_sum = 0
        self._success_count = 0
        self._started_at = now
        self._last_updated_at: float | None = None
        self._last_success_at: float | None = None

    def record(
        self,
        outcome: str,
        *,
        latency_ms: float | None = None,
        mean_conf: float | None = None,
        frac_ge: float | None = None,
        n_keypoints: int | None = None,
    ) -> None:
        """Record one terminal estimate() outcome (and any associated metrics)."""

        with self._lock:
            self._total_calls += 1
            self._outcomes[outcome] = self._outcomes.get(outcome, 0) + 1
            if latency_ms is not None:
                self._lat_recent.append(latency_ms)
                self._lat_count += 1
                self._lat_sum += latency_ms
                self._lat_min = (
                    latency_ms if self._lat_min is None else min(self._lat_min, latency_ms)
                )
                self._lat_max = (
                    latency_ms if self._lat_max is None else max(self._lat_max, latency_ms)
                )
            if outcome == OUTCOME_SUCCESS:
                self._success_count += 1
                if mean_conf is not None:
                    self._conf_sum += mean_conf
                if frac_ge is not None:
                    self._frac_sum += frac_ge
                if n_keypoints is not None:
                    self._kpt_sum += n_keypoints
                self._last_success_at = time.time()
            self._last_updated_at = time.time()

    def record_cold_start(self, latency_ms: float, *, ok: bool) -> None:
        """Record the one-time lazy model build (cold-start cost / failure).

        Called once, on the first inference, from inside the estimator's build
        guard -- this is NOT a per-frame metric.
        """

        with self._lock:
            self._cold_start_ms = latency_ms
            self._build_failed = not ok
            self._last_updated_at = time.time()

    def snapshot(self) -> PoseTelemetrySnapshot:
        with self._lock:
            ordered = sorted(self._lat_recent)
            mean = self._lat_sum / self._lat_count if self._lat_count else None
            successes = self._success_count
            return PoseTelemetrySnapshot(
                total_calls=self._total_calls,
                outcomes=dict(self._outcomes),
                latency_ms_count=self._lat_count,
                latency_ms_mean=mean,
                latency_ms_min=self._lat_min,
                latency_ms_max=self._lat_max,
                latency_ms_p50=_percentile(ordered, 50) if ordered else None,
                latency_ms_p95=_percentile(ordered, 95) if ordered else None,
                latency_ms_p99=_percentile(ordered, 99) if ordered else None,
                latency_window=self._latency_window,
                cold_start_ms=self._cold_start_ms,
                build_failed=self._build_failed,
                mean_kpt_conf=self._conf_sum / successes if successes else None,
                mean_frac_conf_ge=self._frac_sum / successes if successes else None,
                mean_keypoints=self._kpt_sum / successes if successes else None,
                conf_threshold=self._conf_threshold,
                started_at=self._started_at,
                last_updated_at=self._last_updated_at,
                last_success_at=self._last_success_at,
                snapshot_at=time.time(),
            )

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
