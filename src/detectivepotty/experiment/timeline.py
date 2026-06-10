"""Per-second timelines + recall/precision scoring for the harvest bake-off.

The retro-harvest experiment asks one question: *which seconds of a long recording
should we spend YOLO on?* Every candidate strategy (compressed-domain motion,
keyframe diff, time-of-day allowlist, ...) ultimately emits a set of **selected
seconds**; the exhaustive dense-YOLO pass emits the **ground-truth dog-seconds**.
Scoring a strategy is then a pure set comparison, captured here so it is trivially
unit-testable without cameras, models, or ffmpeg.

A "second" is an integer index ``0..duration_s-1``. We work at second granularity
(not frame) because the harvest cut is padded to ±seconds anyway and it keeps the
48h math cheap and legible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecondTimeline:
    """A set of selected whole-second indices over a clip of ``duration_s`` seconds."""

    seconds: frozenset[int]
    duration_s: int

    def __post_init__(self) -> None:
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")

    @classmethod
    def from_iterable(cls, seconds, duration_s: int) -> "SecondTimeline":
        clamped = frozenset(s for s in seconds if 0 <= s < duration_s)
        return cls(seconds=clamped, duration_s=duration_s)

    @classmethod
    def empty(cls, duration_s: int) -> "SecondTimeline":
        return cls(seconds=frozenset(), duration_s=duration_s)

    @classmethod
    def full(cls, duration_s: int) -> "SecondTimeline":
        return cls(seconds=frozenset(range(duration_s)), duration_s=duration_s)

    @property
    def count(self) -> int:
        return len(self.seconds)

    @property
    def selected_fraction(self) -> float:
        return self.count / self.duration_s if self.duration_s else 0.0

    def dilate(self, pad_s: int) -> "SecondTimeline":
        """Grow each selected second by ``pad_s`` on both sides (a recall guard).

        A motion blip a second before the dog is visible should still pull the
        surrounding seconds in, so a strategy can pad its hits before YOLO runs.
        """

        if pad_s <= 0:
            return self
        grown: set[int] = set()
        for s in self.seconds:
            lo = max(0, s - pad_s)
            hi = min(self.duration_s - 1, s + pad_s)
            grown.update(range(lo, hi + 1))
        return SecondTimeline(seconds=frozenset(grown), duration_s=self.duration_s)

    def union(self, other: "SecondTimeline") -> "SecondTimeline":
        return SecondTimeline(
            seconds=self.seconds | other.seconds,
            duration_s=max(self.duration_s, other.duration_s),
        )


@dataclass(frozen=True)
class StrategyScore:
    """How a strategy's selected seconds compare to ground-truth dog-seconds."""

    name: str
    duration_s: int
    true_dog_seconds: int
    selected_seconds: int
    hits: int  # selected ∩ dog
    recall: float  # hits / true_dog_seconds — fraction of real dog time kept
    precision: float  # hits / selected — fraction of selected time that has a dog
    selected_fraction: float  # selected / duration — how much of the clip we keep
    compute_saved: float  # 1 - selected_fraction — seconds YOLO is skipped on
    missed_dog_seconds: int  # dog seconds the strategy dropped (recall misses)

    def as_row(self) -> dict[str, object]:
        return {
            "strategy": self.name,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "selected_frac": round(self.selected_fraction, 4),
            "compute_saved": round(self.compute_saved, 4),
            "missed_dog_s": self.missed_dog_seconds,
            "selected_s": self.selected_seconds,
            "dog_s": self.true_dog_seconds,
        }


def _assemble_score(
    name: str,
    *,
    duration_s: int,
    true_dog_seconds: int,
    selected_seconds: int,
    hits: int,
) -> "StrategyScore":
    """Build a :class:`StrategyScore` from raw counts (shared ratio math)."""

    recall = hits / true_dog_seconds if true_dog_seconds else 1.0
    if selected_seconds:
        precision = hits / selected_seconds
    else:
        precision = 1.0 if true_dog_seconds == 0 else 0.0
    selected_fraction = selected_seconds / duration_s if duration_s else 0.0
    return StrategyScore(
        name=name,
        duration_s=duration_s,
        true_dog_seconds=true_dog_seconds,
        selected_seconds=selected_seconds,
        hits=hits,
        recall=recall,
        precision=precision,
        selected_fraction=selected_fraction,
        compute_saved=1.0 - selected_fraction,
        missed_dog_seconds=true_dog_seconds - hits,
    )


def score_strategy(
    name: str,
    selected: SecondTimeline,
    ground_truth: SecondTimeline,
) -> StrategyScore:
    """Score ``selected`` against the exhaustive ``ground_truth`` dog-seconds.

    ``recall`` is the headline metric: the fraction of true dog-seconds a strategy
    would still feed to YOLO (i.e. not silently drop). ``compute_saved`` is the
    payoff: the fraction of the clip the strategy lets us skip. The bake-off looks
    for the strategy with the best compute_saved at acceptable (near-1.0) recall.
    """

    duration = max(selected.duration_s, ground_truth.duration_s)
    dog = ground_truth.seconds
    sel = selected.seconds
    return _assemble_score(
        name,
        duration_s=duration,
        true_dog_seconds=len(dog),
        selected_seconds=len(sel),
        hits=len(dog & sel),
    )


def aggregate_scores(name: str, scores: list[StrategyScore]) -> StrategyScore:
    """Combine per-chunk scores for one strategy into a single window-wide score.

    A long window is acquired and scored as several chunk files (one ground-truth
    pass each); this sums their raw counts (duration, dog-seconds, selected, hits)
    and recomputes the ratios so the bake-off reports one number per strategy over
    the whole window — the count sums are the correct aggregation (ratios are not).
    """

    if not scores:
        raise ValueError("aggregate_scores requires at least one score")
    return _assemble_score(
        name,
        duration_s=sum(s.duration_s for s in scores),
        true_dog_seconds=sum(s.true_dog_seconds for s in scores),
        selected_seconds=sum(s.selected_seconds for s in scores),
        hits=sum(s.hits for s in scores),
    )


@dataclass
class BakeoffReport:
    """The scored result of running every strategy over one acquired window."""

    source: str
    duration_s: int
    ground_truth_dog_seconds: int
    scores: list[StrategyScore] = field(default_factory=list)

    def format_table(self) -> str:
        cols = [
            ("strategy", 22),
            ("recall", 8),
            ("precision", 10),
            ("selected_frac", 14),
            ("compute_saved", 14),
            ("missed_dog_s", 13),
        ]
        head = "".join(name.ljust(width) for name, width in cols)
        lines = [
            f"# bake-off: {self.source}",
            f"# duration={self.duration_s}s  ground-truth dog-seconds="
            f"{self.ground_truth_dog_seconds}",
            head,
            "-" * len(head),
        ]
        for s in self.scores:
            row = s.as_row()
            lines.append(
                "".join(
                    str(row[key]).ljust(width)
                    for key, width in (
                        ("strategy", 22),
                        ("recall", 8),
                        ("precision", 10),
                        ("selected_frac", 14),
                        ("compute_saved", 14),
                        ("missed_dog_s", 13),
                    )
                )
            )
        return "\n".join(lines)
