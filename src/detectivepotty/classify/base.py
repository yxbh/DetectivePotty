"""Classifier interface for weak v0 and future trained models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from detectivepotty.events import ClassifierGuess, Track

if TYPE_CHECKING:
    from detectivepotty.sources.base import Frame


@dataclass(slots=True)
class ClassifierResult:
    guess: ClassifierGuess
    confidence: float
    needs_label: bool = True


class PottyClassifier(ABC):
    """Classifies a tracked potty candidate.

    The v0 classifier is only a weak heuristic. Its output is a guess for review,
    not ground truth; saved events should keep ``needs_label=True`` until labeled.
    """

    @abstractmethod
    def classify(
        self,
        track: Track,
        frames: Sequence[Frame],
    ) -> ClassifierResult:
        raise NotImplementedError
