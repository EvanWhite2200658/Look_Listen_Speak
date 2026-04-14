# src/turn_prediction/schemas.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from src.gaze.schemas import GazeSample


@dataclass
class GazeWindow:
    """
    Temporal window of gaze samples used as model input.
    """
    samples: List[GazeSample]

    @property
    def length(self) -> int:
        return len(self.samples)


@dataclass
class TurnPrediction:
    """
    Output of the turn-transition model.
    """
    timestamp_ns: int
    probability: float # probability of turn transition (0-1)
    is_turn: bool # thresholded decision for live system