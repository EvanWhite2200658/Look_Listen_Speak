# src/turn_prediction/dummy_model.py

from __future__ import annotations

import random

from src.turn_prediction.interface import TurnPredictionModel
from src.turn_prediction.schemas import GazeWindow, TurnPrediction


class DummyTurnModel(TurnPredictionModel):
    """
    Temporary model for testing pipeline integration.
    Returns random probabilities.
    """

    def predict(self, window: GazeWindow) -> TurnPrediction:
        if not window.samples:
            return TurnPrediction(timestamp_ns=0, probability=0.0)

        latest = window.samples[-1]

        return TurnPrediction(
            timestamp_ns=latest.timestamp_ns,
            probability=random.random()
        )