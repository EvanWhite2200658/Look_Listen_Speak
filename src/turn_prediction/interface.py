# src/turn_prediction/interface.py

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import GazeWindow, TurnPrediction


class TurnPredictionModel(ABC):
    """
    Abstract interface for gaze-based turn-transition prediction models.
    """

    @abstractmethod
    def predict(self, window: GazeWindow) -> TurnPrediction:
        """
        predict probability of a turn transition from a gaze window
        :param window:
        :return:
        """
        pass