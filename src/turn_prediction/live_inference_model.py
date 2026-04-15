# src/turn_prediction/live_inference_model.py

from __future__ import annotations

from typing import Any

import torch

from .live_features import gaze_window_to_live_sequence
from .live_model_config import build_default_live_feature_config
from .model import TransformerConfig, TurnShiftTransformer
from .schemas import GazeWindow, TurnPrediction


class LiveTurnModel:
    """
    Inference wrapper for checkpoints trained on the live-native feature schema.
    """

    def __init__(
            self,
            model_path: str,
            threshold: float | None = None,
            device: str = "cpu",
    ) -> None:
        self.device = device
        checkpoint: dict[str, Any] = torch.load(model_path, map_location=device)

        self.model_config = TransformerConfig(**checkpoint["model_config"])
        self.training_config = checkpoint.get("training_config", {})
        live_feature_config_dict = checkpoint.get("live_feature_config")

        if live_feature_config_dict is None:
            raise ValueError(
                "Checkpoint does not contain live_feature_config. "
                "This loader is only for live-native checkpoints."
            )

        self.feature_config = build_default_live_feature_config().__class__(**live_feature_config_dict)

        saved_threshold = checkpoint.get("best_threshold")
        self.threshold = float(
            saved_threshold
            if threshold is None and saved_threshold is not None
            else (threshold if threshold is not None else 0.20)
        )

        self.model = TurnShiftTransformer(self.model_config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @property
    def expected_input_dim(self) -> int:
        return int(self.model_config.input_dim)

    @property
    def expected_window_size(self) -> int:
        return int(self.model_config.max_seq_len)

    def debug_summary(self) -> str:
        return (
            "LiveTurnModel("
            f"expected_input_dim={self.expected_input_dim}, "
            f"expected_window_size={self.expected_window_size}, "
            f"threshold={self.threshold}"
            ")"
        )

    def predict(self, window: GazeWindow) -> TurnPrediction:
        if not window.samples:
            return TurnPrediction(timestamp_ns=0, probability=0.0, is_turn=False)

        latest_timestamp = window.samples[-1].timestamp_ns
        sequence = gaze_window_to_live_sequence(window, self.feature_config)

        if sequence.shape[0] != self.expected_window_size:
            raise ValueError(
                "Window length does not match checkpoint expectation. "
                f"expected={self.expected_window_size}, got={sequence.shape[0]}"
            )
        if sequence.shape[1] != self.expected_input_dim:
            raise ValueError(
                "Feature dimension does not match checkpoint expectation. "
                f"expected={self.expected_input_dim}, got={sequence.shape[1]}."
            )

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probability = float(torch.sigmoid(logits).item())

        is_turn = probability >= self.threshold
        return TurnPrediction(
            timestamp_ns=latest_timestamp,
            probability=probability,
            is_turn=is_turn,
        )