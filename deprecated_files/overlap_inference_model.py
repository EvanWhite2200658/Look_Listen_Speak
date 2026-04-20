# src/turn_prediction/overlap_inference_model.py

from __future__ import annotations

from typing import Any

import torch

from src.turn_prediction.model import TransformerConfig, TurnShiftTransformer
from .overlap_features import OverlapFeatureConfig, gaze_window_to_overlap_sequence
from src.turn_prediction.schemas import GazeWindow, TurnPrediction


class OverlapTurnModel:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        checkpoint: dict[str, Any] = torch.load(model_path, map_location=device)

        if checkpoint.get("feature_space") != "overlap_v1":
            raise ValueError(
                f"Expected overlap_v1 checkpoint, got {checkpoint.get('feature_space')}"
            )

        model_config_dict = checkpoint.get("model_config")
        if model_config_dict is None:
            raise ValueError("Checkpoint is missing model_config")

        self.model_config = TransformerConfig(**model_config_dict)
        self.model = TurnShiftTransformer(self.model_config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.threshold = float(checkpoint.get("best_threshold", 0.2))
        self.device = device
        self.feature_config = OverlapFeatureConfig(include_deltas=True)

    @property
    def expected_window_size(self) -> int:
        return int(self.model_config.max_seq_len)

    @property
    def expected_input_dim(self) -> int:
        return int(self.model_config.input_dim)

    def predict(self, window: GazeWindow) -> TurnPrediction:
        if not window.samples:
            return TurnPrediction(timestamp_ns=0, probability=0.0, is_turn=False)

        seq = gaze_window_to_overlap_sequence(window, self.feature_config)

        if seq.shape[0] != self.expected_window_size:
            raise ValueError(
                f"Window size mismatch: expected {self.expected_window_size}, got {seq.shape[0]}"
            )
        if seq.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.expected_input_dim}, got {seq.shape[1]}"
            )

        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(x)).item())

        return TurnPrediction(
            timestamp_ns=window.samples[-1].timestamp_ns,
            probability=prob,
            is_turn=(prob >= self.threshold),
        )