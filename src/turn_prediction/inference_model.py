# src/turn_prediction/inference_model.py

from __future__ import annotations

from typing import Any

import torch

from .model import TurnShiftTransformer, TransformerConfig
from .runtime_feature_bridge import RuntimeBridgeConfig, gaze_window_to_strongest_runtime_sequence
from .schemas import GazeWindow, TurnPrediction


class TrainedTurnModel:
    """
    Runtime wrapper for the strongest dataset-trained checkpoint.

    The model remains trained in dataset feature space.
    At runtime we engineer the closest possible matching feature sequence
    from live gaze-wrapper signals and explicitly zero-fill unsupported fields.
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
        self.bridge_config = RuntimeBridgeConfig(include_context_placeholders=True)

        saved_threshold = checkpoint.get("best_threshold")
        self.threshold = float(
            saved_threshold
            if threshold is None and saved_threshold is not None
            else (threshold if threshold is not None else 0.20)
        )

        self.model = TurnShiftTransformer(self.model_config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_loss = checkpoint.get("val_loss")
        self.checkpoint_best_val_f1 = checkpoint.get("best_val_f1")

    @property
    def expected_input_dim(self) -> int:
        return int(self.model_config.input_dim)

    @property
    def expected_window_size(self) -> int:
        return int(self.model_config.max_seq_len)

    @property
    def training_feature_set(self) -> str | None:
        value = self.training_config.get("feature_set")
        return str(value) if value is not None else None

    def debug_summary(self) -> str:
        return(
            "TrainedTurnModel("
            f"expected_input_dim: {self.expected_input_dim}, "
            f"expected_window_size: {self.expected_window_size}, "
            f"threshold: {self.threshold}, "
            f"training_feature_set: {self.training_feature_set}, "
            f"epoch: {self.checkpoint_epoch}, "
            f"best_val_f1: {self.checkpoint_best_val_f1}, "
            f"val_loss: {self.checkpoint_val_loss}, "
            ")"
        )

    def predict(self, window: GazeWindow) -> TurnPrediction:
        if not window.samples:
            return TurnPrediction(timestamp_ns=0, probability=0.0, is_turn=False)

        latest_timestamp = window.samples[-1].timestamp_ns
        sequence = gaze_window_to_strongest_runtime_sequence(window, self.bridge_config)

        if sequence.shape[0] != self.expected_window_size:
            raise ValueError(
                "Window length does not match checkpoint expectation. "
                f"expected={self.expected_window_size}, actual={sequence.shape[0]}"
            )

        if sequence.shape[1] != self.expected_input_dim:
            raise ValueError(
                "Runtime feature dimension does not match checkpoint expectation. "
                f"expected={self.expected_input_dim}, actual={sequence.shape[1]}"
            )

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            print("logit:", float(logits.item()))
            probability = float(torch.sigmoid(logits).item())

        return TurnPrediction(
            timestamp_ns=latest_timestamp,
            probability=probability,
            is_turn=(probability >= self.threshold),
        )