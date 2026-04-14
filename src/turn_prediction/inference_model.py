# src/turn_prediction/inference_model.py

from __future__ import annotations

from typing import Any

import torch

from .model import TurnShiftTransformer, TransformerConfig
from .schemas import GazeWindow, TurnPrediction
from live_feature_contract import build_live_feature_contract_report


class TrainedTurnModel:
    """
    Strict live wrapper for a saved turn prediction checkpoint.

    This class intentionally does NOT invent or approximate the feature
    mapping required by the saved checkpoint. It only loads the checkpoint,
    exposes its expectations, and refuses inference until the live pipeline
    can provide the same feature space used during training.
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
        return (
            "TrainedTurnModel("
            f"expected_input_dim={self.expected_input_dim}, "
            f"expected_window_size={self.expected_window_size}, "
            f"threshold={self.threshold}, "
            f"training_feature_set={self.training_feature_set}, "
            f"epoch={self.checkpoint_epoch}, "
            f"best_val_f1={self.checkpoint_best_val_f1}, "
            f"val_loss={self.checkpoint_val_loss}"
            ")"
        )

    def predict(self, window: GazeWindow) -> TurnPrediction:
        if not window.samples:
            return TurnPrediction(
                timestamp_ns=0,
                probability=0.0,
                is_turn=False,
            )

        latest_timestamp = window.samples[-1].timestamp_ns
        report = build_live_feature_contract_report()

        raise NotImplementedError(
            "This checkpoint was trained in dataset feature space, but the live "
            "pipeline does not yet provide the same per-frame features. "
            f"Checkpoint expects input_dim={self.expected_input_dim}, "
            f"window_size={self.expected_window_size}, "
            f"feature_set={self.training_feature_set}. "
            f"unsupported_required_features={report.unsupported_feature_names}."
        )