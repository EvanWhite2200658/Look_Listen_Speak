# src/turn_prediction/live_model_config.py

from __future__ import annotations

from dataclasses import asdict

from live_features import LiveFeatureConfig, get_live_feature_dim
from model import TransformerConfig

DEFAULT_LIVE_WINDOW_SIZE = 30
DEFAULT_LIVE_THRESHOLD = 0.20


def build_default_live_feature_config() -> LiveFeatureConfig:
    """
    default live-native feature schema used for retraining and runtime.
    keep this central so training and inference cannot silently diverge.
    :return:
    """
    return LiveFeatureConfig(
        include_raw_xy=True,
        include_calibrated_xy=True,
        include_filtered_xy=True,
        include_eye_openness=True,
        include_face_geometry=True,
        include_eye_geometry=True,
        include_tracking_metadata=True,
        include_deltas=True,
    )


def build_live_transformer_config(
        window_size: int = DEFAULT_LIVE_WINDOW_SIZE,
) -> TransformerConfig:
    feature_config = build_default_live_feature_config()
    input_dim = get_live_feature_dim(feature_config)

    return TransformerConfig(
        input_dim=input_dim,
        max_seq_len=window_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_learned_positional_encoding=True,
    )

def serialize_live_feature_config(config: LiveFeatureConfig) -> dict:
    return asdict(config)