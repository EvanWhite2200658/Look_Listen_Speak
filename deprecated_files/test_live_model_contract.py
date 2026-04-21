# src/turn_prediction/test_live_model_contract.py

from __future__ import annotations

from deprecated_files.live_features import get_live_feature_dim
from deprecated_files.live_model_config import (
build_default_live_feature_config,
build_live_transformer_config,
)


def main() -> None:
    feature_config = build_default_live_feature_config()
    transformer_config = build_live_transformer_config(window_size=30)

    live_feature_dim = get_live_feature_dim(feature_config)

    print("Live-native model contract")
    print(f"Live feature dim: {live_feature_dim}")
    print(f"Transformer input_dim: {transformer_config.input_dim}")
    print(f"Transformer max_seq_len: {transformer_config.max_seq_len}")
    print(f"Match: {live_feature_dim == transformer_config.input_dim}")


if __name__ == "__main__":
    main()