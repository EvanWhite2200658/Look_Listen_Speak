# src/turn_prediction/test_checkpoint_metadata.py

from __future__ import annotations

from pathlib import Path
import json
import torch


def main() -> None:
    model_path = (
        Path(__file__).resolve().parent
        / "artifacts"
        / "turn_prediction"
        / "best_model.pt"
    )

    print(f"Checkpoint path: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    print("\nTop-level checkpoint keys:")
    for key in checkpoint.keys():
        print(f"- {key}")

    print("\nModel config:")
    print(json.dumps(checkpoint.get("model_config", {}), indent=2))

    print("\nTraining config:")
    print(json.dumps(checkpoint.get("training_config", {}), indent=2))

    print("\nFeature Columns:")
    print(json.dumps(checkpoint.get("feature_columns", {}), indent=2))

    print("\nBest threshold:", checkpoint.get("best_threshold"))
    print("Best val F1:", checkpoint.get("best_val_f1"))
    print("Validation loss:", checkpoint.get("val_loss"))
    print("Epoch:", checkpoint.get("epoch"))


if __name__ == "__main__":
    main()