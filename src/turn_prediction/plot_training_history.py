# src/turn_prediction/plot_training_history.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts" / "turn_prediction_runtime_compatible"
HISTORY_PATH = ARTIFACT_DIR / "training_history.json"
OUTPUT_DIR = ARTIFACT_DIR / "plots"

def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_plot(x, ys: dict[str, list[float]], ylabel: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in ys.items():
        plt.plot(x, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()

def main() -> None:
    history = load_history(HISTORY_PATH)
    epochs = [entry["epoch"] for entry in history]

    save_plot(
        epochs,
        {
            "train_loss": [entry["train_loss"] for entry in history],
            "val_loss": [entry["val_loss"] for entry in history],
        },
        "Loss",
        "loss_curve_runtime_compatible.png",
    )

    save_plot(
        epochs,
        {
            "train_f1": [entry["train_f1"] for entry in history],
            "val_f1": [entry["val_f1"] for entry in history],
            "best_threshold_val_f1": [entry["best_threshold_f1"] for entry in history],
        },
        "F1",
        "f1_curve_runtime_compatible.png",
    )

    save_plot(
        epochs,
        {
            "train_acc": [entry["train_acc"] for entry in history],
            "val_acc": [entry["val_acc"] for entry in history],
        },
        "Accuracy",
        "accuracy_curve_runtime_compatible.png",
    )

    save_plot(
        epochs,
        {
            "selected_threshold": [entry["best_threshold"] for entry in history],
        },
        "Threshold",
        "selected_threshold_runtime_compatible.png",
    )

    save_plot(
        epochs,
        {
            "predicted_positives": [entry["best_threshold_predicted_positives"] for entry in history],
        },
        "Predicted Positives",
        "predicted_positives_runtime_compatible.png",
    )


if __name__ == "__main__":
    main()