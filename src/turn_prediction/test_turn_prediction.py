# src/turn_prediction/test_turn_prediction.py

from __future__ import annotations

import time
from pathlib import Path

from src.gaze.service import GazeTrackingService
from inference_model import TrainedTurnModel
from schemas import GazeWindow


def main() -> None:
    model_path = (
        Path(__file__).resolve().parent
        / "artifacts"
        / "turn_prediction"
        / "best_model.pt"
    )

    print(f"Using model path: {model_path}")

    gaze_service = GazeTrackingService(max_buffer_size=200)
    model = TrainedTurnModel(model_path=str(model_path), device="cpu")

    print("Loaded model:")
    print(model.debug_summary())

    try:
        print("Starting gaze service...")
        gaze_service.start(preview=False)

        time.sleep(2.0)

        window_samples = gaze_service.get_recent_window(model.expected_window_size)
        print(f"Raw window sample count: {len(window_samples)}")

        window = GazeWindow(samples=window_samples)

        if not window.samples:
            print("No samples available yet.")
            return

        print("First sample:", window.samples[0])

        prediction = model.predict(window)

        print(
            f"t={prediction.timestamp_ns / 1e9:.3f}s | "
            f"p_turn={prediction.probability:.3f} | "
            f"is_turn={prediction.is_turn}"
        )

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()


if __name__ == "__main__":
    main()