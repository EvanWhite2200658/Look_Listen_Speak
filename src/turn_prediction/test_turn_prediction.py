# src/turn_prediction/test_turn_prediction.py
# TODO: verify output
from __future__ import annotations

import time
from src.gaze.service import GazeTrackingService
from dummy_model import DummyTurnModel
from schemas import GazeWindow


def main() -> None:
    gaze_service = GazeTrackingService(max_buffer_size=200)
    model = DummyTurnModel()

    try:
        print("Starting gaze service...")
        gaze_service.start(preview=False)

        time.sleep(2) # let buffer fill

        for _ in range(20):
            window_samples = gaze_service.get_recent_window(30)
            window = GazeWindow(window_samples)

            prediction = model.predict(window)

            print(
                f"t={prediction.timestamp_ns / 1e9:.3f}s | "
                f"p_turn={prediction.probability:.3f}"
            )

            time.sleep(0.2)

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()

if __name__ == "__main__":
    main()