# src/turn_prediction/test_overlap_inference.py

from __future__ import annotations

import time
from pathlib import Path

from src.gaze.service import GazeTrackingService
from overlap_inference_model import OverlapTurnModel
from schemas import GazeWindow



def main() -> None:
    model_path = (
        Path(__file__).resolve().parent
        / "artifacts"
        / "overlap_model"
        / "best_model.pt"
    )

    service = GazeTrackingService(max_buffer_size=200)
    model = OverlapTurnModel(str(model_path))

    try:
        print("Starting gaze service...")
        service.start(preview=False)
        time.sleep(2.0)

        for _ in range(10):
            samples = service.get_recent_window(model.expected_window_size)
            print(f"Window sample count: {len(samples)}")

            if len(samples) != model.expected_window_size:
                print("Waiting for full window...")
                time.sleep(0.2)
                continue

            window = GazeWindow(samples=samples)
            pred = model.predict(window)
            print(
                f"t={pred.timestamp_ns / 1e9:.3f}s | "
                f"p_turn={pred.probability:.3f} | "
                f"is_turn={pred.is_turn}"
            )
            time.sleep(0.2)

    finally:
        print("Stopping service...")
        service.stop()


if __name__ == "__main__":
    main()