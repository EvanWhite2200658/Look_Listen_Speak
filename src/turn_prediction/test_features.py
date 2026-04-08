# src/turn_prediction/test_features.py
# TODO: verify output
from __future__ import annotations

import time
from src.gaze.service import GazeTrackingService
from features import FeatureConfig, gaze_window_to_sequence, get_feature_dim
from schemas import GazeWindow


def main() -> None:
    gaze_service = GazeTrackingService(max_buffer_size=200)

    try:
        print("Starting gaze service...")
        gaze_service.start(preview=False)

        time.sleep(2.0)

        samples = gaze_service.get_recent_window(30)
        window = GazeWindow(samples=samples)

        config = FeatureConfig(
            include_raw_xy=True,
            include_calibrated_xy=True,
            include_filtered_xy=True,
            include_eye_openness=True,
            include_feature_vector=True,
            include_deltas=True,
            max_feature_vector_length=16,
        )

        sequence = gaze_window_to_sequence(window, config)
        feature_dim = get_feature_dim(window, config)

        print(f"Window length: {window.length}")
        print(f"Sequence length: {sequence.shape}")
        print(f"Feature dim: {feature_dim}")

        if sequence.size > 0:
            print("First timestep:")
            print(sequence[0])

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()


if __name__ == "__main__":
    main()