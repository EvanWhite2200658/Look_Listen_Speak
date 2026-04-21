# src/turn_prediction/test_live_features.py

from __future__ import annotations

import time

from src.gaze.service import GazeTrackingService
from deprecated_files.live_features import (
LiveFeatureConfig,
gaze_window_to_live_sequence,
get_live_feature_dim,
get_live_feature_metadata,
)
from schemas import GazeWindow


def main() -> None:
    gaze_service = GazeTrackingService(max_buffer_size=200)

    config = LiveFeatureConfig(
        include_raw_xy=True,
        include_calibrated_xy=True,
        include_filtered_xy=True,
        include_eye_openness=True,
        include_face_geometry=True,
        include_eye_geometry=True,
        include_tracking_metadata=True,
        include_deltas=True,
    )

    metadata = get_live_feature_metadata(config)

    print("Configured live-native feature set")
    print(f"Base feature count: {len(metadata.base_feature_names)}")
    print(f"Total feature count: {len(metadata.feature_names)}")
    print("Base features:")
    for name in metadata.base_feature_names:
        print(f"- {name}")

    try:
        print("Starting gaze service...")
        gaze_service.start(preview=False)

        time.sleep(2.0)

        samples = gaze_service.get_recent_window(30)
        window = GazeWindow(samples=samples)

        print(f"Window length: {window.length}")
        sequence = gaze_window_to_live_sequence(window, config)
        feature_dim = get_live_feature_dim(config)

        print(f"Sequence shape: {sequence.shape}")
        print(f"Configured feature dim: {feature_dim}")

        if sequence.size > 0:
            print("First timestep vector:")
            print(sequence[0])

            if sequence.shape[0] > 1:
                print("Second timestep vector:")
                print(sequence[1])

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()

if __name__ == "__main__":
    main()