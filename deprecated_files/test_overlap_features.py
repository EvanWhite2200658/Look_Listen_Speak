# src/turn_prediction/test_overlap_features.py

from __future__ import annotations

import time

from src.gaze.service import GazeTrackingService
from deprecated_files.overlap_features import (
OverlapFeatureConfig,
gaze_window_to_overlap_sequence,
get_overlap_feature_dim,
)
from schemas import GazeWindow


def main() -> None:
    service = GazeTrackingService(max_buffer_size=200)
    config = OverlapFeatureConfig(include_deltas=True)

    try:
        print("Starting gaze service...")
        service.start(preview=False)

        time.sleep(2)

        samples = service.get_recent_window(30)
        window = GazeWindow(samples)

        seq = gaze_window_to_overlap_sequence(window, config)
        dim = get_overlap_feature_dim(config)

        print(f"Sequence shape: {seq.shape}")
        print(f"Expected dim: {dim}")

        if seq.size > 0:
            print("First vector:")
            print(seq[0])

    finally:
        print("Stopping gaze service...")
        service.stop()


if __name__ == "__main__":
    main()