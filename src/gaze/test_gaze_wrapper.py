# src/gaze/test_gaze_wrapper.py

from __future__ import annotations

import time

from wrapper import GazeTrackerModule


def main() -> None:
    calibration_tracker = GazeTrackerModule(max_buffer_size=200)

    try:
        print("Starting preview...")
        calibration_tracker.start_preview()

        print("Preview closed. Running calibration...")
        calibration_tracker.calibrate()
    finally:
        print("Stopping calibration tracker...")
        calibration_tracker.stop()

    sampling_tracker = GazeTrackerModule(max_buffer_size=200)

    try:
        print("Beginning sampling...")
        sampling_tracker.start_sampling()

        num_valid_samples = 0

        for i in range(100):
            sample = sampling_tracker.poll_latest_from_tracker()

            if sample is None:
                print(f"[{i}] No sample returned")
            else:
                num_valid_samples += 1
                print(
                    f"[{i}] t={sample.timestamp_s:.3f}s | "
                    f"raw={sample.raw_xy} | "
                    f"filtered={sample.filtered_xy} | "
                    f"features={len(sample.features)} | "
                    f"buffer={sampling_tracker.get_buffer_size()}"
                )
            time.sleep(0.05)

        window = sampling_tracker.get_recent_window(window_size=10)
        print(f"\nValid samples collected: {num_valid_samples}")
        print(f"Collected {len(window)} samples in recent window.")
        print(f"Final buffer size: {sampling_tracker.get_buffer_size()}")

    finally:
        print("Stopping tracker...")
        sampling_tracker.stop()

if __name__ == "__main__":
    main()