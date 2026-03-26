# src/gaze/test_gaze_wrapper.py

from __future__ import annotations

import time

from wrapper import GazeTrackerModule


def main() -> None:
    tracker = GazeTrackerModule(max_buffer_size=200)

    try:
        print("Starting preview...")
        tracker.start_preview()

        input("Press Enter after preview check to continue to calibration...")

        print("Running calibration...")
        tracker.calibrate()

        for _ in range(100):
            sample = tracker.poll_latest_from_tracker()
            if sample is not None:
                print(
                    f"t={sample.timestamp_s:.3f}s | "
                    f"raw={sample.raw_xy} | "
                    f"filtered={sample.filtered_xy} | "
                    f"features={len(sample.features)}"
                )
            time.sleep(0.05)

        window = tracker.get_recent_window(window_size=10)
        print(f"\nCollected {len(window)} samples in recent window.")

    finally:
        print("Stopping tracker...")
        tracker.stop()

if __name__ == "__main__":
    main()