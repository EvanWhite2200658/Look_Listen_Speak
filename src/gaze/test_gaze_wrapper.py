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
                landmark_count = sample.face.landmark_count if sample.face is not None else 0
                face_rect = sample.face.face_rect if sample.face is not None else None
                can_gaze_estimation = (
                    sample.face.can_gaze_estimation if sample.face is not None else None
                )

                print(
                    f"[{i}] t={sample.timestamp_s:.3f}s | "
                    f"raw={sample.raw_xy} | "
                    f"filtered={sample.filtered_xy} | "
                    f"features={len(sample.features)} | "
                    f"landmarks={landmark_count} | "
                    f"face_rect={face_rect} | "
                    f"can_gaze={can_gaze_estimation} | "
                    f"buffer={sampling_tracker.get_buffer_size()}"
                )
            time.sleep(0.05)

        window = sampling_tracker.get_recent_window(window_size=10)
        print(f"\nValid samples collected: {num_valid_samples}")
        print(f"Collected {len(window)} samples in recent window.")
        print(f"Final buffer size: {sampling_tracker.get_buffer_size()}")

        if window:
            latest = window[-1]
            print("\nLatest sample summary:")
            print(f"timestamp_ns={latest.timestamp_ns}")
            print(f"event={latest.event}")
            print(f"tracking_state={latest.tracking_state}")
            print(f"status={latest.status}")
            print(f"face_present={latest.face is not None}")
            if latest.face is not None:
                print(f"landmark_count={latest.face.landmark_count}")
                print(f"left_rect={latest.face.left_rect}")
                print(f"right_rect={latest.face.right_rect}")

    finally:
        print("Stopping tracker...")
        sampling_tracker.stop()

if __name__ == "__main__":
    main()