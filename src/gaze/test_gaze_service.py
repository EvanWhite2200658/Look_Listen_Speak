# src/gaze/test_gaze_service.py

from __future__ import annotations

import time

from service import GazeTrackingService


def main() -> None:
    service = GazeTrackingService(
        max_buffer_size=200,
        poll_interval_s=0.05,
        run_calibration=False,
    )

    try:
        print("Starting GazeTrackingService")
        service.start(preview=False)

        if not service.is_running:
            print("Service did not start. Calibration is required for live sampling.")
            return

        for _ in range(40):
            sample = service.get_latest_sample()
            if sample is not None:
                print(
                    f"t={sample.timestamp_s:.3f}s | "
                    f"raw={sample.raw_xy} | "
                    f"filtered={sample.filtered_xy} | "
                    f"features={len(sample.features)}"
                )
            else:
                print("no sample yet")
            time.sleep(0.25)

        window = service.get_recent_window(20)
        print(f"\nRecent window size: {len(window)}")

    finally:
        print("Closing GazeTrackingService")
        service.stop()

if __name__ == "__main__":
    main()