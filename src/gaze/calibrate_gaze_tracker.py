# src/gaze/calibrate_gaze_tracker.py
from __future__ import annotations

from gazefollower import GazeFollower
from patched_svr_calibration import PatchedSVRCalibration

def main() -> None:
    tracker = GazeFollower(calibration=PatchedSVRCalibration())

    try:
        print("Starting calibration...")
        tracker.calibrate()

        saved = tracker.calibration.save_model()
        print(f"save_model() returned: {saved}")
        print(f"Calibration flag: {tracker.calibration.has_calibrated}")
        print(f"Save directory: {tracker.calibration.workplace_calibration_dir}")
    finally:
        try:
            tracker.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()