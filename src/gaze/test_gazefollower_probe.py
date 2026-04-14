# src/gez/test_gazefollower_probe.py

from __future__ import annotations

import time
from pprint import pprint

from gazefollower import GazeFollower

from patched_svr_calibration import PatchedSVRCalibration


def _safe_repr(value, max_len: int = 120) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        return f"<repr failed: {exc}>"

    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

def main() -> None:
    tracker = GazeFollower(calibration=PatchedSVRCalibration())

    try:
        print("Starting sampling...")
        tracker.start_sampling()

        time.sleep(2.0)

        for i in range(30):
            gaze_info = tracker.get_gaze_info()

            print(f"\n--- sample {i} ---")

            if gaze_info is None:
                print("gaze_info is None")
                time.sleep(0.1)
                continue

            print("type:", type(gaze_info))

            try:
                attr_names = sorted(
                    name for name in dir(gaze_info)
                    if not name.startswith("_")
                )
            except Exception as exc:
                print("Could not read dir(gaze_info):", exc)
                attr_names = []

            print("attribute names:")
            pprint(attr_names)

            print("\nselected values:")
            for name in attr_names:
                try:
                    value = getattr(gaze_info, name)
                except Exception as exc:
                    print(f"{name}: <getattr failed: {exc}>")
                    continue

                if callable(value):
                    continue

                print(f"{name}: {_safe_repr(value)}")

            break

    finally:
        print("\nStopping tracker...")
        try:
            tracker.stop_sampling()
        except Exception as exc:
            print("stop_sampling warning:", exc)

        try:
            tracker.release()
        except Exception as exc:
            print("release warning:", exc)


if __name__ == "__main__":
    main()