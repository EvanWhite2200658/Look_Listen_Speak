# src/runtime/test_demo_monitor.py
import math
import time

import cv2

from src.runtime.demo_monitor import DemoMonitor, DemoMonitorState


def main() -> None:
    cap = cv2.VideoCapture(0)
    monitor = DemoMonitor()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                frame = None

            t = time.time()
            confidence = 0.5 + 0.35 * math.sin(t)

            adjusted_wait = int(700 * (1 - confidence))
            adjusted_wait = max(200, adjusted_wait)

            state = DemoMonitorState(
                confidence=confidence,
                baseline_wait_ms=700,
                adjusted_wait_ms=adjusted_wait,
                speech_detected=False,
                response_allowed=confidence > 0.5,
                system_speaking=False,
                latest_delay_ms=adjusted_wait,
                last_event=(
                    f"DEMO | confidence={confidence:.2f} | "
                    f"adjusted_wait={adjusted_wait}ms | "
                    f"response_allowed={confidence > 0.5}"
                ),
            )

            gaze_point = None
            if frame is not None:
                h, w = frame.shape[:2]
                gaze_point = (w // 2, h // 2)

            monitor.draw(
                frame=frame,
                state=state,
                gaze_point=gaze_point,
                face_box=None,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        monitor.close()


if __name__ == "__main__":
    main()