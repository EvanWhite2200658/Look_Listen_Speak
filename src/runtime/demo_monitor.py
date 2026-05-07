# src/runtime/demo_monitor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DemoMonitorState:
    confidence: float = 0.0
    baseline_wait_ms: int = 700
    adjusted_wait_ms: int = 700
    speech_detected: bool = False
    response_allowed: bool = False
    system_speaking: bool = False
    last_event: str = "Waiting for runtime events..."
    latest_delay_ms: Optional[int] = None


class DemoMonitor:
    def __init__(
            self,
            window_name: str = "HRI Runtime Monitor",
            width: int = 1280,
            height: int = 720,
    ) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def draw(
            self,
            frame: Optional[np.ndarray],
            state: DemoMonitorState,
            gaze_point: Optional[Tuple[int, int]] = None,
            face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        self._title(canvas, "Gaze-Informed Turn-Taking Runtime Monitor")

        #layout
        feed_x, feed_y, feed_w, feed_h = 20, 55, 610, 470
        info_x, info_y, info_w, info_h = 650, 55, 610, 470
        log_x, log_y, log_w, log_h = 20, 545, 1240, 135

        self._panel(canvas, feed_x, feed_y, feed_w, feed_h, "Live Gaze Feed")
        self._panel(canvas, info_x, info_y, info_w, info_h, "Timing Decision")
        self._panel(canvas, log_x, log_y, log_w, log_h, "Latest Runtime Event")

        self._draw_feed(canvas, frame, feed_x, feed_y, feed_w, feed_h, gaze_point, face_box)
        self._draw_info(canvas, state, info_x, info_y)
        self._draw_log(canvas, state.last_event, log_x, log_y)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def _title(self, img: np.ndarray, text: str) -> None:
        cv2.putText(
            img,
            text,
            (20, 35),
            self.font,
            0.85,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    def _panel(self, img: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.putText(
            img,
            title,
            (x + 12, y + 28),
            self.font,
            0.65,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _draw_feed(
            self,
            img: np.ndarray,
            frame: Optional[np.ndarray],
            x: int,
            y: int,
            w: int,
            h: int,
            gaze_point: Optional[Tuple[int, int]],
            face_box: Optional[Tuple[int, int, int, int]],
    ) -> None:
        if frame is None:
            self._text(img, "No webcam frame available", x + 30, y + 245, 0.75)
            return

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        view = self._fit(frame, w - 24, h - 58)
        vh, vw = view.shape[:2]

        vx = x + 12
        vy = y + 42

        img[vy:vy + vh, vx:vx + vw] = view

        original_h, original_w = frame.shape[:2]
        sx = vw / original_w
        sy = vh / original_h

        if face_box is not None:
            bx, by, bw, bh = face_box
            p1 = (int(vx + bx * sx), int(vy + by * sy))
            p2 = (int(vx + (bx + bw) * sx), int(vy + (by + bh) * sy))
            cv2.rectangle(img, p1, p2, (80, 220, 80), 2)

        if gaze_point is not None:
            gx, gy = gaze_point
            px = int(vx + gx * sx)
            py = int(vy + gy * sy)

            cv2.circle(img, (px, py), 13, (80, 220, 80), 2)
            cv2.line(img, (px - 28, py), (px + 28, py), (80, 220, 80), 1)
            cv2.line(img, (px, py - 28), (px, py + 28), (80, 220, 80), 1)

    def _draw_info(self, img: np.ndarray, state: DemoMonitorState, x: int, y: int) -> None:
        confidence = max(0.0, min(1.0, float(state.confidence)))

        current_y = y + 60

        self._text(img, "Turn-shift confidence", x + 35, current_y, 0.62)
        self._text(img, f"{confidence:.2f}", x + 420, current_y, 0.95, (90, 220, 90))

        current_y += 28
        self._bar(img, x + 35, current_y, 520, 24, confidence)

        current_y += 60
        self._text(img, f"Baseline wait:        {state.baseline_wait_ms} ms", x + 35, current_y, 0.58)
        current_y += 28
        self._text(img, f"Adjusted wait:        {state.adjusted_wait_ms} ms", x + 35, current_y, 0.58)
        current_y += 28
        self._text(
            img,
            f"Wait reduction:       {state.baseline_wait_ms - state.adjusted_wait_ms} ms",
            x + 35,
            current_y,
            0.58,
        )

        current_y += 28

        if state.latest_delay_ms is None:
            delay_text = "Latest response delay:  n/a"
        else:
            delay_text = f"Latest response delay:  {state.latest_delay_ms} ms"

        self._text(img, delay_text, x + 35, current_y, 0.58)

        current_y += 42
        self._status(img, "Speech detected", state.speech_detected, x + 35, current_y)
        current_y += 30
        self._status(img, "Response allowed", state.response_allowed, x + 35, current_y)
        current_y += 30
        self._status(img, "System speaking", state.system_speaking, x + 35, current_y)

    def _draw_log(self, img: np.ndarray, text: str, x: int, y: int) -> None:
        text = text or "No event yet"
        text = text[-150:]
        self._text(img, text, x + 25, y + 82, 0.65, (220, 220, 220))

    def _bar(self, img: np.ndarray, x: int, y: int, w: int, h: int, value: float) -> None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (140, 140, 140), 1)

        fill_w = int(w * value)
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), (80, 200, 80), -1)

        self._text(img, "0.00", x, y + h + 18, 0.45, (200, 200, 200))
        self._text(img, "0.50", x + w // 2 - 20, y + h + 18, 0.45, (200, 200, 200))
        self._text(img, "1.00", x + w - 45, y + h + 18, 0.45, (200, 200, 200))

    def _status(self, img: np.ndarray, label: str, value: bool, x: int, y: int) -> None:
        self._text(img, f"{label}:", x, y, 0.58)
        status_text = "Yes" if value else "No"
        colour = (80, 220, 80) if value else (130, 130, 130)
        self._text(img, status_text, x + 250, y, 0.58, colour)

    def _text(
            self,
            img: np.ndarray,
            text: str,
            x: int,
            y: int,
            scale: float = 0.65,
            colour: Tuple[int, int, int] = (215, 215, 215),
    ) -> None:
        cv2.putText(
            img,
            text,
            (x, y),
            self.font,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )

    def _fit(self, frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h))