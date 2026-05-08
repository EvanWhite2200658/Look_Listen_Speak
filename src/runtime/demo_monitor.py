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
    mode: str = "gaze"
    queue_size: int = 0
    inference_time_ms: Optional[float] = None


class DemoMonitor:
    def __init__(
        self,
        window_name: str = "HRI Runtime Monitor",
        width: int = 1200,
        height: int = 620,
    ) -> None:
        self.window_name = window_name
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_small = cv2.FONT_HERSHEY_COMPLEX_SMALL

    def draw(self, state: DemoMonitorState) -> None:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        self._title(canvas, "Gaze-Informed Turn-Taking Runtime Monitor")

        # Layout
        confidence_x, confidence_y, confidence_w, confidence_h = 25, 65, 450, 210
        timing_x, timing_y, timing_w, timing_h = 525, 65, 450, 210
        state_x, state_y, state_w, state_h = 25, 305, 450, 190
        log_x, log_y, log_w, log_h = 525, 305, 450, 190
        footer_x, footer_y, footer_w, footer_h = 25, 525, 950, 65

        self._panel(canvas, confidence_x, confidence_y, confidence_w, confidence_h, "Model Output")
        self._panel(canvas, timing_x, timing_y, timing_w, timing_h, "Response Timing")
        self._panel(canvas, state_x, state_y, state_w, state_h, "Runtime State")
        self._panel(canvas, log_x, log_y, log_w, log_h, "Latest Event")
        self._panel(canvas, footer_x, footer_y, footer_w, footer_h, "Demo Notes")

        self._draw_confidence(canvas, state, confidence_x, confidence_y)
        self._draw_timing(canvas, state, timing_x, timing_y)
        self._draw_runtime_state(canvas, state, state_x, state_y)
        self._draw_log(canvas, state, log_x, log_y)
        self._draw_footer(canvas, footer_x, footer_y)

        display = cv2.resize(canvas, (1500, 930), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _draw_confidence(self, img: np.ndarray, state: DemoMonitorState, x: int, y: int) -> None:
        confidence = max(0.0, min(1.0, float(state.confidence)))

        self._text(img, "Turn-shift confidence", x + 25, y + 62, 0.62)
        self._text(img, f"{confidence:.2f}", x + 300, y + 72, 1.15, (90, 220, 90))

        self._bar(img, x + 25, y + 105, 400, 28, confidence)

        if state.inference_time_ms is None:
            inference_text = "Inference time: n/a"
        else:
            inference_text = f"Inference time: {state.inference_time_ms:.1f} ms"

        self._text(img, inference_text, x + 25, y + 175, 0.55, (210, 210, 210))

    def _draw_timing(self, img: np.ndarray, state: DemoMonitorState, x: int, y: int) -> None:
        reduction = state.baseline_wait_ms - state.adjusted_wait_ms

        self._text(img, f"Mode:             {state.mode}", x + 25, y + 62, 0.58)
        self._text(img, f"Baseline wait:    {state.baseline_wait_ms} ms", x + 25, y + 95, 0.58)
        self._text(img, f"Adjusted wait:    {state.adjusted_wait_ms} ms", x + 25, y + 128, 0.58)
        self._text(img, f"Wait reduction:   {reduction} ms", x + 25, y + 161, 0.58)

        if state.latest_delay_ms is None:
            delay_text = "Latest delay:     n/a"
        else:
            delay_text = f"Latest delay:     {state.latest_delay_ms} ms"

        self._text(img, delay_text, x + 25, y + 194, 0.58)

    def _draw_runtime_state(self, img: np.ndarray, state: DemoMonitorState, x: int, y: int) -> None:
        self._status(img, "Speech detected", state.speech_detected, x + 25, y + 65)
        self._status(img, "Response allowed", state.response_allowed, x + 25, y + 105)
        self._status(img, "System speaking", state.system_speaking, x + 25, y + 145)

    def _draw_log(self, img: np.ndarray, state: DemoMonitorState, x: int, y: int) -> None:
        event = state.last_event or "No event yet"
        lines = self._wrap_text(event, max_chars=42)

        current_y = y + 65
        for line in lines[:4]:
            self._text(img, line, x + 25, current_y, 0.5, (220, 220, 220))
            current_y += 28

    def _draw_footer(self, img: np.ndarray, x: int, y: int) -> None:
        self._text(
            img,
            "Diagnostic monitor only: gaze tracking runs in the background; this dashboard shows the timing-control decision.",
            x + 20,
            y + 40,
            0.48,
            (200, 200, 200),
        )

    def _bar(self, img: np.ndarray, x: int, y: int, w: int, h: int, value: float) -> None:
        value = max(0.0, min(1.0, value))

        cv2.rectangle(img, (x, y), (x + w, y + h), (140, 140, 140), 1)

        fill_w = int(w * value)
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), (80, 200, 80), -1)

        self._text(img, "0.00", x, y + h + 22, 0.43, (200, 200, 200))
        self._text(img, "0.50", x + w // 2 - 22, y + h + 22, 0.43, (200, 200, 200))
        self._text(img, "1.00", x + w - 48, y + h + 22, 0.43, (200, 200, 200))

    def _status(self, img: np.ndarray, label: str, value: bool, x: int, y: int) -> None:
        self._text(img, f"{label}:", x, y, 0.58)

        status_text = "Yes" if value else "No"
        colour = (80, 220, 80) if value else (130, 130, 130)

        self._text(img, status_text, x + 260, y, 0.58, colour)

    def _title(self, img: np.ndarray, text: str) -> None:
        cv2.putText(
            img,
            text,
            (25, 38),
            self.font,
            0.82,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    def _panel(self, img: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.putText(
            img,
            title,
            (x + 12, y + 28),
            self.font,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _text(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float = 0.6,
        colour: Tuple[int, int, int] = (215, 215, 215),
    ) -> None:
        cv2.putText(
            img,
            text,
            (x, y),
            self.font_small if scale <= 0.6 else self.font,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )

    def _wrap_text(self, text: str, max_chars: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines