# src/ui/avatar_screen.py

from __future__ import annotations

import math
import threading
import tkinter as tk
from dataclasses import dataclass


@dataclass
class AvatarState:
    mode: str = "idle"  # idle, listening, thinking, speaking


class AvatarScreen:
    """
    Minimal conversational face display.
    - eyes stay fixed
    - mouth is flat while idle/listening/thinking
    - mouth becomes an animated pseudo-wave while speaking
    """

    def __init__(self, width: int = 500, height: int = 300) -> None:
        self.width = width
        self.height = height
        self.state = AvatarState()

        self._root = tk.Tk()
        self._root.title("HRI Avatar")

        self._canvas = tk.Canvas(self._root, width=width, height=height, bg="white")
        self._canvas.pack()

        self._status_text = self._canvas.create_text(
            width // 2,
            height - 30,
            text="idle",
            font=("Arial", 16),
        )

        self._mouth_phase = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """
        Tkinter must run on the main thread.
        """
        self._draw_face()
        self._tick()
        self._root.mainloop()

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.state.mode = mode

    def stop(self) -> None:
        try:
            self._root.quit()
            self._root.destroy()
        except Exception:
            pass

    def _draw_face(self) -> None:
        self._canvas.delete("face")
        self._canvas.create_oval(120, 90, 170, 140, fill="black", tags="face")
        self._canvas.create_oval(330, 90, 380, 140, fill="black", tags="face")
        self._draw_mouth()

    def _draw_mouth(self) -> None:
        self._canvas.delete("mouth")

        with self._lock:
            mode = self.state.mode

        if mode == "speaking":
            points: list[float] = []
            x0, x1 = 150, 350
            y_mid = 210
            width = x1 - x0

            for i in range(40):
                x = x0 + (i / 39.0) * width
                y = y_mid + 12.0 * math.sin((i / 39.0) * 4.0 * math.pi + self._mouth_phase)
                points.extend([x, y])

            self._canvas.create_line(*points, fill="black", width=3, smooth=True, tags="mouth")
        else:
            self._canvas.create_line(150, 210, 350, 210, fill="black", width=3, tags="mouth")

    def _tick(self) -> None:
        with self._lock:
            mode = self.state.mode

        self._canvas.itemconfig(self._status_text, text=mode)

        if mode == "speaking":
            self._mouth_phase += 0.4

        self._draw_mouth()
        self._root.after(50, self._tick)