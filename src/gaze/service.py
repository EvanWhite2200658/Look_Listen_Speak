# src/gaze/service.py

from __future__ import annotations

import threading
import time
from typing import Optional

from schemas import GazeSample
from .wrapper import GazeTrackerModule


class GazeTrackingService:
    """
    Background service for continuous gaze tracking.
    """

    def __init__(
        self,
        max_buffer_size: int = 150,
        poll_interval_s: float = 0.03,
        run_calibration: bool = False,
    ) -> None:
        self._tracker = GazeTrackerModule(max_buffer_size=max_buffer_size)
        self._poll_interval_s = poll_interval_s
        self._run_calibration = run_calibration

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self, preview: bool = False) -> None:
        """
        Start the gaze tracking service.
        Args:
            preview: whether to launch preview before sampling
        :param preview:
        :return:
        """
        if self._started:
            return

        if preview:
            self._tracker.start_preview()

        if self._run_calibration:
            self._tracker.calibrate()

        if not self._tracker.has_calibration:
            print(
                "GazeTrackingService.start(): calibration not available. "
                "Run calibration first or set run_calibration=True."
            )
            return

        self._tracker.start_sampling()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="GazeTrackingService",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _run_loop(self) -> None:
        """
        background polling loop.
        :return:
        """
        while not self._stop_event.is_set():
            try:
                self._tracker.poll_latest_from_tracker()
            except Exception as exc:
                print(f"GazeTrackingService._run_loop(): stopping service loop: {exc}")
                self._stop_event.set()
                break

            time.sleep(self._poll_interval_s)

    def stop(self) -> None:
        """
        Stop the gaze tracking service and release resources.
        :return:
        """
        if not self._started:
            return

        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._tracker.stop()
        self._started = False

    def get_latest_sample(self) -> Optional[GazeSample]:
        """
        return the latest buffered gaze sample
        :return:
        """
        return self._tracker.get_latest_sample()

    def get_recent_window(self, window_size: int) -> list[GazeSample]:
        """
        Return the latest N gaze samples.
        :param window_size:
        :return:
        """
        return self._tracker.get_recent_window(window_size)

    @property
    def is_running(self) -> bool:
        return self._started