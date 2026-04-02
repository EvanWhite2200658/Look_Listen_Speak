# src/gaze/wrapper.py

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional

from gazefollower import GazeFollower

from converters import gaze_info_to_sample
from schemas import GazeSample


class GazeTrackerModule:
    """
    Project-level wrapper for GazeFollower
    - own tracker instance
    - collect standardised gaze samples
    - provide latest sample and temporal window
    """

    def __init__(self, max_buffer_size: int = 150) -> None:
        self._tracker = GazeFollower()
        self._buffer: Deque[GazeSample] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._is_running = False

    def start_preview(self) -> None:
        """
        Start preview for camera
        :return:
        """
        self._tracker.preview()
        self._is_running = True

    def calibrate(self) -> None:
        """
        run the GazeFollower calibration
        :return:
        """
        self._tracker.calibrate()

    def start_sampling(self) -> None:
        """
        start sampling and collecting gaze samples into the project buffer.
        :return:
        """
        if self._is_running:
            return

        self._tracker.start_sampling()
        self._is_running = True

    def poll_latest_from_tracker(self) -> Optional[GazeSample]:
        """
        Pull the latest gaze information from GazeFollower and store it in the buffer
        kept simple for initial integration, can be replaced with event-driven design if needed.
        :return:
        """
        gaze_info = self._tracker.get_gaze_info()
        if gaze_info is None:
            return None

        try:
            sample = gaze_info_to_sample(gaze_info)
        except Exception as exc:
            print(f"GazeTrackerModule.poll_latest_from_tracker(): conversion warning: {exc}")
            return None

        with self._lock:
            should_append = (
                not self._buffer or self._buffer[-1].timestamp_ns != sample.timestamp_ns
            )
            if should_append:
                self._buffer.append(sample)
        return sample

    def get_buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_latest_sample(self) -> Optional[GazeSample]:
        """
        return the newest buffered sample.
        :return:
        """
        with self._buffer:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def get_recent_window(self, window_size: int) -> List[GazeSample]:
        """
        Return the latest N gaze samples for temporal modelling.
        :param window_size:
        :return:
        """
        with self._lock:
            if window_size <= 0:
                return []
            buffer_list = list(self._buffer)
            return buffer_list[-window_size:]

    def stop(self) -> None:
        """
        stop sampling and release resources.
        :return:
        """
        if not self._is_running:
            return

        try:
            self._tracker.stop()
        except Exception as exc:
            print(f"GazeTrackerModule.stop(): stop_sampling warning: {exc}")

        try:
            self._tracker.release()
        except Exception as exc:
            print(f"GazeTrackerModule.stop(): release warning: {exc}")

        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running