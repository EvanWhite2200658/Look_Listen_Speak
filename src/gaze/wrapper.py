# src/gaze/wrapper.py

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, List, Optional

from gazefollower import GazeFollower
from sympy.stats.rv import sample_iter_lambdify

from .patched_svr_calibration import PatchedSVRCalibration
from .converters import face_gaze_to_sample, gaze_info_to_sample
from .schemas import GazeSample


class GazeTrackerModule:
    """
    Project-level wrapper for GazeFollower
    - own tracker instance
    - collect standardised gaze samples
    - provide latest sample and temporal window
    """

    def __init__(self, max_buffer_size: int = 150) -> None:
        self._tracker = GazeFollower(calibration=PatchedSVRCalibration())
        self._buffer: Deque[GazeSample] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._is_running = False
        self._subscriber_registered = False


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
        try:
            saved = self._tracker.calibration.save_model()
            if not saved:
                print("GazeTrackerModule.calibrate(): calibration finished, but save_model() returned False")
        except Exception as exc:
            print(f"GazeTrakcerModule.calibrate(): save_model warning: {exc}")

    def _handle_face_gaze_info(self, face_info, gaze_info, *args, **kwargs) -> None:
        try:
            sample = face_gaze_to_sample(face_info=face_info, gaze_info=gaze_info)
        except Exception as exc:
            print(f"GazeTrackerModule._handle_face_gaze_info(): conversion warning: {exc}")
            return

        if sample is None:
            return

        with self._lock:
            should_append = (
                not self._buffer or self._buffer[-1].timestamp_ns != sample.timestamp_ns
            )
            if should_append:
                self._buffer.append(sample)

    def _register_subscriber(self) -> None:
        if self._subscriber_registered:
            return

        try:
            self._tracker.add_subscriber(self._handle_face_gaze_info)
            self._subscriber_registered = True
        except Exception as exc:
            print(f"GazeTrackerModule._register_subscriber(): warning: {exc}")

    def _remove_subscriber(self) -> None:
        if not self._subscriber_registered:
            return

        try:
            self._tracker.remove_subscriber(self._handle_face_gaze_info)
        except Exception as exc:
            print(f"GazeTrackerModule._remove_subscriber(): warning: {exc}")
        finally:
            self._subscriber_registered = False

    def start_sampling(self) -> None:
        """
        start sampling and collecting gaze samples into the project buffer.
        :return:
        """
        if self._is_running:
            return

        self._tracker.start_sampling()
        self._register_subscriber()
        self._is_running = True

    def poll_latest_from_tracker(self) -> Optional[GazeSample]:
        """
        Return the most recent buffered sample.

        The primary collection path is subscriber-driven so that face_info
        and gaze_info are preserved. This method remains for backwards
        compatibility with the service polling loop.
        :return:
        """
        latest = self.get_latest_sample()
        if latest is not None:
            return latest

        gaze_info = self._tracker.get_gaze_info()
        if gaze_info is None:
            return None

        try:
            sample = gaze_info_to_sample(gaze_info)
        except Exception as exc:
            print(f"GazeTrackerModule.poll_latest_from_tracker(): conversion warning: {exc}")
            return None

        if sample is None:
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
        with self._lock:
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
            self._tracker.stop_sampling()
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

    @property
    def has_calibration(self) -> bool:
        return bool(self._tracker.calibration.has_calibrated)