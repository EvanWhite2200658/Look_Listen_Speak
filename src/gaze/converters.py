# src/gaze/converters.py

from __future__ import annotations

from typing import Any, Optional, Tuple
from schemas import GazeSample


def _safe_xy(value: Any) -> Optional[Tuple[float, float]]:
    """
    Convert a coordinate-like object into a (x, y) tuple.
    Accepts tuples, lists, numpy arrays, or objects with x/y attributes.
    """
    if value is None:
        return None

    if hasattr(value, "__len__") and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            return None

    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return float(value.x), float(value.y)
        except (TypeError, ValueError):
            return None

    return None


def gaze_info_to_sample(gaze_info: Any) -> GazeSample:
    """
    Convert GazeFollower gaze_info object to standard GazeSample.
    """
    timestamp_ns = int(getattr(gaze_info, "timestamp", 0))

    raw_xy = _safe_xy(getattr(gaze_info, "raw_gaze_coordinates", None))
    calibrated_xy = _safe_xy(getattr(gaze_info, "calibrated_gaze_coordinates", None))
    filtered_xy = _safe_xy(getattr(gaze_info, "filtered_gaze_coordinates", None))

    left_eye_openness = getattr(gaze_info, "left_openness", None)
    right_eye_openness = getattr(gaze_info, "right_openness", None)
    tracking_state = getattr(gaze_info, "tracking_state", None)
    status = getattr(gaze_info, "status", None)

    raw_features = getattr(gaze_info, "features", None)
    if raw_features is None:
        features = []
    else:
        try:
            features = [float(x) for x in raw_features]
        except TypeError:
            features = []

    return GazeSample(
        timestamp_ns=timestamp_ns,
        raw_xy=raw_xy,
        calibrated_xy=calibrated_xy,
        filtered_xy=filtered_xy,
        left_eye_openness=(
            float(left_eye_openness) if left_eye_openness is not None else None
        ),
        right_eye_openness=(
            float(right_eye_openness) if right_eye_openness is not None else None
        ),
        tracking_state=int(tracking_state) if tracking_state is not None else None,
        status=bool(status) if status is not None else None,
        features=features,
    )

