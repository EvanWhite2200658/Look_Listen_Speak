# src/gaze/converters.py

from __future__ import annotations

from enum import Enum
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

def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_tracking_state(value: Any) -> Optional[int]:
    """
    COnvert tracking state into an integer if possible.
    :param value:
    :return:
    """
    if value is None:
        return None

    if isinstance(value, Enum):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            return None

    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_status(value: Any) -> Optional[bool]:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    try:
        return bool(value)
    except Exception:
        return None

def gaze_info_to_sample(gaze_info: Any) -> GazeSample:
    """
    Convert GazeFollower gaze_info object to standard GazeSample.
    """
    timestamp_ns = int(getattr(gaze_info, "timestamp", 0))

    raw_xy = _safe_xy(getattr(gaze_info, "raw_gaze_coordinates", None))
    calibrated_xy = _safe_xy(getattr(gaze_info, "calibrated_gaze_coordinates", None))
    filtered_xy = _safe_xy(getattr(gaze_info, "filtered_gaze_coordinates", None))

    left_eye_openness = _float_or_none(getattr(gaze_info, "left_openness", None))
    right_eye_openness = _float_or_none(getattr(gaze_info, "right_openness", None))
    tracking_state = _safe_tracking_state(getattr(gaze_info, "tracking_state", None))
    status = _safe_status(getattr(gaze_info, "status", None))

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
        left_eye_openness=left_eye_openness,
        right_eye_openness=right_eye_openness,
        tracking_state=tracking_state,
        status=status,
        features=features,
    )

