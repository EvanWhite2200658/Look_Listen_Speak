# src/gaze/converters.py

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple

from schemas import FaceSample, GazeSample


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

def _safe_rect(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None

    if hasattr(value, "__len__") and len(value) >= 4:
        try:
            return(
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )
        except (TypeError, ValueError, IndexError):
            return None

    return None

def _safe_landmarks(value: Any) -> list[Tuple[float, float, float]]:
    if value is None:
        return []

    landmarks: list[Tuple[float, float, float]] = []

    try:
        for point in value:
            if hasattr(point, "__len__") and len(point) >= 3:
                landmarks.append((float(point[0]), float(point[1]), float(point[2])))
            elif hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                landmarks.append((float(point.x), float(point.y), float(point.z)))
    except TypeError:
        return []
    except (ValueError, IndexError):
        return []

    return landmarks



def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_tracking_state(value: Any) -> Optional[int]:
    """
    Convert tracking state into an integer if possible.
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

def _safe_event(value: Any) -> Optional[int]:
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

def _safe_features(value: Any) -> list[float]:
    if value is None:
        return []

    try:
        return [float(x) for x in value]
    except TypeError:
        return []
    except ValueError:
        return []

def face_info_to_face_sample(face_info: Any) -> Optional[FaceSample]:
    if face_info is None:
        return None

    return FaceSample(
        face_rect=_safe_rect(getattr(face_info, "face_rect", None)),
        left_rect=_safe_rect(getattr(face_info, "left_rect", None)),
        right_rect=_safe_rect(getattr(face_info, "right_rect", None)),
        face_landmarks=_safe_landmarks(getattr(face_info, "face_landmarks", None)),
        left_eye_openness=_float_or_none(getattr(face_info, "left_eye_openness", None)),
        right_eye_openness=_float_or_none(getattr(face_info, "right_eye_openness", None)),
        can_gaze_estimation=_safe_status(getattr(face_info, "can_gaze_estimation", None)),
    )

def face_gaze_to_sample(face_info: Any, gaze_info: Any) -> Optional[GazeSample]:
    if gaze_info is None:
        return None

    timestamp_ns = int(getattr(gaze_info, "timestamp", 0))

    raw_xy = _safe_xy(getattr(gaze_info, "raw_gaze_coordinates", None))
    calibrated_xy = _safe_xy(getattr(gaze_info, "calibrated_gaze_coordinates", None))
    filtered_xy = _safe_xy(getattr(gaze_info, "filtered_gaze_coordinates", None))

    face_sample = face_info_to_face_sample(face_info)

    gaze_left_eye_openness = _float_or_none(getattr(gaze_info, "left_openness", None))
    gaze_right_eye_openness = _float_or_none(getattr(gaze_info, "right_openness", None))

    left_eye_openness = (
        gaze_left_eye_openness
        if gaze_left_eye_openness is not None
        else (face_sample.left_eye_openness if face_sample is not None else None)
    )
    right_eye_openness = (
        gaze_right_eye_openness
        if gaze_right_eye_openness is not None
        else (face_sample.right_eye_openness if face_sample is not None else None)
    )

    return GazeSample(
        timestamp_ns=timestamp_ns,
        raw_xy=raw_xy,
        calibrated_xy=calibrated_xy,
        filtered_xy=filtered_xy,
        left_eye_openness=left_eye_openness,
        right_eye_openness=right_eye_openness,
        tracking_state=_safe_tracking_state(getattr(gaze_info, "tracking_state", None)),
        status=_safe_status(getattr(gaze_info, "status", None)),
        event=_safe_event(getattr(gaze_info, "event", None)),
        features=_safe_features(getattr(gaze_info, "features", None)),
        face=face_sample,
    )

def gaze_info_to_sample(gaze_info: Any) -> Optional[GazeSample]:
    """
    Backward-compatible helper for code paths that only have gaze_info.
    Replaced with face_gaze_to_sample().
    :param gaze_info:
    :return:
    """
    return face_gaze_to_sample(face_info=None, gaze_info=gaze_info)