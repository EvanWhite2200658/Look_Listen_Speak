# src/gaze/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Rect = Tuple[float, float, float, float]
Point3D = Tuple[float, float, float]

@dataclass
class FaceSample:
    """
    Standardised face-alignment sample extracted from GazeFollower face_info.
    Keeps the project isolated from third-party face structures while preserving
    enough information for future live feature engineering.
    """

    face_rect: Optional[Rect] = None
    left_rect: Optional[Rect] = None
    right_rect: Optional[Rect] = None
    face_landmarks: List[Point3D] = field(default_factory=list)
    left_eye_openness: Optional[float] = None
    right_eye_openness: Optional[float] = None
    can_gaze_estimation: Optional[bool] = None

    @property
    def landmark_count(self) -> int:
        return len(self.face_landmarks)

@dataclass
class GazeSample:
    """
    Standardised gaze sample used project-wide
    Isolates system from third-party libraries
    """

    timestamp_ns: int
    raw_xy: Optional[Tuple[float, float]] = None
    calibrated_xy: Optional[Tuple[float, float]] = None
    filtered_xy: Optional[Tuple[float, float]] = None
    left_eye_openness: Optional[float] = None
    right_eye_openness: Optional[float] = None
    tracking_state: Optional[int] = None
    status: Optional[bool] = None
    event: Optional[bool] = None
    features : List[float] = field(default_factory=list)
    face: Optional[FaceSample] = None

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0