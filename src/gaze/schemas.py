# src/gaze/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    features : List[float] = field(default_factory=list)

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0