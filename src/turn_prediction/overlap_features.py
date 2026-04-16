# src/turn_prediction/overlap_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.gaze.schemas import GazeSample
from schemas import GazeWindow


BASE_FEATURE_NAMES: List[str] = [
    "eye.x", "eye.y", "eye.z",
    "gaze.x", "gaze.y", "gaze.z",
    "head.x", "head.y", "head.z",
    "headpose.roll", "headpose.pitch", "headpose.yaw",
]

DELTA_FEATURE_NAMES: List[str] = [f"delta_{name}" for name in BASE_FEATURE_NAMES]

ALL_FEATURE_NAMES: List[str] = [*BASE_FEATURE_NAMES, *DELTA_FEATURE_NAMES]


@dataclass(frozen=True)
class OverlapFeatureConfig:
    include_deltas: bool = True


def _safe_xy(xy):
    if xy is None:
        return 0.0, 0.0
    return float(xy[0]), float(xy[1])

def _face_center(sample:GazeSample):
    if sample.face is None or sample.face.face_rect is None:
        return 0.0, 0.0
    x, y, w, h = sample.face.face_rect
    return float(x + w / 2), float(y + h / 2)

def sample_to_overlap_base(sample: GazeSample) -> np.ndarray:
    # gaze-derived
    gx, gy = _safe_xy(sample.filtered_xy or sample.calibrated_xy or sample.raw_xy)

    # eye proxy (reusing gaze for consistent signal
    ex, ey = gx, gy

    # head proxy (face centre)
    hx, hy = _face_center(sample)

    # depth proxies (not available -> 0)
    ez = 0.0
    gz = 0.0
    hz = 0.0

    # headpose placeholders (future upgrade)
    roll = 0.0
    pitch = 0.0
    yaw = 0.0

    return np.array([
        ex, ey, ez,
        gx, gy, gz,
        hx, hy, hz,
        roll, pitch, yaw,
    ], dtype=np.float32)


def _append_deltas(sequence: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(sequence)
    if sequence.shape[0] > 1:
        deltas[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, deltas], axis=1)


def gaze_window_to_overlap_sequence(
        window: GazeWindow,
        config: OverlapFeatureConfig,
) -> np.ndarray:
    vectors = [sample_to_overlap_base(s) for s in window.samples]

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    seq = np.stack(vectors)

    if config.include_deltas:
        seq = _append_deltas(seq)

    return seq.astype(np.float32)

def get_overlap_feature_dim(config: OverlapFeatureConfig) -> int:
    base = len(BASE_FEATURE_NAMES)
    return base * 2 if config.include_deltas else base