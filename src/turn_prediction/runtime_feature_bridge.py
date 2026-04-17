# src/turn_prediction/runtime_feature_bridge.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.schemas import GazeSample
from schemas import GazeWindow


@dataclass(frozen=True)
class RuntimeBridgeConfig:
    """
    Runtime feature bridge for the strongest dataset-trained model.

    This bridge tries to preserve the trained feature contract as closely as
    possible from live gaze-wrapper signals
    """

    include_context_placeholders: bool = True


BASE_DATASET_FEATURES = [
    "eye.x",
    "eye.y",
    "eye.z",
    "gaze.x",
    "gaze.y",
    "gaze.z",
    "head.x",
    "head.y",
    "head.z",
    "headpose.roll",
    "headpose.pitch",
    "headpose.yaw",
]

DELTA_DATASET_FEATURES = [f"delta_{name}" for name in BASE_DATASET_FEATURES]

STRONGEST_MODEL_CONTEXT_FEATURES = [
    "speaking",
    "target0.speaking",
    "turn_shift",
    "target0.x",
    "target0.y",
    "target0.z",
]

STRONGEST_MODEL_RUNTIME_FEATURES = [
    *BASE_DATASET_FEATURES,
    *DELTA_DATASET_FEATURES,
    *STRONGEST_MODEL_CONTEXT_FEATURES,
]

def _xy_or_default(value: Optional[tuple[float, float]]) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    return float(value[0]), float(value[1])

def _best_gaze_xy(sample: GazeSample) -> tuple[float, float]:
    return _xy_or_default(sample.filtered_xy or sample.calibrated_xy or sample.filtered_xy)

def _face_center_xy(sample: GazeSample) -> tuple[float, float]:
    if sample.face is None or sample.face.face_rect is None:
        return 0.0, 0.0

    x, y, w, h = sample.face.face_rect
    return float(x + (w / 2.0)), float(y + (h / 2.0))

def _estimate_head_pose_from_landmarks(sample: GazeSample) -> tuple[float, float, float]:
    """
    Placeholder hook for live head pose estimation from FaceMesh
    :param sample:
    :return:
    """
    return 0.0, 0.0, 0.0

def sample_to_dataset_base_features(sample: GazeSample) -> np.ndarray:
    gaze_x, gaze_y = _best_gaze_xy(sample)
    eye_x, eye_y = gaze_x, gaze_y # TODO: make these standalone and not derived
    head_x, head_y = _face_center_xy(sample)

    # unavailable live
    eye_z = 0.0
    gaze_z = 0.0
    head_z = 0.0

    # live-derived from FaceMesh where available
    headpose_roll, headpose_pitch, headpose_yaw = _estimate_head_pose_from_landmarks(sample)

    return np.asarray(
        [
            eye_x,
            eye_y,
            eye_z,
            gaze_x,
            gaze_y,
            gaze_z,
            head_x,
            head_y,
            head_z,
            headpose_roll,
            headpose_pitch,
            headpose_yaw,
        ],
        dtype=np.float32,
    )

def _append_deltas(sequence: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(sequence, dtype=np.float32)
    if sequence.shape[0] > 1:
        deltas[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, deltas], axis=1)

def _context_placeholders(sequence_len: int) -> np.ndarray:
    return np.zeros((sequence_len, len(STRONGEST_MODEL_CONTEXT_FEATURES)), dtype=np.float32)

def gaze_window_to_strongest_runtime_sequence(
        window: GazeWindow,
        config: RuntimeBridgeConfig,
) -> np.ndarray:
    vectors = [sample_to_dataset_base_features(sample) for sample in window.samples]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    base = np.stack(vectors, axis=0).astype(np.float32)
    sequence = _append_deltas(base)

    if config.include_context_placeholders:
        sequence = np.concatenate([sequence, _context_placeholders(sequence.shape[0])], axis=1)

    return sequence.astype(np.float32)

def get_strongest_runtime_feature_names(config: RuntimeBridgeConfig) -> list[str]:
    names = [*BASE_DATASET_FEATURES, *DELTA_DATASET_FEATURES]
    if config.include_context_placeholders:
        names.extend(STRONGEST_MODEL_CONTEXT_FEATURES)
    return names

def get_strongest_runtime_feature_dim(config: RuntimeBridgeConfig) -> int:
    return len(get_strongest_runtime_feature_names(config))