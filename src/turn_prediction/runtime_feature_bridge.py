# src/turn_prediction/runtime_feature_bridge.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.schemas import GazeSample
from .schemas import GazeWindow


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

def _best_eye_xy(sample: GazeSample) -> tuple[float, float]:
    """
    Derive a live eye-position estimate from eye-region geometry.

    Priority:
    1. eye rectangle centres from GazeFollower face info
    2. FaceMesh eye landmarks if available
    3. zero fallback only if neither source exists
    :param sample:
    :return:
    """

    if sample.face is not None:
        if sample.face.left_rect is not None and sample.face.right_rect is not None:
            lx, ly, lw, lh = sample.face.left_rect
            rx, ry, rw, rh = sample.face.right_rect
            left_cx = float(lx + (lw / 2.0))
            left_cy = float(ly + (lh / 2.0))
            right_cx = float(rx + (rw / 2.0))
            right_cy = float(ry + (rh / 2.0))
            return (left_cx + right_cx) / 2.0, (left_cy + right_cy) / 2.0

    landmarks = sample.face.face_landmarks
    if len(landmarks) > 263:
        left_eye_indices = [33, 133, 159, 145]
        right_eye_indices = [362, 263, 386, 374]
        eye_points = [landmarks[i] for i in [*left_eye_indices, *right_eye_indices] if i < len(landmarks)]
        if eye_points:
            xs = [float(p[0]) for p in eye_points]
            ys = [float(p[1]) for p in eye_points]
            return float(np.mean(xs)), float(np.mean(ys))

    return 0.0, 0.0

def _face_center_xy(sample: GazeSample) -> tuple[float, float]:
    if sample.face is None or sample.face.face_rect is None:
        return 0.0, 0.0

    x, y, w, h = sample.face.face_rect
    return float(x + (w / 2.0)), float(y + (h / 2.0))

def _estimate_head_pose_from_landmarks(sample: GazeSample) -> tuple[float, float, float]:
    """
    Approximate roll, pitch, yaw from live FaceMesh landmarks.

    This is lightweight geometric estimate rather than full PnP pose recovery,
    but it gives meaningful non-zero live head-pose features from the landmark set.
    :param sample:
    :return:
    """
    if sample.face is None:
        return 0.0, 0.0, 0.0

    landmarks = sample.face.face_landmarks
    if len(landmarks) <= 291:
        return 0.0, 0.0, 0.0

    nose = landmarks[1]
    chin = landmarks[152]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    nose_x, nose_y, nose_z = map(float, nose)
    chin_x, chin_y, chin_z = map(float, chin)
    left_eye_x, left_eye_y, left_eye_z = map(float, left_eye_outer)
    right_eye_x, right_eye_y, right_eye_z = map(float, right_eye_outer)
    mouth_left_x, mouth_left_y, mouth_left_z = map(float, mouth_left)
    mouth_right_x, mouth_right_y, mouth_right_z = map(float, mouth_right)

    eye_dx = right_eye_x - left_eye_x
    eye_dy = right_eye_y - left_eye_y
    eye_dz = right_eye_z - left_eye_z
    eye_dist_xy = max(np.hypot(eye_dx, eye_dy), 1e-6)

    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    eye_mid_y = (left_eye_y + right_eye_y) / 2.0
    eye_mid_z = (left_eye_z + right_eye_z) / 2.0

    mouth_mid_x = (mouth_left_x + mouth_right_x) / 2.0
    mouth_mid_y = (mouth_left_y + mouth_right_y) / 2.0
    mouth_mid_z = (mouth_left_z + mouth_right_z) / 2.0

    #Roll: tilt of the eye line in image space
    roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

    # Yaw: horizontal displacement of nose from eye midpoint, normalised by eye width
    yaw = float(np.degrees(np.arctan2(nose_x - eye_mid_x, eye_dist_xy)))

    # Pitch: vertical displacement of nose relative to eye-mouth facial axis
    facial_axis_y = mouth_mid_y - eye_mid_y
    facial_axis_z = mouth_mid_z - eye_mid_z
    pitch = float(np.degrees(np.arctan2(nose_y - ((eye_mid_y + mouth_mid_y) / 2.0), max(abs(facial_axis_y), 1e-6))))

    return roll, pitch, yaw

def sample_to_dataset_base_features(sample: GazeSample) -> np.ndarray:
    gaze_x, gaze_y = _best_gaze_xy(sample)
    eye_x, eye_y = _best_eye_xy(sample)
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