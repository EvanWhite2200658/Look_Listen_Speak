# src/turn_prediction/runtime_hcs_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.schemas import GazeSample
from .schemas import GazeWindow


@dataclass(frozen=True)
class RuntimeHCSAdapterConfig:
    include_deltas: bool = True


BASE_HCS_STYLE_FEATURE_NAMES = [
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

DELTA_HCS_STYLE_FEATURE_NAMES = [
    f"delta_{name}" for name in BASE_HCS_STYLE_FEATURE_NAMES
]

ALL_HCS_STYLE_FEATURE_NAMES = [
    *BASE_HCS_STYLE_FEATURE_NAMES,
    *DELTA_HCS_STYLE_FEATURE_NAMES,
]


LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
RIGHT_EYE_LANDMARKS = [362, 263, 386, 374]
NOSE_INDEX = 1
CHIN_INDEX = 152
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
LEFT_FACE_INDEX = 234
RIGHT_FACE_INDEX = 454


def _has_landmark_indices(landmarks: list[tuple[float, float, float]], indices: list[int]) -> bool:
    return all(index < len(landmarks) for index in indices)


def _mean_landmarks(
    landmarks: list[tuple[float, float, float]],
    indices: list[int],
) -> tuple[float, float, float]:
    points = [landmarks[index] for index in indices]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))


def _face_scale(sample: GazeSample) -> float:
    if sample.face is not None and sample.face.face_rect is not None:
        _, _, w, h = sample.face.face_rect
        return max(float((w + h) / 2.0), 1e-6)
    if sample.face is not None and sample.face.face_landmarks:
        landmarks = sample.face.face_landmarks
        if _has_landmark_indices(landmarks, [LEFT_FACE_INDEX, RIGHT_FACE_INDEX]):
            left_face = landmarks[LEFT_FACE_INDEX]
            right_face = landmarks[RIGHT_FACE_INDEX]
            return max(float(np.linalg.norm(np.asarray(right_face) - np.asarray(left_face))), 1e-6)
    return 1.0


def _head_origin(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is None:
        return 0.0, 0.0, 0.0

    landmarks = sample.face.face_landmarks
    if landmarks and _has_landmark_indices(landmarks, [NOSE_INDEX, CHIN_INDEX, LEFT_FACE_INDEX, RIGHT_FACE_INDEX]):
        nose = np.asarray(landmarks[NOSE_INDEX], dtype=np.float32)
        chin = np.asarray(landmarks[CHIN_INDEX], dtype=np.float32)
        left_face = np.asarray(landmarks[LEFT_FACE_INDEX], dtype=np.float32)
        right_face = np.asarray(landmarks[RIGHT_FACE_INDEX], dtype=np.float32)
        origin = (nose + chin + left_face + right_face) / 4.0
        return float(origin[0]), float(origin[1]), float(origin[2])

    if sample.face.face_rect is not None:
        x, y, w, h = sample.face.face_rect
        return float(x + (w / 2.0)), float(y + (h / 2.0)), float((w + h) / 2.0)

    return 0.0, 0.0, 0.0


def _estimate_head_pose_from_landmarks(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is None:
        return 0.0, 0.0, 0.0

    landmarks = sample.face.face_landmarks
    required = [NOSE_INDEX, CHIN_INDEX, LEFT_EYE_LANDMARKS[0], RIGHT_EYE_LANDMARKS[1], MOUTH_LEFT_INDEX, MOUTH_RIGHT_INDEX]
    if not landmarks or not _has_landmark_indices(landmarks, required):
        return 0.0, 0.0, 0.0

    nose = np.asarray(landmarks[NOSE_INDEX], dtype=np.float32)
    left_eye = np.asarray(landmarks[LEFT_EYE_LANDMARKS[0]], dtype=np.float32)
    right_eye = np.asarray(landmarks[RIGHT_EYE_LANDMARKS[1]], dtype=np.float32)
    mouth_left = np.asarray(landmarks[MOUTH_LEFT_INDEX], dtype=np.float32)
    mouth_right = np.asarray(landmarks[MOUTH_RIGHT_INDEX], dtype=np.float32)

    eye_vec = right_eye - left_eye
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (mouth_left + mouth_right) / 2.0

    roll = float(np.radians(np.arctan2(eye_vec[1], eye_vec[0])))
    yaw = float(np.radians(np.arctan2(nose[0] - eye_mid[0], max(np.linalg.norm(eye_vec[:2]), 1e-6))))
    pitch = float(np.radians(np.arctan2(nose[2] - eye_mid[2], max(np.linalg.norm(mouth_mid - eye_mid), 1e-6))))
    return roll, pitch, yaw


def _rotation_matrix_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]], dtype=np.float32)
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]], dtype=np.float32)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _best_eye_xyz(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is not None:
        landmarks = sample.face.face_landmarks
        if landmarks and _has_landmark_indices(landmarks, LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS):
            left_eye = np.asarray(_mean_landmarks(landmarks, LEFT_EYE_LANDMARKS))
            right_eye = np.asarray(_mean_landmarks(landmarks, RIGHT_EYE_LANDMARKS))
            eye_world = (left_eye + right_eye) / 2.0

            head_origin = np.asarray(_head_origin(sample), dtype=np.float32)
            scale = _face_scale(sample)

            eye_rel = (eye_world - head_origin) / max(scale, 1e-6)
            return float(eye_rel[0]), float(eye_rel[1]), float(eye_rel[2])

    return 0.0, 0.0, 0.0

def _face_center_2d(sample: GazeSample) -> tuple[float, float]:
    if sample.face is not None and sample.face.face_rect is not None:
        x, y, w, h = sample.face.face_rect
        return float(x + w / 2.0), float(y + h / 2.0)

    return 0.0, 0.0

def _best_gaze_vector(sample: GazeSample) -> np.ndarray:
    gaze_xy = sample.filtered_xy or sample.calibrated_xy or sample.raw_xy
    if gaze_xy is None:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    gx, gy = float(gaze_xy[0]), float(gaze_xy[1])

    cx, cy = _face_center_2d(sample)
    scale = _face_scale(sample)

    dx = (gx - cx) / max(scale, 1e-6)
    dy = (gy - cy) / max(scale, 1e-6)
    print(f"dx={dx:.4f}, dy={dy:.4f}")
    vec = np.array([dx, -dy, 1.0], dtype=np.float32)
    return vec / max(np.linalg.norm(vec), 1e-6)


def sample_to_hcs_style_base_features(sample: GazeSample) -> np.ndarray:
    scale = _face_scale(sample)
    head_origin = np.asarray(_head_origin(sample), dtype=np.float32)
    roll, pitch, yaw = _estimate_head_pose_from_landmarks(sample)
    rotation = _rotation_matrix_from_euler_deg(roll, pitch, yaw)
    inv_rotation = rotation.T

    eye_world = np.asarray(_best_eye_xyz(sample), dtype=np.float32)
    eye_rel = inv_rotation @ ((eye_world - head_origin) / scale)

    gaze_vec_world = _best_gaze_vector(sample)
    gaze_vec_head = inv_rotation @ gaze_vec_world

    head_world = np.asarray(_head_origin(sample), dtype=np.float32)
    head_rel = (head_world - head_world.mean()) / _face_scale(sample)

    return np.asarray(
        [
            float(eye_rel[0]),
            float(eye_rel[1]),
            float(eye_rel[2]),
            float(gaze_vec_head[0]),
            float(gaze_vec_head[1]),
            float(gaze_vec_head[2]),
            float(head_rel[0]),
            float(head_rel[1]),
            float(head_rel[2]),
            float(roll),
            float(pitch),
            float(yaw),
        ],
        dtype=np.float32,
    )


def _append_deltas(sequence: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(sequence, dtype=np.float32)
    if sequence.shape[0] > 1:
        deltas[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, deltas], axis=1)


def gaze_window_to_hcs_style_sequence(
    window: GazeWindow,
    config: RuntimeHCSAdapterConfig,
) -> np.ndarray:
    vectors = [sample_to_hcs_style_base_features(sample) for sample in window.samples]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    sequence = np.stack(vectors, axis=0).astype(np.float32)
    if config.include_deltas:
        sequence = _append_deltas(sequence)
    return sequence.astype(np.float32)


def get_hcs_style_feature_names(config: RuntimeHCSAdapterConfig) -> list[str]:
    if not config.include_deltas:
        return list(BASE_HCS_STYLE_FEATURE_NAMES)
    return list(ALL_HCS_STYLE_FEATURE_NAMES)


def get_hcs_style_feature_dim(config: RuntimeHCSAdapterConfig) -> int:
    return len(get_hcs_style_feature_names(config))