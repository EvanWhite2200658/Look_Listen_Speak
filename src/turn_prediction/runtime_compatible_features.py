# src/turn_prediction/runtime_compatible_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.schemas import GazeSample
from .schemas import GazeWindow

@dataclass(frozen=True)
class RuntimeCompatibleFeatureConfig:
    include_deltas: bool = True


BASE_RUNTIME_COMPATIBLE_FEATURE_NAMES = [
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

DELTA_RUNTIME_COMPATIBLE_FEATURE_NAMES = [f"delta_{name}" for name in BASE_RUNTIME_COMPATIBLE_FEATURE_NAMES]

ALL_RUNTIME_COMPATIBLE_FEATURE_NAMES = [
    *BASE_RUNTIME_COMPATIBLE_FEATURE_NAMES,
    *DELTA_RUNTIME_COMPATIBLE_FEATURE_NAMES,
]

LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
RIGHT_EYE_LANDMARKS = [362, 263, 386, 374]
NOSE_INDEX = [1]
CHIN_INDEX = [152]
MOUTH_LEFT_INDEX = [61]
MOUTH_RIGHT_INDEX = [291]
LEFT_FACE_INDEX = [234]
RIGHT_FACE_INDEX = [454]

def _xy_or_default(value: Optional[tuple[float, float]]) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    return float(value[0]), float(value[1])

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

def _best_eye_xyz(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is not None:
        landmarks = sample.face.face_landmarks
        if landmarks and _has_landmark_indices(landmarks, LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS):
            left_eye = _mean_landmarks(landmarks, LEFT_EYE_LANDMARKS)
            right_eye = _mean_landmarks(landmarks, RIGHT_EYE_LANDMARKS)
            eye_x = (left_eye[0] + right_eye[0]) / 2.0
            eye_y = (left_eye[1] + right_eye[1]) / 2.0
            eye_z = (left_eye[2] + right_eye[2]) / 2.0
            return eye_x, eye_y, eye_z

        if sample.face.left_rect is not None and sample.face.right_rect is not None:
            lx, ly, lw, lh = sample.face.left_rect
            rx, ry, rw, rh = sample.face.right_rect
            eye_x = float((lx + (lw / 2.0) + rx + (rw / 2.0)) / 2.0)
            eye_y = float((ly + (lh / 2.0) + ry + (rh / 2.0)) / 2.0)
            eye_z = float(((lw + lh) + (rw + rh)) / 4.0)
            return eye_x, eye_y, eye_z

    return 0.0, 0.0, 0.0

def _best_gaze_xyz(sample: GazeSample) -> tuple[float, float, float]:
    gaze_x, gaze_y = _xy_or_default(sample.filtered_xy or sample.calibrated_xy or sample.raw_xy)

    gaze_z = 0.0
    if sample.face is not None:
        landmarks = sample.face.face_landmarks
        if landmarks and _has_landmark_indices(landmarks, [NOES_INDEX]):
            gaze_z = float(landmarks[NOSE_INDEX][2])
        elif sample.face.face_rect is not None:
            _, _, w, h = sample.face.face_rect
            gaze_z = float((w + h) / 2.0)

    return gaze_x, gaze_y, gaze_z

def _best_head_xyz(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is None:
        return 0.0, 0.0, 0.0

    landmarks = sample.face.face_landmarks
    if landmarks and _has_landmark_indices(landmarks, [LEFT_FACE_INDEX, RIGHT_FACE_INDEX, NOSE_INDEX, CHIN_INDEX]):
        left_face = landmark[LEFT_FACE_INDEX]
        right_face = landmarks[RIGHT_FACE_INDEX]
        nose = landmarks[NOSE_INDEX]
        chin = landmarks[CHIN_INDEX]
        head_x = float((left_face[0] + right_face[0] + nose[0] + chin[0]) / 4.0)
        head_y = float((left_face[1] + right_face[1] + nose[1] + chin[1]) / 4.0)
        head_z = float((left_face[2] + right_face[2] + nose[2] + chin[2]) / 4.0)
        return head_x, head_y, head_z

    if sample.face.face_rect is not None:
        x, y, w, h = sample.face.face_rect
        head_x = float(x + (w / 2.0))
        head_y = float(y + (h / 2.0))
        head_z = float((w + h) / 2.0)
        return head_x, head_y, head_z

    return 0.0, 0.0, 0.0

def _estimate_head_pose_from_landmarks(sample: GazeSample) -> tuple[float, float, float]:
    if sample.face is None:
        return 0.0, 0.0, 0.0

    landmarks = sample.face.face_landmarks
    required = [
        NOSE_INDEX,
        CHIN_INDEX,
        LEFT_EYE_LANDMARKS[0],
        RIGHT_EYE_LANDMARKS[1],
        MOUTH_LEFT_INDEX,
        MOUTH_RIGHT_INDEX,
    ]
    if not landmarks or not _has_landmark_indices(landmarks, required):
        return 0.0, 0.0, 0.0

    nose = landmarks[NOSE_INDEX]
    chin = landmarks[CHIN_INDEX]
    left_eye_outer = landmarks[LEFT_EYE_LANDMARKS[0]]
    right_eye_outer = landmarks[RIGHT_EYE_LANDMARKS[1]]
    mouth_left = landmarks[MOUTH_LEFT_INDEX]
    mouth_right = landmarks[MOUTH_RIGHT_INDEX]

    nose_x, nose_y, nose_z = map(float, nose)
    chin_x, chin_y, chin_z = mapp(float, chin)
    left_eye_x, left_eye_y, left_eye_z = map(float, left_eye_outer)
    right_eye_x, right_eye_y, right_eye_z = map(float, right_eye_outer)
    mouth_left_x, mouth_left_y, mouth_left_z = map(float, mouth_left)
    mouth_right_x, mouth_right_y, mouth_right_z = map(float, mouth_right)

    eye_dx = right_eye_x - left_eye_x
    eye_dy = right_eye_y - left_eye_y
    eye_dz = right_eye_z - left_eye_z
    eye_dist_xy = max(np.hypot(eye_dx, eye_dy), 1e-6)
    eye_dist_xyz = max(float(np.sqrt((eye_dx ** 2) + (eye_dy ** 2) + (eye_dz ** 2))), 1e-6)

    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    eye_mid_y = (left_eye_y + right_eye_y) / 2.0
    eye_mid_z = (left_eye_z + right_eye_z) / 2.0

    mouth_mid_x = (mouth_left_x + mouth_right_x) / 2.0
    mouth_mid_y = (mouth_left_y + mouth_right_y) / 2.0
    mouth_mid_z = (mouth_left_z + mouth_right_z) / 2.0

    roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
    yaw = float(np.degrees(np.arctan2(nose_x - eye_mid_x, eye_dist_xy)))

    facial_axis_y = mouth_mid_y - eye_mid_y
    facial_axis_z = mouth_mid_z - eye_mid_z
    pitch = float(np.degrees(np.arctan2((nose_z - eye_mid_z), eye_dist_xyz)))

    if abs(facial_axis_y) > 1e-6:
        pitch += float(np.degrees(np.arctan2(nose_y - ((eye_mid_y + mouth_mid_y) / 2.0), abs(facial_axis_y))))

    return roll, pitch, yaw

def sample_to_runtime_compatible_base_features(sample: GazeSample) -> np.ndarray:
    eye_x, eye_y, eye_z = _best_eye_xyz(sample)
    gaze_x, gaze_y, gaze_z = _best_gaze_xyz(sample)
    head_x, head_y, head_z = _best_head_xyz(sample)
    headpose_roll, headpose_pitch, headpose_yaw = _estimate_head_pose_from_landmarks(sample)

    return np.asarray([
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

def gaze_window_to_runtime_compatible_sequence(
        window: GazeWindow,
        config: RuntimeCompatibleFeatureConfig,
) -> np.ndarray:
    vectors = [sample_to_runtime_compatible_base_features(sample) for sample in window.samples]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    sequence = np.stack(vectors, axis=0).astype(np.float32)
    if config.include_deltas:
        sequence = _append_deltas(sequence)
    return sequence.astype(np.float32)

def get_runtime_compatible_feature_names(config: RuntimeCompatibleFeatureConfig) -> list[str]:
    if not config.include_deltas:
        return list(BASE_RUNTIME_COMPATIBLE_FEATURE_NAMES)
    return list(ALL_RUNTIME_COMPATIBLE_FEATURE_NAMES)

def get_runtime_compatible_feature_dim(config: RuntimeCompatibleFeatureConfig) -> int:
    return len(get_runtime_compatible_feature_names(config))