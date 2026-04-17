# src/turn_prediction/live_features.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.schemas import FaceSample, GazeSample
from schemas import GazeWindow


@dataclass(frozen=True)
class LiveFeatureConfig:
    """
    Interpretable live-native feature configuration

    This first live feature set is deliberately gaze-centred and avoids:
    - undocumented latent vectors from GazeFollower
    - speaking/activity context
    - target tracking/context features
    """

    include_raw_xy: bool = True
    include_calibrated_xy: bool = True
    include_filtered_xy: bool = True
    include_eye_openness: bool = True
    include_face_geometry: bool = True
    include_eye_geometry: bool = True
    include_tracking_metadata: bool = True
    include_deltas: bool = True


@dataclass(frozen=True)
class LiveFeatureMetadata:
    base_feature_names: list[str]
    feature_names: list[str]


def _xy_or_default(value: Optional[tuple[float, float]]) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    return float(value[0]), float(value[1])

def _rect_center(rect: Optional[tuple[float, float, float, float]]) -> tuple[float, float]:
    if rect is None:
        return 0.0, 0.0

    x, y, w, h = rect
    return float(x + (w / 2.0)), float(y + (h / 2.0))

def _rect_size(rect: Optional[tuple[float, float, float, float]]) -> tuple[float, float]:
    if rect is None:
        return 0.0, 0.0

    _, _, w, h = rect
    return float(w), float(h)

def _float_or_default(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)

def _bool_or_default(value: Optional[bool]) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else 0.0

def _distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _face_or_empty(face: Optional[FaceSample]) -> FaceSample:
    if face is not None:
        return face
    return FaceSample()



def get_live_base_feature_names(config: LiveFeatureConfig) -> list[str]:
    names: list[str] = []

    if config.include_raw_xy:
        names.extend([
            "raw_gaze_x",
            "raw_gaze_y",
        ])

    if config.include_calibrated_xy:
        names.extend([
            "calibrated_gaze_x",
            "calibrated_gaze_y",
        ])

    if config.include_filtered_xy:
        names.extend([
            "filtered_gaze_x",
            "filtered_gaze_y",
        ])

    if config.include_eye_openness:
        names.extend([
            "left_eye_openness",
            "right_eye_openness",
        ])

    if config.include_face_geometry:
        face_center_x, face_center_y = _rect_center(face.face_rect)
        
        names.extend([
            "face_center_x",
            "face_center_y",
            "face_width",
            "face_height",
        ])

    if config.include_eye_geometry:
        names.extend([
            "left_eye_center_x",
            "left_eye_center_y",
            "right_eye_center_x",
            "right_eye_center_y",
            "inter_eye_distance",
            "landmark_count",
        ])

    if config.include_tracking_metadata:
        names.extend([
            "tracking_state",
            "status",
            "event",
            "can_gaze_estimation",
        ])

    return names


def get_live_feature_names(config: LiveFeatureConfig) -> list[str]:
    base_names = get_live_base_feature_names(config)
    if not config.include_deltas:
        return list(base_names)

    delta_names = [f"delta_{name}" for name in base_names]
    return [*base_names, *delta_names]

def get_live_feature_metadata(config: LiveFeatureConfig) -> LiveFeatureMetadata:
    return LiveFeatureMetadata(
        base_feature_names=get_live_base_feature_names(config),
        feature_names=get_live_feature_names(config),
    )


def sample_to_live_base_feature_vector(
        sample: GazeSample,
        config: LiveFeatureConfig,
) -> np.ndarray:
    features: list[float] = []
    face = _face_or_empty(sample.face)

    if config.include_raw_xy:
        raw_x, raw_y = _xy_or_default(sample.raw_xy)
        features.extend([raw_x, raw_y])

    if config.include_calibrated_xy:
        cal_x, cal_y = _xy_or_default(sample.calibrated_xy)
        features.extend([cal_x, cal_y])

    if config.include_filtered_xy:
        fil_x, fil_y = _xy_or_default(sample.filtered_xy)
        features.extend([fil_x, fil_y])

    if config.include_eye_openness:
        features.extend([
            _float_or_default(sample.left_eye_openness),
            _float_or_default(sample.right_eye_openness),
        ])

    if config.include_face_geometry:
        face_center_x, face_center_x = _rect_center(face.face_rect)
        face_width, face_height = _rect_size(face.face_rect)
        features.extend([
            face_center_x,
            face_center_x,
            face_width,
            face_height,
        ])

    if config.include_eye_geometry:
        left_eye_center_x, left_eye_center_y = _rect_center(face.left_rect)
        right_eye_center_x, right_eye_center_y = _rect_center(face.right_rect)
        inter_eye_distance = _distance_2d(
            (left_eye_center_x, left_eye_center_y),
            (right_eye_center_x, right_eye_center_y),
        )
        landmark_count = float(face.landmark_count)
        features.extend([
            left_eye_center_x,
            left_eye_center_y,
            right_eye_center_x,
            right_eye_center_y,
            inter_eye_distance,
            landmark_count,
        ])

    if config.include_tracking_metadata:
        tracking_state = 0.0 if sample.tracking_state is None else float(sample.tracking_state)
        event = 0.0 if sample.event is None else float(sample.event)
        features.extend([
            tracking_state,
            _bool_or_default(sample.status),
            event,
            _bool_or_default(face.can_gaze_estimation),
        ])

    return np.asarray(features, dtype=np.float32)


def _append_deltas(sequence: np.ndarray) -> np.ndarray:
    if sequence.size == 0:
        return sequence

    deltas = np.zeros_like(sequence)
    if sequence.shape[0] > 1:
        deltas[1:] = sequence[1:] - sequence[:-1]

    return np.concatenate([sequence, deltas], axis=1)


def gaze_window_to_live_sequence(
        window: GazeWindow,
        config: LiveFeatureConfig,
) -> np.ndarray:
    vectors = [sample_to_live_base_feature_vector(sample, config) for sample in window.samples]

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    sequence = np.stack(vectors, axis=0).astype(np.float32)

    if config.include_deltas:
        sequence = _append_deltas(sequence)

    return sequence

def get_live_feature_dim(config: LiveFeatureConfig) -> int:
    return len(get_live_feature_names(config))