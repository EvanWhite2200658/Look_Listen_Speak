# src/turn_prediction/features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.gaze.schemas import GazeSample
from schemas import GazeWindow

@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for gaze feature extraction

    These switches keep the extraction logic stable while changing
    the exact input representation used for experiments.
    """

    include_raw_xy: bool = True
    include_calibrated_xy: bool = True
    include_filtered_xy: bool = True
    include_eye_openness: bool = True
    include_feature_vector: bool = True
    include_deltas: bool = True
    max_feature_vector_length: Optional[int] = None


def _xy_or_default(value: Optional[tuple[float, float]]) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    return float(value[0]), float(value[1])

def _float_or_default(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)

def _tracking_state_or_default(value: Optional[int]) -> float:
    if value is None:
        return 0.0
    return float(value)

def _status_or_default(value: Optional[bool]) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else 0.0

def sample_to_feature_vector(
        sample: GazeSample,
        config: FeatureConfig,
) -> np.ndarray:
    """
    Convert one GazeSample into a 1D numeric feature vector.
    :param sample:
    :param config:
    :return:
    """
    features: List[float] = []

    if config.include_raw_xy:
        raw_xy, raw_xy = _xy_or_default(sample.raw_xy)
        features.extend([raw_xy, raw_xy])

    if config.include_calibrated_xy:
        cal_x, cal_y = _xy_or_default(sample.calibrated_xy)
        features.extend([cal_x, cal_y])

    if config.include_filtered_xy:
        fil_x, fil_y = _xy_or_default(sample.filtered_xy)
        features.extend([fil_x, fil_y])

    if config.include_eye_openness:
        features.extend(
            [
                _float_or_default(sample.left_eye_openness),
                _float_or_default(sample.right_eye_openness),
            ]
        )

    # add tracking metadata as numeric signals
    features.extend(
        [
            _float_or_default(sample.tracking_state),
            _status_or_default(sample.status),
        ]
    )

    if config.include_feature_vector:
        raw_features = sample.features
        if config.max_feature_vector_length is not None:
            raw_features = raw_features[:config.max_feature_vector_length]
        features.extend(float(x) for x in raw_features)

    return np.asarray(features, dtype=np.float32)


def _pad_vectors(vectors: List[np.ndarray]) -> np.ndarray:
    """
    pad variable-length vectors to a consistent feature dimension.

    output shape:
        (seq_len, max_feature_dim)
    :param vectors:
    :return:
    """
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    max_dim = max(vec.shape[0] for vec in vectors)
    output = np.zeros((len(vectors), max_dim), dtype=np.float32)

    for i, vec in enumerate(vectors):
        output[i, :vec.shape[0]] = vec

    return output


def _append_deltas(sequence: np.ndarray) -> np.ndarray:
    """
    append first-order frame-to-frame deltas to each timestep
    if sequence has shape (T, D), output has shape (T, 2D).
    the first timestep delta is zero
    :param sequence:
    :return:
    """
    if sequence.size == 0:
        return sequence

    deltas = np.zeros_like(sequence, dtype=np.float32)
    if sequence.shape[0] > 1:
        deltas[1:] = sequence[1:] - sequence[:-1]

    return np.concatenate([sequence, deltas], axis=1)


def gaze_window_to_sequence(
        window: GazeWindow,
        config: FeatureConfig,
) -> np.ndarray:
    """
    convert a GazeWindow into a 2D sequence tensor for model input.
    output shape:
        (seq_len, feature_dim)
    :param window:
    :param config:
    :return:
    """
    per_frame_vectors = [
        sample_to_feature_vector(sample, config)
        for sample in window.samples
    ]

    sequence = _pad_vectors(per_frame_vectors)

    if config.include_deltas:
        sequence = _append_deltas(sequence)

    return sequence.astype(np.float32)


def get_feature_dim(
        example_window: GazeWindow,
        config: FeatureConfig,
) -> int:
    """
    utility to inspect the final feature dimension for a given config.
    :param example_window:
    :param config:
    :return:
    """
    sequence = gaze_window_to_sequence(example_window, config)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        return 0
    return int(sequence.shape[1])