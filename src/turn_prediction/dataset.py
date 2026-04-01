# src/turn_prediction/dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    """
    configuration for building turn-prediction training windows
    """

    feature_columns: Sequence[str]
    label_column: str = "turn_shift_in_next_15_frames"
    window_size: int = 30
    stride: int = 1
    drop_missing_labels: bool = True
    fillna_value: float = 0.0
    positive_label_value: int = 1


@dataclass(frozen=True)
class WindowedSample:
    """
    one fixed-length training sample

    attributes:
        features:
            Shape (window_size, feature_dim)
        label:
            binary label for the final frame in the window
        end_frame_index:
            frameIndex of the final frame in the window
        sequence_id:
            original conversation / recording id for traceability
    """

    features: np.ndarray
    label: float
    end_frame_index: int
    sequence_id: str


def load_turn_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """
    Load one processed CSV file for turn prediction
    :param csv_path:
    :return:
    """
    df = pd.read_csv(csv_path)

    required_base_columns = {"id", "frameIndex"}
    missing = [col for col in required_base_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns {missing}")

    return df


def validate_dataset_columns(
        df: pd.DataFrame,
        config: DatasetConfig,
) -> None:
    """
    Ensure required feature and label columns are present.
    :param df:
    :param config:
    :return:
    """
    required_columns = ["id", "frameIndex", *config.feature_columns, config.label_column]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns {missing_columns}")


def prepare_turn_dataframe(
        df: pd.DataFrame,
        config: DatasetConfig,
) -> pd.DataFrame:
    """
    clean and prepare the dataframe for window extraction
    :param df:
    :param config:
    :return:
    """
    # validate required columns
    validate_dataset_columns(df, config)

    # sort by sequence id and frame index
    output = df.copy()
    output = output.sort_values(["id", "frameIndex"]).reset_index(drop=True)

    # optionally drop rows with missing labels
    if config.drop_missing_labels:
        output = output[output[config.label_column].notna()].copy()

    # fill missing feature values
    output[list(config.feature_columns)] = output[list(config.feature_columns)].fillna(
        config.fillna_value
    )

    # fill missing labels with 0 if not dropped
    output[config.label_column] = output[config.label_column].fillna(0)

    return output


def _build_windowed_sampled_for_group(
        group_df: pd.DataFrame,
        config: DatasetConfig,
) -> List[WindowedSample]:
    """
    build fixed-length sliding windows for one sequence id only

    important: windows must never cross sequence boundaries
    :param group_df:
    :param config:
    :return:
    """
    feature_matrix = group_df[list(config.feature_columns)].to_numpy(dtype=np.float32)
    labels = group_df[config.label_column].to_numpy()
    frame_indices = group_df["frameIndex"].to_numpy()
    sequence_ids = group_df["id"].astype(str).to_numpy()

    windowed_samples: List[WindowedSample] = []

    total_frames = len(group_df)
    window_size = config.window_size
    stride = config.stride

    if total_frames < window_size:
        return windowed_samples

    for start_idx in range(0, total_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_features = feature_matrix[start_idx:end:idx]

        end_label_raw = labels[end_idx -1]
        end_label = float(int(end_label_raw == config.positive_label_value))

        end_frame_index = int(frame_indices[end_idx-1])
        sequence_id = str(sequence_ids[end_idx -1])

        windowed_samples.append(
            WindowedSample(
                features=window_features,
                label=end_label,
                end_frame_index=end_frame_index,
                sequence_id=sequence_id,
            )
        )

    return windowed_samples


def build_windowed_samples(
        df: pd.DataFrame,
        config: DatasetConfig,
) -> List[WindowedSample]:
    """
    convert a processed frame-level dataframe into fixed-length windowed samples

    label rule:
    - each window uses the label value of the final frame in the window
    :param df:
    :param config:
    :return:
    """
    prepared_df = prepare_turn_dataframe(df, config)

    all_samples: List[WindowedSample] = []

    for _, group_df in prepared_df.groupby("id", sort=False):
        group_samples = _build_windowed_sampled_for_group(group_df, config)
        all_samples.extend(group_samples)

    return all_samples


def stack_samples(
        samples: Sequence[WindowedSample],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack windowed samples into arrays
    :param samples:
    :return: X: shape (num_samples, window_size, feature_dim)
    y: shape (num_samples,)
    """

    if not samples:
        return (
            np.zeros((0, 0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    x = np.stack([sample.features for sample in samples]).astype(np.float32)
    y = np.asarray([sample.label for sample in samples], dtype=np.float32)

    return x, y

class TurnPredictionDataset(Dataset):
    """
    PyTorch Dataset for turn-transition prediction
    """

    def __init__(self, samples: Sequence[WindowedSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[index]

        x = torch.tensor(sample.features, dtype=torch.float32)
        y = torch.tensor(sample.label, dtype=torch.float32)

        return x, y


def build_dataset_from_csv(
        csv_path: str | Path,
        config: DatasetConfig,
) -> TurnPredictionDataset:
    """
    build a PyTorch dataset from one processed CSV file
    :param csv_path:
    :param config:
    :return:
    """
    df = load_turn_dataframe(csv_path)
    samples = build_windowed_samples(df, config)
    return TurnPredictionDataset(samples)


def build_dataset_from_multiple_csvs(
        csv_paths: Iterable[str | Path],
        config: DatasetConfig,
) -> TurnPredictionDataset:
    """
    build a combined PyTorch dataset from multiple processed CSV files
    :param csv_paths:
    :param config:
    :return:
    """
    all_samples: List[WindowedSample] = []

    for csv_path in csv_paths:
        df = load_turn_dataframe(csv_path)
        samples = build_windowed_samples(df, config)
        all_samples.extend(samples)

    return TurnPredictionDataset(all_samples)