# src/turn_prediction/overlap_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from overlap_columns import ALL_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class OverlapWindowedSample:
    features: np.ndarray
    label: float
    end_frame_index: int
    sequence_id: str


@dataclass(frozen=True)
class OverlapDatasetMetadata:
    num_samples: int
    seq_len: int
    feature_dim: int
    num_sequences: int
    positive_count: int
    negative_count: int


class OverlapSequenceDataset(Dataset):
    """
    Builds windowed training samples from one or more processed CSV files
    using the deployment-safe overlap feature schema.

    Important:
    - supports multiple CSV files
    - preserves sequence ids
    - never allows windows to cross sequence boundaries
    """

    def __init__(
        self,
        data_path: str | Path,
        window_size: int = 30,
        stride: int = 5,
        positive_label_value: int = 1,
        fillna_value: float = 0.0,
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.positive_label_value = int(positive_label_value)
        self.fillna_value = float(fillna_value)

        csv_paths = self._collect_csv_paths(data_path)
        self._samples = self._build_samples(csv_paths)

        if not self._samples:
            raise ValueError("No overlap training samples were generated from the provided CSV files.")

        self.X = np.stack([sample.features for sample in self._samples]).astype(np.float32)
        self.y = np.asarray([sample.label for sample in self._samples], dtype=np.float32)
        self.sequence_ids = np.asarray([sample.sequence_id for sample in self._samples], dtype=str)

    @staticmethod
    def _collect_csv_paths(input_path: str | Path) -> list[Path]:
        path = Path(input_path)

        if path.is_file():
            return [path]

        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in {path}")
            return csv_files

        raise ValueError(f"Invalid path: {input_path}")

    def _load_and_validate_csv(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        required = ["id", "frameIndex", *ALL_COLUMNS, TARGET_COLUMN]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

        df = df.copy()
        df = df.sort_values(["id", "frameIndex"]).reset_index(drop=True)
        df[ALL_COLUMNS] = df[ALL_COLUMNS].fillna(self.fillna_value)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0)
        return df

    def _build_samples(self, csv_paths: Sequence[Path]) -> list[OverlapWindowedSample]:
        samples: list[OverlapWindowedSample] = []

        for csv_path in csv_paths:
            df = self._load_and_validate_csv(csv_path)

            for _, group_df in df.groupby("id", sort=False):
                feature_matrix = group_df[ALL_COLUMNS].to_numpy(dtype=np.float32)
                labels = group_df[TARGET_COLUMN].to_numpy()
                frame_indices = group_df["frameIndex"].to_numpy()
                sequence_ids = group_df["id"].astype(str).to_numpy()

                total_frames = len(group_df)
                if total_frames < self.window_size:
                    continue

                for start_idx in range(0, total_frames - self.window_size + 1, self.stride):
                    end_idx = start_idx + self.window_size
                    window_features = feature_matrix[start_idx:end_idx]

                    end_label_raw = labels[end_idx - 1]
                    end_label = float(int(end_label_raw == self.positive_label_value))
                    end_frame_index = int(frame_indices[end_idx - 1])
                    sequence_id = str(sequence_ids[end_idx - 1])

                    samples.append(
                        OverlapWindowedSample(
                            features=window_features,
                            label=end_label,
                            end_frame_index=end_frame_index,
                            sequence_id=sequence_id,
                        )
                    )

        return samples

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

    @property
    def metadata(self) -> OverlapDatasetMetadata:
        positives = int(np.sum(self.y == 1))
        negatives = int(np.sum(self.y == 0))
        return OverlapDatasetMetadata(
            num_samples=int(self.X.shape[0]),
            seq_len=int(self.X.shape[1]),
            feature_dim=int(self.X.shape[2]),
            num_sequences=int(len(np.unique(self.sequence_ids))),
            positive_count=positives,
            negative_count=negatives,
        )