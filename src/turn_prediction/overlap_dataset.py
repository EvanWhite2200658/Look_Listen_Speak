# src/turn_prediction/overlap_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from overlap_columns import ALL_COLUMNS, TARGET_COLUMN

@dataclass(frozen=True)
class OverlapDatasetMetadata:
    num_samples: int
    seq_len: int
    feature_dim: int


class OverlapSequenceDataset(Dataset):
    """
    Builds sequences directly from the EXISTING processed dataset.
    """

    def __init__(
            self,
            data_path: str | Path,
            window_size: int = 30,
            stride: int = 5,
    ) -> None:
        df = pd.read_csv(data_path)

        missing = [c for c in ALL_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        X_list = []
        y_list = []

        for i in range(0, len(df) - window_size, stride):
            window = df.iloc[i : i + window_size]

            X = window[ALL_COLUMNS].values.astype(np.float32)
            y = window[TARGET_COLUMN].iloc[-1]

            X_list.append(X)
            y_list.append(float(y))

        self.X = np.stack(X_list)
        self.y = np.array(y_list, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
        )

    @property
    def metadata(self):
        return OverlapDatasetMetadata(
            num_samples=self.X.shape[0],
            seq_len=self.X.shape[1],
            feature_dim=self.X.shape[2],
        )