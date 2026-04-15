# src/turn_prediction/live_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class LiveSequenceDatasetMetadata:
    num_samples: int
    seq_len: int
    feature_dim: int
    positive_count: int
    negative_count: int
    has_sequence_ids: bool


class LiveSequenceDataset(Dataset):
    """
    Dataset for pre-extracted live-native turn-prediction sequences.

    Expected NPZ keys:
    - X: float32 array of shape (N, T, D)
    - y: float32/int array of shape (N,)

    Optional NPZ keys:
    - sequence_ids: array of shape (N,) for sequence-aware splitting.
    """
    def __init_(self, npz_path: str | Path) -> None:
        self._path = Path(npz_path)
        data = np.load(self._path, allow_pickle=True)

        if "X" not in data or "y" not in data:
            raise ValueError(
                "LiveSequenceDataset requires NPZ keys 'X' and 'y'."
            )

        self.X = np.asarray(data["X"], dtype=np.float32)
        self.y = np.asarray(data["y"], dtype=np.float32)

        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (N, T, D). Go {self.X.shape}.")
        if self.y.ndim != 1:
            raise ValueError(f"y must have shape (N,). GOt {self.y.shape}.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples."
                f"Got X={self.X.shape[0]} and y={self.y.shape[0]}."
            )

        if "sequence_ids" in data:
            self.sequence_ids = np.asarray(data["sequence_ids"]).astype(str)
            if self.sequence_ids.shape[0] != self.X.shape[0]:
                raise ValueError(
                    "sequence_ids must match number of samples. "
                    f"Got sequence_ids={self.sequence_ids.shape[0]} and X={self.X.shape[0]}."
                )
        else:
            self.sequence_ids = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(self.X[index], dtype=torch.float32)
        y = torch.Tensor(self.y[index], dtype=torch.float32)
        return x, y

    @property
    def metadata(self) -> LiveSequenceDatasetMetadata:
        positives = int(np.sum(self.y == 1))
        negatives = int(np.sum(self.y == 0))
        return LiveSequenceDatasetMetadata(
            num_samples=int(self.X.shape[0]),
            seq_len=int(self.X.shape[1]),
            feature_dim=int(self.X.shape[2]),
            positive_count=positives,
            negative_count=negatives,
            has_sequence_ids=(self.sequence_ids is not None),
        )


def split_live_dataset(
        dataset: LiveSequenceDataset,
        validation_split: float,
        seed: int,
) -> tuple[Subset, Subset]:
    rng = np.random.default_rng(seed)

    if dataset.sequence_ids is not None:
        unique_ids = np.unique(dataset.sequence_ids)
        shuffled_ids = unique_ids.copy()
        rng.shuffle(shuffled_ids)

        val_size = max(1, int(len(shuffled_ids) * validation_split))
        val_ids = set(shuffled_ids[:val_size])

        train_indices: list[int] = []
        val_indices: list[int] = []

        for idx, seq_id in enumerate(dataset.sequence_ids):
            if seq_id in val_ids:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    else:
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        val_size = max(1, int(len(indices) * validation_split))
        val_indices = indices[:val_size].tolist()
        train_indices = indices[val_size:].tolist()

    return Subset(dataset, train_indices), Subset(dataset, val_indices)