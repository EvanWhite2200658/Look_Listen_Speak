# src/turn_prediction/train.py

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .columns import GAZE_TRANSFORMER_FEATURE_COLUMNS, MAIN_TARGET_COLUMN
from .dataset import DatasetConfig, TurnPredictionDataset, build_dataset_from_multiple_csvs
from .model import TransformerConfig, TurnShiftTransformer


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training configuration for turn-transition prediction
    """

    csv_paths: Sequence[str]
    output_dir: str = "artifacts/turn_prediction"
    window_size: int = 30
    stride: int = 5
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    positive_class_weight: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
        dataset: TurnPredictionDataset,
        validation_split: float,
        seed: int,
) -> tuple[Subset, Subset]:
    """
    Random train/validation split.
    :param dataset:
    :param validation_split:
    :param seed:
    :return:
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(dataset_size * validation_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def build_dataloaders(
        dataset: TurnPredictionDataset,
        config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    train_subset, val_subset = split_dataset(
        dataset=dataset,
        validation_split=config.validation_split,
        seed=config.random_seed,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader


def compute_accuracy_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float((preds == labels).float().mean().item())


def run_epoch(
        model: TurnShiftTransformer,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """
    Run one train or validation epoch.
    :param model:
    :param dataloader:
    :param criterion:
    :param device:
    :param optimizer:
    :return: mean_loss, mean_accuracy
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if is_training:
            loss.backward()
            optimizer.step()

        acc = compute_accuracy_from_logits(logits, batch_y)

        total_loss += float(loss.item())
        total_acc += acc
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0

    return total_loss / total_batches, total_acc / total_batches


def save_checkpoint(
        model: TurnShiftTransformer,
        model_config: TransformerConfig,
        training_config: TrainingConfig,
        output_dir: Path,
        epoch: int,
        val_loss: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pt"
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
    }
    torch.save(payload, checkpoint_path)

    metadata_path = output_dir / "best_model_metadata.json"
    metadata = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return checkpoint_path


def train_model(training_config: TrainingConfig) -> Path:
    """
    Full training pipeline.
    - load data
    - build dataset
    - create model
    - train / validate
    - save best checkpoint
    :param training_config:
    :return:
    """
    set_seed(training_config.random_seed)

    dataset_config = DatasetConfig(
        feature_columns=GAZE_TRANSFORMER_FEATURE_COLUMNS,
        label_column=MAIN_TARGET_COLUMN,
        window_size=training_config.window_size,
        stride=training_config.stride,
        drop_missing_labels=True,
        fillna_value=0.0,
        positive_label_value=1,
    )

    dataset = build_dataset_from_multiple_csvs(
        csv_paths=training_config.csv_paths,
        config=dataset_config,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check CSV paths and preprocessing.")

    train_loader, val_loader = build_dataloaders(dataset, training_config)

    input_dim = len(GAZE_TRANSFORMER_FEATURE_COLUMNS)
    model_config = TransformerConfig(
        input_dim=input_dim,
        max_seq_len=training_config.window_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_learned_positional_encoding=True,
    )

    device = torch.device(training_config.device)
    model = TurnShiftTransformer(model_config).to(device)

    if training_config.positive_class_weight is not None:
        pos_weight = torch.tensor(
            [training_config.positive_class_weight],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_checkpoint_path: Path | None = None

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Input dim: {input_dim}")

    for epoch in range(1, training_config.num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d}/{training_config.num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model=model,
                model_config=model_config,
                training_config=training_config,
                output_dir=output_dir,
                epoch=epoch,
                val_loss=val_loss,
            )
            print(f"  Saved new best checkpoint to: {best_checkpoint_path}")

    if best_checkpoint_path is None:
        raise RuntimeError("Training completed but no checkpoint was saved.")

    return best_checkpoint_path


def main() -> None:
    config= TrainingConfig(
        csv_paths=[
            "data/processed/example_turn_data.csv",
        ],
        output_dir="artifacts/turn_prediction",
        window_size=30,
        stride=5,
        batch_size=64,
        num_epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        validation_split=0.2,
        random_seed=42,
        num_workers=0,
        positive_class_weight=None,
    )

    checkpoint_path = train_model(config)
    print(f"\nBest checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()