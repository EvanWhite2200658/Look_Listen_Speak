from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from deprecated_files.live_dataset import LiveSequenceDataset, split_live_dataset
from deprecated_files.live_model_config import (
    DEFAULT_LIVE_THRESHOLD,
    build_default_live_feature_config,
    build_live_transformer_config,
    serialize_live_feature_config,
)
from src.turn_prediction.model import TurnShiftTransformer


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        p_t = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        return (alpha_t * focal_weight * bce_loss).mean()


@dataclass(frozen=True)
class LiveTrainingConfig:
    data_path: str
    output_dir: str = "artifacts/live_turn_prediction"
    window_size: int = 30
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_dataloaders(dataset: LiveSequenceDataset, config: LiveTrainingConfig) -> tuple[DataLoader, DataLoader]:
    train_subset, val_subset = split_live_dataset(
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



def run_epoch(
    model: TurnShiftTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    losses: list[float] = []
    all_probs: list[float] = []
    all_labels: list[float] = []

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

        probs = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(batch_y.detach().cpu().numpy().tolist())

    if not losses:
        return 0.0, 0.0, 0.0, 0.0

    preds = (np.asarray(all_probs) >= DEFAULT_LIVE_THRESHOLD).astype(np.float32)
    labels = np.asarray(all_labels)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return float(np.mean(losses)), precision, recall, f1



def save_live_checkpoint(
    model: TurnShiftTransformer,
    training_config: LiveTrainingConfig,
    output_dir: Path,
    epoch: int,
    val_loss: float,
    best_val_f1: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer_config = build_live_transformer_config(window_size=training_config.window_size)
    live_feature_config = build_default_live_feature_config()

    checkpoint_path = output_dir / "best_live_model.pt"
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "best_threshold": DEFAULT_LIVE_THRESHOLD,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(transformer_config),
        "training_config": asdict(training_config),
        "live_feature_config": serialize_live_feature_config(live_feature_config),
        "feature_space": "live_native_v1",
    }
    torch.save(payload, checkpoint_path)

    metadata_path = output_dir / "best_live_model_metadata.json"
    metadata_path.write_text(json.dumps({
        "epoch": epoch,
        "val_loss": val_loss,
        "best_threshold": DEFAULT_LIVE_THRESHOLD,
        "best_val_f1": best_val_f1,
        "model_config": asdict(transformer_config),
        "training_config": asdict(training_config),
        "live_feature_config": serialize_live_feature_config(live_feature_config),
        "feature_space": "live_native_v1",
    }, indent=2))

    return checkpoint_path



def train_live_model(config: LiveTrainingConfig) -> Path:
    set_seed(config.random_seed)

    dataset = LiveSequenceDataset(config.data_path)
    metadata = dataset.metadata

    transformer_config = build_live_transformer_config(window_size=config.window_size)
    if metadata.seq_len != transformer_config.max_seq_len:
        raise ValueError(
            "Dataset sequence length does not match configured window size. "
            f"dataset_seq_len={metadata.seq_len}, model_seq_len={transformer_config.max_seq_len}"
        )
    if metadata.feature_dim != transformer_config.input_dim:
        raise ValueError(
            "Dataset feature dimension does not match live feature schema. "
            f"dataset_feature_dim={metadata.feature_dim}, expected_feature_dim={transformer_config.input_dim}"
        )

    train_loader, val_loader = build_dataloaders(dataset, config)
    device = torch.device(config.device)

    model = TurnShiftTransformer(transformer_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_f1 = -1.0
    best_path: Path | None = None

    print("Live dataset metadata:")
    print(metadata)
    print("\nTraining live-native turn model...")

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_precision, train_recall, train_f1 = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_precision, val_recall, val_f1 = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            f"epoch={epoch:02d} | "
            f"train_loss={train_loss:.4f} | train_p={train_precision:.4f} | train_r={train_recall:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_p={val_precision:.4f} | val_r={val_recall:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = save_live_checkpoint(
                model=model,
                training_config=config,
                output_dir=Path(config.output_dir),
                epoch=epoch,
                val_loss=val_loss,
                best_val_f1=val_f1,
            )

    if best_path is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    print(f"\nSaved best live-native checkpoint to: {best_path}")
    return best_path


if __name__ == "__main__":
    default_config = LiveTrainingConfig(data_path="data/live_sequences/live_sequences.npz")
    train_live_model(default_config)