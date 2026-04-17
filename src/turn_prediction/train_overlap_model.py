# src/turn_prediction/train_overlap_model.py

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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


from model import TransformerConfig, TurnShiftTransformer
from overlap_dataset import OverlapSequenceDataset


@dataclass(frozen=True)
class OverlapTrainingConfig:
    data_path: str
    output_dir: str = "artifacts/overlap_model"
    window_size: int = 30
    stride: int = 5
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    random_seed: int = 42
    positive_class_weight: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transformer_config(window_size: int = 30) -> TransformerConfig:
    return TransformerConfig(
        input_dim=24,
        max_seq_len=window_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_learned_positional_encoding=True,
    )


def split_dataset_by_sequence(
    dataset: OverlapSequenceDataset,
    validation_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    seq_to_indices: dict[str, list[int]] = {}

    for idx, sample in enumerate(dataset._samples):
        seq_id = sample.sequence_id
        seq_to_indices.setdefault(seq_id, []).append(idx)

    sequence_ids = list(seq_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(sequence_ids)

    val_size = max(1, int(len(sequence_ids) * validation_split))
    val_seq_ids = set(sequence_ids[:val_size])

    train_indices: list[int] = []
    val_indices: list[int] = []

    for seq_id, indices in seq_to_indices.items():
        if seq_id in val_seq_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_dataloaders(dataset: OverlapSequenceDataset, config: OverlapTrainingConfig) -> tuple[DataLoader, DataLoader]:
    train_subset, val_subset = split_dataset_by_sequence(
        dataset=dataset,
        validation_split=config.validation_split,
        seed=config.random_seed,
    )

    summarize_subset_labels(dataset, train_subset, "TRAIN")
    summarize_subset_labels(dataset, val_subset, "VALIDATION")

    # build class-balanced sampler
    train_labels = [dataset.y[idx] for idx in train_subset.indices]

    class_counts = np.bincount(np.array(train_labels, dtype=int))
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[int(label)] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        sampler=sampler,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
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

    preds = (np.asarray(all_probs) >= 0.2).astype(np.float32)
    labels = np.asarray(all_labels)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return float(np.mean(losses)), precision, recall, f1

def evaluate_thresholds(
        model: TurnShiftTransformer,
        dataloader: DataLoader,
        device: torch.device,
        thresholds: list[float] | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    results = []

    print("\nValidation threshold sweep:")
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(np.float32)

        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        predicted_positives = int(preds.sum())

        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_positives": predicted_positives,
        }
        results.append(result)

        print(
            f"threshold={threshold:.2f} | "
            f"precision={precision:.4f} | "
            f"recall={recall:.4f} | "
            f"f1={f1:.4f} | "
            f"predicted_positives={predicted_positives}"
        )

    return results

def save_checkpoint(
    model: TurnShiftTransformer,
    config: OverlapTrainingConfig,
    output_dir: Path,
    epoch: int,
    val_loss: float,
    best_val_f1: float,
    best_threshold: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = build_transformer_config(window_size=config.window_size)

    checkpoint_path = output_dir / "best_model.pt"
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "best_threshold": best_threshold,
        "best_val_f1": best_val_f1,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "feature_space": "overlap_v1",
        "feature_dim": 24,
        "window_size": config.window_size,
    }
    torch.save(payload, checkpoint_path)

    metadata_path = output_dir / "best_model_metadata.json"
    metadata_path.write_text(json.dumps({
        "epoch": epoch,
        "val_loss": val_loss,
        "best_threshold": 0.2,
        "best_val_f1": best_val_f1,
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "feature_space": "overlap_v1",
        "feature_dim": 24,
        "window_size": config.window_size,
    }, indent=2))

    return checkpoint_path

def summarize_subset_labels(dataset, subset, name):
    import numpy as np

    labels = [dataset.y[idx] for idx in subset.indices]

    positives = int(np.sum(np.array(labels) == 1))
    negatives = int(np.sum(np.array(labels) == 0))
    total = len(labels)

    print(f"\n{name} label summary:")
    print(f"Total: {total}")
    print(f"Positives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"Positive rate: {positives / total:.6f}")

def train(config: OverlapTrainingConfig) -> Path:
    set_seed(config.random_seed)

    dataset = OverlapSequenceDataset(
        data_path=config.data_path,
        window_size=config.window_size,
        stride=config.stride,
    )

    metadata = dataset.metadata
    print("Overlap dataset metadata:")
    print(metadata)

    if metadata.feature_dim != 24:
        raise ValueError(f"Expected overlap feature dim 24, got {metadata.feature_dim}")
    if metadata.seq_len != config.window_size:
        raise ValueError(f"Expected sequence length {config.window_size}, got {metadata.seq_len}")

    train_loader, val_loader = build_dataloaders(dataset, config)
    device = torch.device(config.device)

    model_config = build_transformer_config(window_size=config.window_size)
    model = TurnShiftTransformer(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_f1 = -1.0
    best_path: Path | None = None

    print("\nTraining overlap-feature turn model...")
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

        threshold_results = evaluate_thresholds(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        best_threshold_result = max(threshold_results, key=lambda f: f["f1"])
        best_threshold = float(best_threshold_result["threshold"])
        best_epoch_val_f1 = float(best_threshold_result["f1"])

        print(
            f"epoch={epoch:02d} | "
            f"train_loss={train_loss:.4f} | train_p={train_precision:.4f} | train_r={train_recall:.4f} | train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | val_p={val_precision:.4f} | val_r={val_recall:.4f} | val_f1={val_f1:.4f}"
        )

        if best_epoch_val_f1 > best_val_f1:
            best_val_f1 = best_epoch_val_f1
            best_path = save_checkpoint(
                model=model,
                config=config,
                output_dir=Path(config.output_dir),
                epoch=epoch,
                val_loss=val_loss,
                best_val_f1=best_epoch_val_f1,
                best_threshold=best_threshold,
            )

    if best_path is None:
        raise RuntimeError("Training completed without saving a checkpoint.")

    print(f"\nSaved best overlap checkpoint to: {best_path}")
    return best_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed"
    output_dir = Path(__file__).resolve().parent / "artifacts" / "overlap_model"

    cfg = OverlapTrainingConfig(
        data_path=str(data_path),
        output_dir=str(output_dir),
    )
    train(cfg)