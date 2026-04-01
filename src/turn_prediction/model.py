# src/turn_prediction/model.py

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

@dataclass(frozen=True)
class TransformerConfig:
    """
    configuration for the gaze-based turn-transition Transformer.
    """

    input_dim: int
    max_seq_len: int = 30
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    use_learned_positional_encoding: bool = True


class PositionalEncoding(nn.Module):
    """
    standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0)) # shape: (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (B, T, D)
        :return:
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """
    learnable positional encoding
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (B, T, D)
        :return:
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.embedding(positions)


class TurnShiftTransformer(nn.Module):
    """
    Small Transformer encoder for gaze-based turn-transition prediction.

    Input:
        x of shape (batch_size, seq_len, input_dim)

    Output:
        logits of shape (batch_size,)
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        if config.use_learned_positional_encoding:
            self.positional_encoding = LearnedPositionalEncoding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
            )
        else:
            self.positional_encoding = PositionalEncoding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass
        :param x: Tensor of shape (B, T, input_dim)
        :return: Logits: Tensor of shape (B,)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, seq_len, input_dim), got {tuple(x.shape)}"
            )

        if x.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.input_projection(x)    # (B, T, D)
        x = self.positional_encoding(x) # (B, T, D)
        x = self.encoder(x)             # (B T, D)

        pooled = x.mean(dim=1)          # mean pooling over time
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled).squeeze(-1) # (B,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        return probabilities in [0, 1]
        :param x:
        :return: Probabilities in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)