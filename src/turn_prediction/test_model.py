# src/turn_prediction/test_model.py

from __future__ import annotations

import torch

from model import TransformerConfig, TurnShiftTransformer


def main() -> None:
    batch_size = 8
    seq_len = 30
    input_dim = 24

    config = TransformerConfig(
        input_dim=input_dim,
        max_seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_learned_positional_encoding=True,
    )

    model = TurnShiftTransformer(config)

    x = torch.randn(batch_size, seq_len, input_dim)

    logits = model(x)
    probs = model.predict_proba(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probabilities shape: {tuple(probs.shape)}")
    print(f"First 3 probabilities: {probs[:3].detach().cpu().numpy()}")


if __name__ == "__main__":
    main()