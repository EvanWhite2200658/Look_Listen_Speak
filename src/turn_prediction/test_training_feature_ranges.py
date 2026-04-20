# src/turn_prediction/test_training_feature_ranges.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.turn_prediction.columns import RUNTIME_COMPATIBLE_FEATURE_COLUMNS

def main() -> None:
    root_path = Path(__file__).resolve().parents[2]
    data_dir = root_path / "data" / "processed"

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {data_dir}")

    df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)

    print("\nTRAINING FEATURE RANGES")
    for name in RUNTIME_COMPATIBLE_FEATURE_COLUMNS:
        col = df[name].fillna(0.0).to_numpy(dtype=np.float32)
        print(
            f"{name:22s} "
            f"min={np.min(col):10.4f} "
            f"max={np.max(col):10.4f} "
            f"mean={np.mean(col):10.4f} "
            f"std={np.std(col):10.4f}"
        )

if __name__ == "__main__":
    main()