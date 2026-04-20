# src/turn_prediction/test_dataset.py

from __future__ import annotations

from pathlib import Path

from columns import GAZE_TRANSFORMER_FEATURE_COLUMNS, MAIN_TARGET_COLUMN
from dataset import (
    DatasetConfig,
    build_windowed_samples,
    load_turn_dataframe,
    stack_samples,
)

def main() -> None:
    root_path = Path(__file__).resolve().parents[2]
    csv_path = root_path / "data" / "processed" / "data_UBImpressed_001MD_Client_preprocessed.csv"

    config = DatasetConfig(
        feature_columns=GAZE_TRANSFORMER_FEATURE_COLUMNS,
        label_column=MAIN_TARGET_COLUMN,
        window_size=30,
        stride=5,
        drop_missing_labels=True,
        fillna_value=0.0,
        positive_label_value=1,
    )

    df = load_turn_dataframe(csv_path)
    samples = build_windowed_samples(df, config)
    x, y = stack_samples(samples)

    print(f"Loaded dataframe rows: {len(df)}")
    print(f"Windowed samples: {len(samples)}")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Configured label column: {config.label_column}")

    if len(samples) > 0:
        print(f"First sample label: {samples[0].label}")
        print(f"First sample end_frame_index: {samples[0].end_frame_index}")
        print(f"First sample sequence_id: {samples[0].sequence_id}")
        print(f"Feature dimension: {x.shape[2]}")


if __name__ == "__main__":
    main()