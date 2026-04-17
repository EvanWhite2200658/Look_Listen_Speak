# src/turn_prediction/test_overlap_training_contract.py

from overlap_dataset import OverlapSequenceDataset
from pathlib import Path

def main():
    path = Path(__file__).resolve().parents[2]
    dataset = OverlapSequenceDataset(
        data_path=path / "data" / "processed" / "data_UBImpressed_001MD_Client_preprocessed.csv",
        window_size=30,
        stride=5,
    )

    meta = dataset.metadata

    print("Overlap training dataset contract")
    print(f"Samples: {meta.num_samples}")
    print(f"Seq len: {meta.seq_len}")
    print(f"Feature dim: {meta.feature_dim}")
    print(f"Sequences: {meta.num_sequences}")
    print(f"Positive labels: {meta.positive_count}")
    print(f"Negative labels: {meta.negative_count}")

    x, y = dataset[0]
    print("Sample shape:", x.shape)
    print("Label:", y)


if __name__ == "__main__":
    main()