 # src/turn_prediction/test_live_vs_training_feature_ranges.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.gaze.service import GazeTrackingService
from src.turn_prediction.columns import RUNTIME_COMPATIBLE_FEATURE_COLUMNS
from src.turn_prediction.runtime_hcs_adapter import (
    RuntimeHCSAdapterConfig,
    gaze_window_to_hcs_style_sequence,
)
from src.turn_prediction.schemas import GazeWindow


def describe_array(arr: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    root_path = Path(__file__).resolve().parents[2]
    data_dir = root_path / "data" / "processed"

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found in {data_dir}")

    train_df = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)

    train_stats: dict[str, tuple[float, float, float, float]] = {}
    for name in RUNTIME_COMPATIBLE_FEATURE_COLUMNS:
        arr = train_df[name].fillna(0.0).to_numpy(dtype=np.float32)
        train_stats[name] = describe_array(arr)

    gaze_service = GazeTrackingService(max_buffer_size=200)

    try:
        print("Starting gaze service...")
        gaze_service.start(preview=False)
        time.sleep(2.0)

        samples = gaze_service.get_recent_window(30)
        window = GazeWindow(samples=samples)

        sequence = gaze_window_to_hcs_style_sequence(
            window,
            RuntimeHCSAdapterConfig(include_deltas=True),
        )

        if sequence.shape[0] == 0:
            print("No live sequence available.")
            return

        live_stats: dict[str, tuple[float, float, float, float]] = {}
        for i, name in enumerate(RUNTIME_COMPATIBLE_FEATURE_COLUMNS):
            arr = sequence[:, i].astype(np.float32)
            live_stats[name] = describe_array(arr)

        print("\nTRAIN vs LIVE")
        for name in RUNTIME_COMPATIBLE_FEATURE_COLUMNS:
            tmin, tmax, tmean, tstd = train_stats[name]
            lmin, lmax, lmean, lstd = live_stats[name]
            print(
                f"{name:22s} "
                f"train_mean={tmean:10.4f} "
                f"live_mean={lmean:10.4f} "
                f"train_std={tstd:10.4f} "
                f"live_std={lstd:10.4f}"
            )

        print("\nLIVE MEAN SHIFT IN TRAINING STD UNITS")
        for name in RUNTIME_COMPATIBLE_FEATURE_COLUMNS:
            _, _, tmean, tstd = train_stats[name]
            _, _, lmean, _ = live_stats[name]

            if tstd < 1e-6:
                shift = float("inf") if abs(lmean - tmean) > 1e-6 else 0.0
            else:
                shift = (lmean - tmean) / tstd

            print(f"{name:22s} z_shift={shift:10.4f}")

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()


if __name__ == "__main__":
    main()