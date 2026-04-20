# src/runtime/test_strongest_runtime_pipeline.py

from __future__ import annotations

import time
from pathlib import Path
import numpy as np

from src.gaze.service import GazeTrackingService
from src.turn_prediction.inference_model import TrainedTurnModel
from src.runtime.response_gate import TurnResponseGate
from src.turn_prediction.schemas import GazeWindow
from src.runtime.timing_controller import ConfidenceTimingController
from src.turn_prediction.runtime_hcs_adapter import (
RuntimeHCSAdapterConfig,
gaze_window_to_hcs_style_sequence,
)

class DummySileroVAD:
    def user_is_speaking(self) -> bool:
        return False
    # TODO: fill this when installing Silero

class DummyPiperTTS:
    def speak(self, text: str) -> None:
        print(f"[TTS] {text}")

    # TODO: fill this when installing Piper

    def stop(self) -> None:
        print("[TTS STOP]")

HCS_STYLE_FEATURE_NAMES = [
    "eye.x", "eye.y", "eye.z",
    "gaze.x", "gaze.y", "gaze.z",
    "head.x", "head.y", "head.z",
    "headpose.roll", "headpose.pitch", "headpose.yaw",
    "delta_eye.x", "delta_eye.y", "delta_eye.z",
    "delta_gaze.x", "delta_gaze.y", "delta_gaze.z",
    "delta_head.x", "delta_head.y", "delta_head.z",
    "delta_headpose.roll", "delta_headpose.pitch", "delta_headpose.yaw",
]

def print_live_feature_ranges(sequence: np.ndarray, feature_names: list[str]) -> None:
    print("\nLIFE FEATURE RANGES")
    for i, name in enumerate(feature_names):
        col = sequence[:, i]
        print(
            f"{name:22s} "
            f"min={np.min(col):10.4f} "
            f"max={np.max(col):10.4f} "
            f"mean={np.mean(col):10.4f} "
            f"std={np.std(col):10.4f}"
        )

def main() -> None:
    model_path = (
        Path(__file__).resolve().parents[1]
        / "turn_prediction"
        / "artifacts"
        / "turn_prediction_runtime_compatible"
        / "best_model.pt"
    )

    gaze_service = GazeTrackingService(max_buffer_size=200)
    model = TrainedTurnModel(model_path=str(model_path), device="cpu")
    timing_controller = ConfidenceTimingController()
    gate = TurnResponseGate(
        timing_controller=timing_controller,
        vad=DummySileroVAD(),
        tts=DummyPiperTTS(),
    )

    try:
        print("Loaded model:")
        print(model.debug_summary())

        print("Starting gaze service...")
        gaze_service.start(preview=False)
        time.sleep(2.0)

        for _ in range(10):
            samples = gaze_service.get_recent_window(model.expected_window_size)
            print(f"Window sample count: {len(samples)}")

            if len(samples) != model.expected_window_size:
                print("Waiting for full window...")
                time.sleep(0.2)
                continue

            window = GazeWindow(samples=samples)
            sequence = gaze_window_to_hcs_style_sequence(
                window,
                RuntimeHCSAdapterConfig(include_deltas=True),
            )
            print_live_feature_ranges(sequence, HCS_STYLE_FEATURE_NAMES)
            prediction = model.predict(window)
            result = gate.execute_response(
                prediction=prediction,
                text="I think it is my turn to respond now."
            )

            print(
                f"t={prediction.timestamp_ns / 1e9:.3f}s | "
                f"p_turn={prediction.probability:.6f} | "
                f"adjusted_wait_ms={result.timing.adjusted_wait_ms} | "
                f"cancelled_by_vad={result.cancelled_by_vad} | "
                f"spoke={result.spoke}"
            )
            time.sleep(0.2)

    finally:
        print("Stopping gaze service...")
        gaze_service.stop()

if __name__ == "__main__":
    main()