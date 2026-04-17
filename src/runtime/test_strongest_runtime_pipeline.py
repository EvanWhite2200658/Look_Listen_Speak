# src/runtime/test_strongest_runtime_pipeline.py

from __future__ import annotations

import time
from pathlib import Path

from src.gaze.service import GazeTrackingService
from src.turn_prediction.inference_model import TrainedTurnModel
from src.runtime.response_gate import TurnResponseGate
from src.turn_prediction.schemas import GazeWindow
from src.runtime.timing_controller import ConfidenceTimingController

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


def main() -> None:
    model_path = (
        Path(__file__).resolve().parent
        / "turn_prediction"
        / "artifacts"
        / "turn_prediction"
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
            prediction = model.predict(window)
            result = gate.execute_response(
                prediction=prediction,
                text="I think it is my turn to respond now."
            )

            print(
                f"t={prediction.timestamp_ns / 1e9:.3f}s | "
                f"p_turn={prediction.probability:.3f} | "
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