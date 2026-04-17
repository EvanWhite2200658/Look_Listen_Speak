# src/runtime/response_gate.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from src.turn_prediction.schemas import TurnPrediction
from timing_controller import ConfidenceTimingController, TimingDecision

class SileroVADLike(Protocol):
    def user_is_speaking(self) -> bool:
        ...

class PiperTTSLike(Protocol):
    def speak(self, text: str) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class ResponseGateResult:
    prediction_probability: float
    timing: TimingDecision
    cancelled_by_vad: bool
    spoke: bool


class TurnResponseGate:
    """
    Applies the learned model only as a timing informer.

    Final behaviour:
    1. model produces confidence
    2. timing controller shortens baseline wait
    3. VAD is checked during the wait and immediately before speaking
    4. if speech is detected, system output is cancelled or stopped
    """

    def __init__(
            self,
            timing_controller: ConfidenceTimingController,
            vad: SileroVADLike,
            tts: PiperTTSLike,
            poll_interval_s: float = 0.02,
    ) -> None:
        self.timing_controller = timing_controller
        self.vad = vad
        self.tts = tts
        self.poll_interval_s = poll_interval_s

    def execute_response(self, prediction: TurnPrediction, text: str) -> ResponseGateResult:
        timing = self.timing_controller.compute_wait(prediction.probability)

        wait_s = timing.adjusted_wait_ms / 1000.0
        deadline = time.monotonic() + wait_s

        while time.monotonic() < deadline:
            if self.vad.user_is_speaking():
                return ResponseGateResult(
                    prediction_probability=prediction.probability,
                    timing=timing,
                    cancelled_by_vad=True,
                    spoke=False,
                )
            time.sleep(self.poll_interval_s)

        if self.vad.user_is_speaking():
            return ResponseGateResult(
                prediction_probability=prediction.probability,
                timing=timing,
                cancelled_by_vad=True,
                spoke=False,
            )

        self.tts.speak(text)

        if self.vad.user_is_speaking():
            self.tts.stop()
            return ResponseGateResult(
                prediction_probability=prediction.probability,
                timing=timing,
                cancelled_by_vad=True,
                spoke=False,
            )

        return ResponseGateResult(
            prediction_probability=prediction.probability,
            timing=timing,
            cancelled_by_vad=False,
            spoke=True,
        )