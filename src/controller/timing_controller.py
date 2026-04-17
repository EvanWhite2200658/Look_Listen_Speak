# src/controller/timing_controller.py

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class TimingDecision:
    baseline_wait_ms: int
    confidence: float
    adjusted_wait_ms: int

@dataclass(frozen=True)
class TimingControllerConfig:
    baseline_wait_ms: int = 700
    min_wait_ms: int = 150


class ConfidenceTimingController:
    """
    Reduce a baseline wait using model confidence.

    Formula:
        adjusted_wait = baseline_wait * (1 - confidence)

    Then clamp to a minimum wait to avoid overly aggressive interruption.
    """

    def __init__(self, config: TimingControllerConfig | None = None) -> None:
        self.config = config or TimingControllerConfig()

    def compute_wait(self, confidence: float) -> TimingDecision:
        confidence = max(0.0, min(1.0, float(confidence)))

        adjusted = int(round(self.config.baseline_wait_ms * (1.0 - confidence)))
        adjusted = max(self.config.min_wait_ms, adjusted)

        return TimingDecision(
            baseline_wait_ms=self.config.baseline_wait_ms,
            confidence=confidence,
            adjusted_wait_ms=adjusted,
        )