# src/tts/schemas.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    interruption_token: Optional[int] = None


@dataclass(frozen=True)
class SynthesisResult:
    text: str
    synthesis_time_s: float
    playback_started: bool
    playback_stopped_early: bool
    sample_rate: Optional[int]
    model_path: str