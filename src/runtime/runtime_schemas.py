# src/runtime/runtime_schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.audio_vad.utterance_capture import CapturedUtterance
from src.turn_prediction.schemas import TurnPrediction
from src.runtime.timing_controller import TimingDecision
from src.transcription.schemas import TranscriptionResult
from src.language.schemas import ResponseResult
from src.tts.schemas import SynthesisResult


@dataclass(frozen=True)
class RuntimeLogEvent:
    event_type: str
    timestamp_ns: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DownstreamWorkItem:
    utterance: CapturedUtterance


@dataclass(frozen=True)
class DownstreamResult:
    utterance: CapturedUtterance
    transcription: Optional[TranscriptionResult]
    response: Optional[ResponseResult]
    synthesis: Optional[SynthesisResult]


@dataclass(frozen=True)
class TurnCycleSnapshot:
    prediction: TurnPrediction
    timing: TimingDecision
    inference_time_s: float
    window_size: int
    full_window_ready: bool