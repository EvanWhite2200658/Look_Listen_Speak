# src/transcription/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TranscriptionSegment:
    start_s: float
    end_s: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration_s: float
    processing_time_s: float
    segments: List[TranscriptionSegment] = field(default_factory=list)