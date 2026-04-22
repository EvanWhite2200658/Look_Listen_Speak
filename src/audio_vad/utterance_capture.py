# src/audio_vad/utterance_capture.py

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from src.runtime.runtime_logging import RuntimeLogger


@dataclass(frozen=True)
class CapturedUtterance:
    audio: np.ndarray
    sample_rate: int
    start_time_ns: int
    end_time_ns: int
    duration_s: float

class UtteranceCapture:
    """
    Buffers mono audio chunks and finalises an utterance after trailing silence.

    This class does not perform transcription. It only converts VAD state changes
    into completed utterance audio segments.
    """

    def __init__(
            self,
            sample_rate: int,
            silence_hold_ms: int = 300,
            min_utterance_ms: int = 200,
            max_pre_speech_ms: int = 150,
            log_path: str = "logs/runtime_events.jsonl",
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_hold_ms = silence_hold_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_pre_speech_ms = max_pre_speech_ms
        self.logger = RuntimeLogger(log_path)

        self._lock = threading.Lock()
        self._pre_speech_chunks: list[np.ndarray] = []
        self._active_chunks: list[np.ndarray] = []
        self._in_speech = False
        self._speech_start_ns: Optional[int] = None
        self._last_speech_ns: Optional[int] = None
        self._latest_completed: Optional[CapturedUtterance] = None

        self._max_pre_speech_samples = int(self.sample_rate * self.max_pre_speech_ms / 1000)
        self._min_utterance_samples = int(self.sample_rate * self.min_utterance_ms / 1000)
        self._silence_hold_ns = int(self.silence_hold_ms * 1_000_000)

    def push_audio(self, chunk: np.ndarray, is_speaking: bool, timestamp_ns: Optional[int] = None) -> None:
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.logger.log(
            "utterance_capture_progress",
            is_speaking=is_speaking,
        )
        if mono.size == 0:
            return

        with self._lock:

            if not self._in_speech:
                self._append_pre_speech(mono)

            if is_speaking:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_start_ns = ts
                    self._active_chunks = list(self._pre_speech_chunks)
                self._active_chunks.append(mono)
                self._last_speech_ns = ts
                return

            if self._in_speech:
                self._active_chunks.append(mono)
                if self._last_speech_ns is not None and (ts - self._last_speech_ns) >= self._silence_hold_ns:

                    self._finalise(ts)



    def pop_completed_utterance(self) -> Optional[CapturedUtterance]:
        with self._lock:
            utterance = self._latest_completed
            self._latest_completed = None
            return utterance

    def _append_pre_speech(self, chunk: np.ndarray) -> None:
        self._pre_speech_chunks.append(chunk)
        merged = np.concatenate(self._pre_speech_chunks, axis=0)
        if merged.size > self._max_pre_speech_samples:
            merged = merged[-self._max_pre_speech_samples:]
        self._pre_speech_chunks = [merged]

    def _finalise(self, end_time_ns: int) -> None:
        if not self._active_chunks or self._speech_start_ns is None:
            self._reset_state()
            return

        audio = np.concatenate(self._active_chunks, axis=0).astype(np.float32, copy=False)

        if audio.size < self._min_utterance_samples:
            self._reset_state()
            return

        self._latest_completed = CapturedUtterance(
            audio=audio,
            sample_rate=self.sample_rate,
            start_time_ns=self._speech_start_ns,
            end_time_ns=end_time_ns,
            duration_s=audio.size / float(self.sample_rate),
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self._pre_speech_chunks = []
        self._active_chunks = []
        self._in_speech = False
        self._speech_start_ns = None
        self._last_speech_ns = None