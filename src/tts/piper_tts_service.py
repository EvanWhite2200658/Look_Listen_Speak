# src/tts/piper_tts_service.py

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import sounddevice as sd
from piper import PiperVoice

from src.tts.schemas import SynthesisRequest, SynthesisResult


class PiperTTSService:
    """
    Modular Piper TTS service.

    This module converts text to spoken audio and plays it back.
    It must remain off the critical turn-timing path.
    """

    def __init__(
            self,
            model_path: str,
            output_device_index: int | None = None,
            use_cuda: bool = False,
    ) -> None:
        self.model_path = model_path
        self.output_device_index = output_device_index
        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)

        self._stop_event = threading.Event()
        self._play_lock = threading.Lock()
        self._is_speaking = False

    def speak(self, text: str) -> SynthesisResult:
        request = SynthesisRequest(text=text)

        start = time.perf_counter()
        playback_started = False
        playback_stopped_early = False
        sample_rate: Optional[int] = None

        with self._play_lock:
            self._stop_event.clear()
            self._is_speaking = True

            try:
                for chunk in self.voice.synthesize(request.text):
                    sample_rate = int(chunk.sample_rate)

                    if self._stop_event.is_set():
                        playback_stopped_early = True
                        break

                    audio_bytes = chunk.audio_int16_bytes
                    audio = self._int16_bytes_to_float32(audio_bytes)
                    if audio.size == 0:
                        continue

                    playback_started = True
                    sd.play(
                        audio,
                        samplerate=sample_rate,
                        device=self.output_device_index,
                        blocking=True,
                    )

                    if self._stop_event.is_set():
                        playback_stopped_early = True
                        break

            finally:
                self._is_speaking = False

        elapsed = time.perf_counter() - start

        return SynthesisResult(
            text=request.text,
            synthesis_time_s=elapsed,
            playback_started=playback_started,
            playback_stopped_early=playback_stopped_early,
            sample_rate=sample_rate,
            model_path=self.model_path,
        )

    def stop(self) -> None:
        self._stop_event.set()
        sd.stop()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @staticmethod
    def _int16_bytes_to_float32(audio_bytes: bytes):
        import numpy as np

        if not audio_bytes:
            return np.array([], dtype=np.float32)

        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return (audio_int16.astype(np.float32) / 32768.0)