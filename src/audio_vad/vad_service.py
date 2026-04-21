# src/audio_vad/vad_service.py

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

AudioSubscriber = Callable[[np.ndarray, bool, int], None]


class SileroVADService:
    """
    Real-time microphone VAD service.

    Exposes a lightweight 'user_is_speaking()' method for runtime gating.
    Audio capture runs continuously in the background and speech state is updated
    from a short rolling window.

    Optional subscribers can receive raw input audio chunks, the derived speaking
    state, and a callback timestamp. This allows utterance capture to reuse the
    same microphone stream without creating a second competing input device.
    """

    def __init__(
            self,
            device_index: int | None = None,
            input_sample_rate: int = 48000,
            model_sample_rate: int = 16000,
            block_duration_ms: int = 30,
            speech_threshold: float = 0.5,
            min_silence_duration_ms: int = 150,
    ) -> None:
        self.device_index = device_index
        self.input_sample_rate = input_sample_rate
        self.model_sample_rate = model_sample_rate
        self.block_duration_ms = block_duration_ms
        self.speech_threshold = speech_threshold
        self.min_silence_duration_ms = min_silence_duration_ms

        self._model = load_silero_vad()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._subscribers: list[AudioSubscriber] = []

        self._is_running = False
        self._is_speaking = False
        self._input_block_size = int(self.input_sample_rate * self.block_duration_ms / 1000)

    def start(self) -> None:
        if self._is_running:
            return

        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.input_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._input_block_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_running = True

    def stop(self) -> None:
        if not self._is_running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            self._is_speaking = False

        self._is_running = False

    def add_audio_subscriber(self, subscriber: AudioSubscriber) -> None:
        with self._lock:
            self._subscribers.append(subscriber)

    def remove_audio_subscriber(self, subscriber: AudioSubscriber) -> None:
        with self._lock:
            self._subscribers = [fn for fn in self._subscribers if fn is not subscriber]

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            return

        timestamp_ns = time.time_ns()
        audio_48k = np.squeeze(np.copy(indata), axis=1).astype(np.float32, copy=False)
        audio_16k = self._resample_to_16k(audio_48k)
        if audio_16k.size == 0:
            return

        chunk = torch.from_numpy(audio_16k).float()
        speech_dicts = get_speech_timestamps(
            chunk,
            self._model,
            threshold=self.speech_threshold,
            sampleing_rate=self.model_sample_rate,
            min_silence_duration_ms=self.min_silence_duration_ms,
        )

        speaking_now = len(speech_dicts) > 0
        with self._lock:
            self._is_speaking = speaking_now
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber(audio_48k, speaking_now, timestamp_ns)
            except Exception:
                # keep the audio callback resilient; logging can be added outside
                # the callback if needed
                pass

    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        if self.input_sample_rate == self.model_sample_rate:
            return audio.astype(np.float32, copy=False)

        duration = len(audio) / float(self.input_sample_rate)
        target_length = int(round(duration * self.model_sample_rate))
        if target_length <= 0:
            return np.array([], dtype=np.float32)

        source_positions = np.linspace(0.0, len(audio) - 1, num=len(audio), dtype=np.float32)
        target_positions = np.linspace(0.0, len(audio) - 1, num=target_length, dtype=np.float32)
        resampled = np.interp(target_positions, source_positions, audio)
        return resampled.astype(np.float32, copy=False)

    def user_is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def is_running(self) -> bool:
        return self._is_running