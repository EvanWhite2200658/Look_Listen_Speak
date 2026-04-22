# src/transcription/transcription_service.py

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from src.audio_vad.utterance_capture import CapturedUtterance
from src.transcription.schemas import TranscriptionResult, TranscriptionSegment


class FasterWhisperTranscriptionService:
    """
    Transcribes completed utterances using faster-whisper.
    This service does not own the microphone and does not perform endpointing.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        target_sample_rate: int = 16000,
    ) -> None:
        self.language = language
        self.target_sample_rate = target_sample_rate
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_utterance(self, utterance: CapturedUtterance) -> TranscriptionResult:
        start = time.perf_counter()

        audio = self._prepare_audio(
            audio=utterance.audio,
            source_sample_rate=utterance.sample_rate,
            target_sample_rate=self.target_sample_rate,
        )

        print(
            f"[ASR] duration_s={utterance.duration_s:.3f} "
            f"source_sr={utterance.sample_rate} "
            f"target_sr={self.target_sample_rate} "
            f"input_samples={len(utterance.audio)} "
            f"resampled_samples={len(audio)} "
            f"dtype={audio.dtype}"
        )

        print("[ASR] calling model.transcribe(...)")
        segments_iter, info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
            beam_size=1,
            best_of=1,
        )
        print("[ASR] model.transcribe(...) returned")

        print("[ASR] consuming segments iterator")
        segments = [
            TranscriptionSegment(
                start_s=float(segment.start),
                end_s=float(segment.end),
                text=segment.text.strip(),
                avg_logprob=getattr(segment, "avg_logprob", None),
                no_speech_prob=getattr(segment, "no_speech_prob", None),
            )
            for segment in segments_iter
        ]
        print(f"[ASR] consumed segments iterator, count={len(segments)}")

        text = " ".join(segment.text for segment in segments).strip()
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_s=utterance.duration_s,
            processing_time_s=elapsed,
            segments=segments,
        )

    def _prepare_audio(
        self,
        audio: np.ndarray,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)

        # normalize if incoming audio is not already in [-1, 1]
        max_abs = np.max(np.abs(mono)) if mono.size else 0.0
        if max_abs > 1.5:
            mono = mono / 32768.0

        if source_sample_rate == target_sample_rate:
            return mono.astype(np.float32, copy=False)

        duration = len(mono) / float(source_sample_rate)
        target_length = int(round(duration * target_sample_rate))
        if target_length <= 0:
            return np.array([], dtype=np.float32)

        source_positions = np.linspace(0.0, len(mono) - 1, num=len(mono), dtype=np.float32)
        target_positions = np.linspace(0.0, len(mono) - 1, num=target_length, dtype=np.float32)
        resampled = np.interp(target_positions, source_positions, mono)
        return resampled.astype(np.float32, copy=False)