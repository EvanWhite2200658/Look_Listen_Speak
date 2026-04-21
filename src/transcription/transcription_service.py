# src/transcription/transcription_service.py

from __future__ import annotations

import time
from typing import Optional

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
            device:str = "cpu",
            compute_type: str = "int8",
            language: Optional[str] = None,
    ) -> None:
        self.language = language
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_utterance(self, utterance: CapturedUtterance) -> TranscriptionResult:
        start = time.perf_counter()

        segments_iter, info = self.model.transcribe(
            utterance.audio,
            language=self.language,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

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

        text = " ".join(segment.text for segment in segments).strip()
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_s=utterance.duration_s,
            processing_time_s=elapsed,
            segments=segments,
        )