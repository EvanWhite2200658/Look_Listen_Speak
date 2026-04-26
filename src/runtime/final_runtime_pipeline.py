from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional
import sys

import numpy as np

from src.audio_vad.utterance_capture import CapturedUtterance, UtteranceCapture
from src.audio_vad.vad_service import SileroVADService
from src.gaze.service import GazeTrackingService
from src.language.response_generator import QwenResponseGenerator
from src.language.schemas import ResponseRequest
from src.runtime.response_gate import TurnResponseGate
from src.runtime.runtime_logging import RuntimeLogger
from src.runtime.runtime_monitor import RuntimeMonitor
from src.runtime.timing_controller import ConfidenceTimingController
from src.transcription.transcription_service import FasterWhisperTranscriptionService
from src.tts.piper_tts_service import PiperTTSService
from src.turn_prediction.inference_model import TrainedTurnModel
from src.turn_prediction.schemas import GazeWindow
from src.ui.avatar_screen import AvatarScreen

def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(".").resolve() / relative_path


class FinalRuntimePipeline:
    def __init__(
        self,
        model_path: str,
        tts_model_path: str,
        avatar: AvatarScreen,
        log_path: str = "logs/runtime_events.jsonl",
        vad_device_index: int | None = None,
        tts_output_device_index: int | None = None,
    ) -> None:
        self.logger = RuntimeLogger(log_path)
        self.monitor = RuntimeMonitor(self.logger)
        self.avatar = avatar

        self.gaze_service = GazeTrackingService(max_buffer_size=200)
        self.turn_model = TrainedTurnModel(model_path=model_path, device="cpu")
        self.timing_controller = ConfidenceTimingController()

        self.vad = SileroVADService(device_index=vad_device_index)
        self.tts = PiperTTSService(
            model_path=tts_model_path,
            output_device_index=tts_output_device_index,
        )

        self.response_gate = TurnResponseGate(
            timing_controller=self.timing_controller,
            vad=self.vad,
        )

        self.utterance_capture = UtteranceCapture(
            sample_rate=self.vad.input_sample_rate,
        )
        self.vad.add_audio_subscriber(self._handle_audio_chunk)

        self.transcription = FasterWhisperTranscriptionService(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
        )
        self.response_generator = QwenResponseGenerator()

        self._downstream_queue: queue.Queue[CapturedUtterance] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._loop_poll_s = 0.03

        self._response_permission_event = threading.Event()

        self._tts_interrupt_grace_ms = 400
        self._tts_started_at_ns: int | None = None

    def start(self) -> None:
        self.logger.log("runtime_start")
        self.avatar.set_mode("listening")

        self.monitor.start()
        self.gaze_service.start(preview=False)
        self.vad.start()

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._downstream_worker,
            daemon=True,
            name="DownstreamWorker",
        )
        self._worker_thread.start()

        try:
            self._critical_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._response_permission_event.clear()
        self.logger.log("runtime_stop")

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        self.tts.stop()
        self.vad.stop()
        self.gaze_service.stop()
        self.monitor.stop()
        self.avatar.stop()

    def _tts_interrupt_grace_active(self) -> bool:
        if self._tts_started_at_ns is None:
            return False

        elapsed_ms = (time.time_ns() - self._tts_started_at_ns) / 1_000_000.0
        return elapsed_ms < self._tts_interrupt_grace_ms

    def _handle_audio_chunk(self, chunk_48k: np.ndarray, is_speaking: bool, timestamp_ns: int) -> None:
        self.utterance_capture.push_audio(
            chunk=chunk_48k,
            is_speaking=is_speaking,
            timestamp_ns=timestamp_ns,
        )

        self.logger.log(
            "audio_chunk_received",
            chunk_size=len(chunk_48k),
            is_speaking=is_speaking,
        )

        utterance = self.utterance_capture.pop_completed_utterance()
        if utterance is None:
            return

        self.logger.log(
            "utterance_completed",
            start_time_ns=utterance.start_time_ns,
            end_time_ns=utterance.end_time_ns,
            duration_s=utterance.duration_s,
        )
        self.enqueue_completed_utterance(utterance)

    def _critical_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            samples = self.gaze_service.get_recent_window(self.turn_model.expected_window_size)
            full_window_ready = len(samples) == self.turn_model.expected_window_size

            self.logger.log(
                "gaze_window_status",
                sample_count=len(samples),
                expected_window_size=self.turn_model.expected_window_size,
                full_window_ready=full_window_ready,
            )

            if not full_window_ready:
                self._response_permission_event.clear()
                time.sleep(self._loop_poll_s)
                continue

            latest_sample = self.gaze_service.get_latest_sample()
            if latest_sample is not None and latest_sample.features:
                self.logger.log(
                    "gaze_sample",
                    timestamp_ns=latest_sample.timestamp_ns,
                    feature_count=len(latest_sample.features),
                )

            window = GazeWindow(samples=samples)

            infer_start = time.perf_counter()
            prediction = self.turn_model.predict(window)
            infer_elapsed = time.perf_counter() - infer_start

            self.logger.log(
                "turn_prediction",
                probability=prediction.probability,
                is_turn=prediction.is_turn,
                timestamp_ns=prediction.timestamp_ns,
                inference_time_s=infer_elapsed,
                window_size=len(samples),
            )

            timing = self.timing_controller.compute_wait(prediction.probability)
            self.logger.log(
                "timing_decision",
                baseline_wait_ms=timing.baseline_wait_ms,
                adjusted_wait_ms=timing.adjusted_wait_ms,
                confidence=timing.confidence,
            )

            self.avatar.set_mode("listening")
            result = self.response_gate.execute_response(prediction)

            if result.permission_granted:
                self._response_permission_event.set()
            else:
                self._response_permission_event.clear()

                if self.vad.user_is_speaking() and self.tts.is_speaking:
                    if self._tts_interrupt_grace_active():
                        self.logger.log(
                            "tts_interrupt_ignored_in_grace_window",
                            grace_ms=self._tts_interrupt_grace_ms,
                        )
                    else:
                        self.tts.stop()
                        self.logger.log("tts_interrupted_by_vad")

            self.logger.log(
                "response_gate_result",
                cancelled_by_vad=result.cancelled_by_vad,
                permission_granted=result.permission_granted,
                adjusted_wait_ms=result.timing.adjusted_wait_ms,
                prediction_probability=result.prediction_probability,
            )

            self.logger.log(
                "critical_loop_timing",
                loop_time_s=time.perf_counter() - loop_start,
                downstream_queue_size=self._downstream_queue.qsize(),
            )
            time.sleep(self._loop_poll_s)

    def enqueue_completed_utterance(self, utterance: CapturedUtterance) -> None:
        self._downstream_queue.put(utterance)
        self.logger.log(
            "utterance_enqueued",
            duration_s=utterance.duration_s,
            start_time_ns=utterance.start_time_ns,
            end_time_ns=utterance.end_time_ns,
            queue_size=self._downstream_queue.qsize(),
        )

    def _downstream_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                utterance = self._downstream_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.avatar.set_mode("thinking")

                self.logger.log(
                    "utterance_received",
                    duration_s=utterance.duration_s,
                    start_time_ns=utterance.start_time_ns,
                    end_time_ns=utterance.end_time_ns,
                )

                self.logger.log(
                    "transcription_start",
                    duration_s=utterance.duration_s,
                    sample_rate=utterance.sample_rate,
                    sample_count=len(utterance.audio),
                )

                transcription_start = time.perf_counter()
                transcription = self.transcription.transcribe_utterance(utterance)
                self.logger.log(
                    "transcription_complete",
                    text=transcription.text,
                    language=transcription.language,
                    duration_s=transcription.duration_s,
                    processing_time_s=time.perf_counter() - transcription_start,
                )

                self.logger.log("llm_response_start")

                response_start = time.perf_counter()
                response = self.response_generator.generate_response(
                    ResponseRequest(user_text=transcription.text)
                )
                self.logger.log(
                    "llm_response_complete",
                    text=response.text,
                    prompt_tokens=response.prompt_tokens,
                    generated_tokens=response.generated_tokens,
                    generation_time_s=time.perf_counter() - response_start,
                )

                self.logger.log("waiting_for_response_permission")
                while not self._stop_event.is_set():
                    if self._response_permission_event.is_set():
                        break
                    time.sleep(0.01)

                if self._stop_event.is_set():
                    break

                if self.vad.user_is_speaking():
                    self.logger.log("tts_suppressed_by_vad_before_start")
                    continue

                self.avatar.set_mode("speaking")
                self._tts_started_at_ns = time.time_ns()

                synthesis_start = time.perf_counter()
                try:
                    synthesis = self.tts.speak(response.text)
                    self.logger.log(
                        "tts_complete",
                        playback_started=synthesis.playback_started,
                        playback_stopped_early=synthesis.playback_stopped_early,
                        synthesis_time_s=time.perf_counter() - synthesis_start,
                        sample_rate=synthesis.sample_rate,
                    )
                finally:
                    self._tts_started_at_ns = None

                self.avatar.set_mode("listening")

            except Exception as exc:
                import traceback

                self.logger.log(
                    "downstream_worker_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    error_repr=repr(exc),
                    traceback=traceback.format_exc(),
                )
                self.avatar.set_mode("listening")

            finally:
                self._downstream_queue.task_done()


def main() -> None:
    model_path = resource_path("src/turn_prediction/artifacts/turn_prediction_runtime_compatible/best_model.pt")
    tts_model_path = resource_path("models/tts/en_GB-alba-medium.onnx")

    avatar = AvatarScreen()

    runtime = FinalRuntimePipeline(
        model_path=str(model_path),
        tts_model_path=str(tts_model_path),
        avatar=avatar,
        vad_device_index=1,
        tts_output_device_index=None,
    )

    runtime_thread = threading.Thread(
        target=runtime.start,
        daemon=True,
        name="RuntimePipeline",
    )
    runtime_thread.start()

    try:
        avatar.start()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()