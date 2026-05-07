from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import torch

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
from src.turn_prediction.schemas import GazeWindow, TurnPrediction
from src.runtime.demo_monitor import DemoMonitor, DemoMonitorState
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
        avatar: AvatarScreen | None = None,
        log_path: str = "logs/runtime_events.jsonl",
        vad_device_index: int | None = None,
        tts_output_device_index: int | None = None,
        mode: str = "gaze",
        enable_demo_monitor: bool = True,
    ) -> None:
        self.logger = RuntimeLogger(log_path)
        self.monitor = RuntimeMonitor(self.logger)
        self.avatar = avatar
        self.mode = mode

        self.demo_monitor = DemoMonitor() if enable_demo_monitor else None
        self.demo_state = DemoMonitorState(
            confidence=0.0,
            baseline_wait_ms=700,
            adjusted_wait_ms=700,
            speech_detected=False,
            response_allowed=False,
            system_speaking=False,
            latest_delay_ms=None,
            last_event="Runtime initialising...",
        )
        self.latest_frame = None
        self.latest_gaze_point = None
        self.latest_face_box = None

        self.gaze_service = GazeTrackingService(max_buffer_size=200)
        self.turn_model = TrainedTurnModel(model_path=model_path, device="cuda" if torch.cuda.is_available() else "cpu")
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

        asr_device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_compute_type = "float16" if asr_device == "cuda" else "int8"

        self.transcription = FasterWhisperTranscriptionService(
            model_size="small",
            device=asr_device,
            compute_type=asr_compute_type,
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

    def _set_avatar_mode(self, mode: str) -> None:
        if self.avatar is not None:
            self.avatar.set_mode(mode)

    def _draw_demo_monitor(self) -> None:
        if self.demo_monitor is None:
            return

        self.demo_monitor.draw(
            frame=self.latest_frame,
            state=self.demo_state,
            gaze_point=self.latest_gaze_point,
            face_box=self.latest_face_box,
        )

    def start(self) -> None:
        self.logger.log("runtime_start", mode=self.mode)
        self._set_avatar_mode("listening")

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
        if self.avatar is not None:
            self.avatar.stop()
        if self.demo_monitor is not None:
            self.demo_monitor.close()

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

        self.demo_state.speech_detected = bool(is_speaking)
        if is_speaking:
            self.demo_state.last_event = "User speech detected"

        utterance = self.utterance_capture.pop_completed_utterance()
        if utterance is None:
            return

        self.logger.log(
            "utterance_completed",
            start_time_ns=utterance.start_time_ns,
            end_time_ns=utterance.end_time_ns,
            duration_s=utterance.duration_s,
        )

        self.demo_state.speech_detected = False
        self.demo_state.last_event = (
            f"Utterance completed | duration={utterance.duration_s:.2f}s"
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
                self.demo_state.response_allowed = False
                self.demo_state.last_event = (
                    f"Collecting gaze window | {len(samples)}/"
                    f"{self.turn_model.expected_window_size} samples"
                )
                self._draw_demo_monitor()
                time.sleep(self._loop_poll_s)
                continue

            latest_sample = self.gaze_service.get_latest_sample()
            if latest_sample is not None and latest_sample.features:
                self.logger.log(
                    "gaze_sample",
                    timestamp_ns=latest_sample.timestamp_ns,
                    feature_count=len(latest_sample.features),
                )
                raw_gaze = getattr(latest_sample, "raw_gaze", None)
                filtered_gaze = getattr(latest_sample, "filtered_gaze", None)

                if filtered_gaze is not None:
                    try:
                        self.latest_gaze_point = (int(filtered_gaze[0]), int(filtered_gaze[1]))
                    except Exception:
                        pass
                elif raw_gaze is not None:
                    try:
                        self.latest_gaze_point = (int(raw_gaze[0]), int(raw_gaze[1]))
                    except Exception:
                        pass

            window = GazeWindow(samples=samples)

            infer_start = time.perf_counter()
            raw_prediction = self.turn_model.predict(window)
            if self.mode == "baseline":
                prediction = TurnPrediction(
                    timestamp_ns=raw_prediction.timestamp_ns,
                    probability=0.0,
                    is_turn=False,
                )
            else:
                prediction = raw_prediction
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

            self.demo_state.confidence = float(timing.confidence)
            self.demo_state.baseline_wait_ms = int(timing.baseline_wait_ms)
            self.demo_state.adjusted_wait_ms = int(timing.adjusted_wait_ms)
            self.demo_state.latest_delay_ms = int(timing.adjusted_wait_ms)
            self.demo_state.last_event = (
                f"Timing decision | confidence={timing.confidence:.2f} | "
                f"wait={timing.adjusted_wait_ms:.0f}ms"
            )

            self._set_avatar_mode("listening")

            result = self.response_gate.execute_response(prediction)
            speech_end_time_ns = prediction.timestamp_ns
            response_start_time_ns = time.time_ns()
            response_latency_ms = (response_start_time_ns - speech_end_time_ns) / 1_000_000.0

            self.logger.log(
                "response_timing",
                mode=self.mode,  # "baseline" or "gaze"
                speech_end_time_ns=speech_end_time_ns,
                response_start_time_ns=response_start_time_ns,
                response_latency_ms=response_latency_ms,
                model_confidence=raw_prediction.probability,
                permission_granted=result.permission_granted,
                cancelled_by_vad=result.cancelled_by_vad,
            )

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

            self.demo_state.response_allowed = bool(result.permission_granted)
            self.demo_state.speech_detected = bool(self.vad.user_is_speaking())
            self.demo_state.system_speaking = bool(self.tts.is_speaking)

            if result.permission_granted:
                self.demo_state.last_event = (
                    f"Response permission granted | confidence={timing.confidence:.2f} | "
                    f"wait={timing.adjusted_wait_ms:.0f}ms"
                )
            else:
                self.demo_state.last_event = (
                    f"Response blocked | VAD speaking={self.vad.user_is_speaking()} | "
                    f"confidence={timing.confidence:.2f}"
                )

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
            self._draw_demo_monitor()
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
                self._set_avatar_mode("thinking")
                self.demo_state.last_event = "Processing completed utterance"
                self._draw_demo_monitor()

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

                self.demo_state.last_event = "Transcribing user speech"
                self._draw_demo_monitor()

                transcription_start = time.perf_counter()
                transcription = self.transcription.transcribe_utterance(utterance)
                self.logger.log(
                    "transcription_complete",
                    text=transcription.text,
                    language=transcription.language,
                    duration_s=transcription.duration_s,
                    processing_time_s=time.perf_counter() - transcription_start,
                )

                self.demo_state.last_event = f"Transcription complete | {transcription.text[:70]}"
                self._draw_demo_monitor()

                self.logger.log("llm_response_start")

                self.demo_state.last_event = "Generating system response"
                self._draw_demo_monitor()

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

                self.demo_state.last_event = "System response generated, waiting for timing permission"
                self._draw_demo_monitor()

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

                self._set_avatar_mode("speaking")
                self.demo_state.system_speaking = True
                self.demo_state.last_event = "System speaking"
                self._draw_demo_monitor()

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
                    self.demo_state.system_speaking = False
                    self.demo_state.last_event = "System response finished"
                    self._draw_demo_monitor()

                self._set_avatar_mode("listening")

            except Exception as exc:
                import traceback

                self.logger.log(
                    "downstream_worker_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    error_repr=repr(exc),
                    traceback=traceback.format_exc(),
                )
                self._set_avatar_mode("listening")
                self.demo_state.system_speaking = False
                self.demo_state.last_event = f"Worker error | {type(exc).__name__}"
                self._draw_demo_monitor()

            finally:
                self._downstream_queue.task_done()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    log_path = project_root / "src" / "runtime" / "logs" / "gaze_runtime_events.jsonl"

    model_path = (
            project_root
            / "src"
            / "turn_prediction"
            / "artifacts"
            / "turn_prediction_runtime_compatible"
            / "best_model.pt"
    )

    tts_model_path = (
            project_root
            / "models"
            / "tts"
            / "en_GB-alba-medium.onnx"
    )
    avatar = AvatarScreen()


    runtime = FinalRuntimePipeline(
        model_path=str(model_path),
        tts_model_path=str(tts_model_path),
        avatar=avatar,
        log_path=str(log_path),
        vad_device_index=None,
        tts_output_device_index=None,
        mode="gaze",
        enable_demo_monitor=True,
    )

    runtime_thread = threading.Thread(
        target=runtime.start,
        daemon=True,
        name="RuntimePipeline",
    )

    try:
        runtime_thread.start()
    except KeyboardInterrupt:
        print("Stopping runtime pipeline...")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()