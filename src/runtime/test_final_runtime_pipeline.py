# src/runtime/test_final_runtime_pipeline.py

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from src.runtime.final_runtime_pipeline import FinalRuntimePipeline


def tail_log_file(log_path: Path, stop_event: threading.Event) -> None:
    """
    print runtime log events live so test behaviour can be inspected
    :param log_path:
    :param stop_event:
    :return:
    """
    print(f"[LOG TAIL] Watching: {log_path}")
    last_size = 0

    while not stop_event.is_set():
        if log_path.exists():
            current_size = log_path.stat().st_size
            if current_size < last_size:
                last_size = 0

            if current_size > last_size:
                with log_path.open("r", encoding="utf-8") as fh:
                    fh.seek(last_size)
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            print(
                                f"[{event['event_type']}] "
                                f"{event.get('data', {})}"
                            )
                        except Exception:
                            print(f"[RAW LOG] {line}")
                    last_size = fh.tell()
        time.sleep(0.2)

def main() -> None:
    model_path = (
        Path(__file__).resolve().parents[1]
        / "turn_prediction"
        / "artifacts"
        / "turn_prediction_runtime_compatible"
        / "best_model.pt"
    )

    tts_model_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "tts"
        / "en_GB-alba-medium.onnx"
    )

    log_path = Path(__file__).resolve().parents[2] / "logs" / "runtime_smoke_test.jsonl"

    runtime = FinalRuntimePipeline(
        model_path=str(model_path),
        tts_model_path=str(tts_model_path),
        log_path=str(log_path),
        vad_device_index=17,
        tts_output_device_index=None,
    )

    stop_tail = threading.Event()
    tail_thread = threading.Thread(
        target=tail_log_file,
        args=(log_path, stop_tail),
        daemon=True,
        name="RuntimeLogTail"
    )

    try:
        tail_thread.start()
        runtime.start()
    except KeyboardInterrupt:
        print("\n[SMOKE TEST] Stopping on user interrupt...")
    finally:
        stop_tail.set()
        runtime.stop()
        print("[SMOKE TEST] Finished.")

if __name__ == "__main__":
    main()