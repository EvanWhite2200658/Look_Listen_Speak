# src/runtime/runtime_logging.py

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class RuntimeLogger:
    """
    Thread-safe JSON event logger for runtime instrumentation.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, **data: Any) -> None:
        record = {
            "event_type": event_type,
            "timestamp_ns": time.time_ns(),
            "data": data,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "")