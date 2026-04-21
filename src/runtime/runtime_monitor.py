# src/runtime/runtime_monitor.py

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import psutil

from src.runtime.runtime_logging import RuntimeLogger

try:
    import torch
except Exception:
    torch = None

try:
    import pynvml
except Exception:
    pynvml = None

class RuntimeMonitor:
    """
    Periodically records process/system resource metrics.
    """

    def __init__(
            self,
            logger: RuntimeLogger,
            poll_interval_s: float = 1.0,
    ) -> None:
        self.logger = logger
        self.poll_interval_s = poll_interval_s
        self._process = psutil.Process(os.getpid())
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_ready = False

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="RuntimeMonitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run_loop(self) -> None:
        self._process.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self.logger.log(
                "resource_usage",
                process_cpu_percent=self._process.cpu_percent(interval=None),
                process_rss_mb=self._process.memory_info().rss / (1024 * 1024),
                system_memory_percent=psutil.virtual_memory().percent,
                gpu_memory_allocated_mb=self._get_torch_gpu_allocated_mb(),
                gpu_memory_reserved_mb=self._get_torch_gpu_reserved_mb(),
                gpu_util_percent=self._get_nvml_gpu_util_percent(),
                gpu_memory_used_mb=self._get_nvml_gpu_memory_mb(),
            )
            time.sleep(self.poll_interval_s)

    def _get_torch_gpu_allocated_mb(self) -> Optional[float]:
        if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
            return None
        try:
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            return None

    def _get_torch_gpu_reserved_mb(self) -> Optional[float]:
        if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
            return None
        try:
            return float(torch.cuda.memory_reserved() / (1024 * 1024))
        except Exception:
            return None

    def _get_nvml_gpu_util_percent(self) -> Optional[float]:
        if not self._nvml_ready:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return None

    def _get_nvml_gpu_memory_mb(self) -> Optional[float]:
        if not self._nvml_ready:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(mem.used / (1024 * 1024))
        except Exception:
            return None