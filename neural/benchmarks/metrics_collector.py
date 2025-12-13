"""
Metrics collection utilities for benchmarking.
"""

import os
import platform
import time
from typing import Any, Dict, List, Optional

import numpy as np
import psutil


class MetricsCollector:
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())

    def start_collection(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self.metrics_history = []

    def collect_snapshot(self) -> Dict[str, Any]:
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > self.peak_memory:
            self.peak_memory = memory_mb

        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        snapshot = {
            "timestamp": time.time() - self.start_time if self.start_time else 0,
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
        }
        
        self.metrics_history.append(snapshot)
        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {
                "peak_memory_mb": 0,
                "avg_memory_mb": 0,
                "avg_cpu_percent": 0,
                "total_time_seconds": 0,
            }

        return {
            "peak_memory_mb": self.peak_memory,
            "avg_memory_mb": np.mean([m["memory_mb"] for m in self.metrics_history]),
            "avg_cpu_percent": np.mean([m["cpu_percent"] for m in self.metrics_history]),
            "total_time_seconds": time.time() - self.start_time if self.start_time else 0,
        }

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "available_memory_gb": psutil.virtual_memory().available / (1024 ** 3),
        }


class PerformanceTimer:
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

    def get_duration_ms(self) -> float:
        if self.duration is None:
            return 0.0
        return self.duration * 1000


class ThroughputMeter:
    def __init__(self):
        self.samples_processed = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.samples_processed = 0

    def update(self, batch_size: int):
        self.samples_processed += batch_size

    def get_throughput(self) -> float:
        if self.start_time is None:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        
        return self.samples_processed / elapsed_time


class MemoryProfiler:
    def __init__(self):
        self.baseline_memory = 0
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())

    def start_profiling(self):
        memory_info = self.process.memory_info()
        self.baseline_memory = memory_info.rss / (1024 * 1024)
        self.peak_memory = self.baseline_memory

    def update(self):
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss / (1024 * 1024)
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_memory_increase_mb(self) -> float:
        return self.peak_memory - self.baseline_memory

    def get_peak_memory_mb(self) -> float:
        return self.peak_memory
