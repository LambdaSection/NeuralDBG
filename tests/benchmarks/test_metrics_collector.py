"""
Tests for metrics collection utilities.
"""

import time

import pytest

from neural.benchmarks.metrics_collector import (
    MemoryProfiler,
    MetricsCollector,
    PerformanceTimer,
    ThroughputMeter,
)


class TestMetricsCollector:
    def test_collector_initialization(self):
        collector = MetricsCollector()
        assert collector.metrics_history == []
        assert collector.start_time is None
        assert collector.peak_memory == 0

    def test_start_collection(self):
        collector = MetricsCollector()
        collector.start_collection()
        assert collector.start_time is not None
        assert collector.metrics_history == []

    def test_collect_snapshot(self):
        collector = MetricsCollector()
        collector.start_collection()
        time.sleep(0.1)
        
        snapshot = collector.collect_snapshot()
        
        assert "timestamp" in snapshot
        assert "memory_mb" in snapshot
        assert "cpu_percent" in snapshot
        assert snapshot["timestamp"] >= 0
        assert snapshot["memory_mb"] > 0

    def test_get_summary(self):
        collector = MetricsCollector()
        collector.start_collection()
        
        for _ in range(5):
            collector.collect_snapshot()
            time.sleep(0.05)
        
        summary = collector.get_summary()
        
        assert "peak_memory_mb" in summary
        assert "avg_memory_mb" in summary
        assert "avg_cpu_percent" in summary
        assert "total_time_seconds" in summary
        assert summary["total_time_seconds"] >= 0

    def test_get_system_info(self):
        collector = MetricsCollector()
        system_info = collector.get_system_info()
        
        assert "platform" in system_info
        assert "processor" in system_info
        assert "python_version" in system_info
        assert "cpu_count" in system_info
        assert "total_memory_gb" in system_info
        assert system_info["cpu_count"] > 0


class TestPerformanceTimer:
    def test_timer_context_manager(self):
        with PerformanceTimer("test_operation") as timer:
            time.sleep(0.1)
        
        assert timer.duration is not None
        assert timer.duration >= 0.1
        assert timer.get_duration_ms() >= 100

    def test_timer_name(self):
        timer = PerformanceTimer("my_operation")
        assert timer.name == "my_operation"


class TestThroughputMeter:
    def test_throughput_calculation(self):
        meter = ThroughputMeter()
        meter.start()
        
        time.sleep(0.1)
        meter.update(100)
        
        throughput = meter.get_throughput()
        assert throughput > 0
        assert meter.samples_processed == 100

    def test_multiple_updates(self):
        meter = ThroughputMeter()
        meter.start()
        
        for _ in range(5):
            meter.update(10)
            time.sleep(0.02)
        
        assert meter.samples_processed == 50
        assert meter.get_throughput() > 0


class TestMemoryProfiler:
    def test_profiler_initialization(self):
        profiler = MemoryProfiler()
        assert profiler.baseline_memory == 0
        assert profiler.peak_memory == 0

    def test_start_profiling(self):
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        assert profiler.baseline_memory > 0
        assert profiler.peak_memory > 0

    def test_memory_tracking(self):
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        data = [i for i in range(100000)]
        profiler.update()
        
        assert profiler.get_peak_memory_mb() >= profiler.baseline_memory
        
        del data

    def test_memory_increase(self):
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        initial_peak = profiler.peak_memory
        
        large_data = [i for i in range(1000000)]
        profiler.update()
        
        assert profiler.get_memory_increase_mb() >= 0
        
        del large_data
