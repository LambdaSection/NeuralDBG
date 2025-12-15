"""
Enhanced metrics collection for comprehensive benchmarking.

This module provides utilities for collecting detailed performance,
code quality, and resource utilization metrics.
"""

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class SystemInfo:
    """System and environment information for reproducibility."""
    python_version: str
    platform: str
    processor: str
    cpu_count: int
    memory_total_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @classmethod
    def collect(cls) -> "SystemInfo":
        """Collect current system information."""
        gpu_available = False
        gpu_name = None
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_available = True
                gpu_name = gpus[0].name if gpus else None
        except Exception:
            pass
        
        if not gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
        
        memory_gb = 0.0
        if PSUTIL_AVAILABLE:
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        return cls(
            python_version=sys.version,
            platform=platform.platform(),
            processor=platform.processor() or platform.machine(),
            cpu_count=os.cpu_count() or 0,
            memory_total_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total_gb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "timestamp": self.timestamp,
        }


class ResourceMonitor:
    """Monitor CPU, memory, and GPU usage during benchmark execution."""
    
    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.monitoring = False
        self.samples: List[Dict[str, float]] = []
    
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.samples = []
        
        if self.process:
            self.process.cpu_percent()
    
    def sample(self):
        """Take a resource usage sample."""
        if not self.monitoring or not self.process:
            return
        
        try:
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            cpu_percent = self.process.cpu_percent()
            
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            
            self.samples.append({
                "timestamp": time.time(),
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
            })
        except Exception:
            pass
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return results."""
        self.monitoring = False
        
        avg_memory_mb = 0.0
        avg_cpu_percent = 0.0
        
        if self.samples:
            avg_memory_mb = sum(s["memory_mb"] for s in self.samples) / len(self.samples)
            avg_cpu_percent = sum(s["cpu_percent"] for s in self.samples) / len(self.samples)
        
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": avg_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "avg_cpu_percent": avg_cpu_percent,
            "num_samples": len(self.samples),
        }


class CodeMetrics:
    """Analyze code quality and complexity metrics."""
    
    @staticmethod
    def count_lines(code: str) -> Dict[str, int]:
        """Count different types of lines in code."""
        lines = code.split("\n")
        
        total = len(lines)
        empty = sum(1 for line in lines if not line.strip())
        comments = sum(1 for line in lines if line.strip().startswith("#"))
        code_lines = total - empty - comments
        
        return {
            "total": total,
            "empty": empty,
            "comments": comments,
            "code": code_lines,
            "non_empty": total - empty,
        }
    
    @staticmethod
    def count_tokens(code: str) -> Dict[str, int]:
        """Count various code tokens."""
        return {
            "imports": code.count("import"),
            "classes": code.count("class "),
            "functions": code.count("def "),
            "decorators": code.count("@"),
            "list_comprehensions": code.count("for ") + code.count("if ") - code.count("def "),
        }
    
    @staticmethod
    def calculate_complexity(code: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        lines_info = CodeMetrics.count_lines(code)
        tokens = CodeMetrics.count_tokens(code)
        
        setup_complexity = (
            tokens["imports"] + 
            tokens["classes"] * 2 + 
            tokens["functions"]
        )
        
        readability_score = max(0, min(10, 10 - (lines_info["code"] / 10)))
        
        nesting_depth = max(
            len(line) - len(line.lstrip())
            for line in code.split("\n")
            if line.strip()
        ) // 4 if code.strip() else 0
        
        return {
            "lines": lines_info,
            "tokens": tokens,
            "setup_complexity": setup_complexity,
            "readability_score": readability_score,
            "max_nesting_depth": nesting_depth,
        }


class PerformanceTimer:
    """High-precision timer for performance measurements."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            if self.start_time is None:
                return 0.0
            return time.perf_counter() - self.start_time
        return self.elapsed


class MetricsCollector:
    """
    Comprehensive metrics collector for benchmarking.
    
    Collects:
    - System information
    - Resource usage
    - Code quality metrics
    - Performance timings
    """
    
    def __init__(self):
        self.system_info = SystemInfo.collect()
        self.resource_monitor = ResourceMonitor()
        self.timers: Dict[str, PerformanceTimer] = {}
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.resource_monitor.start()
    
    def sample_resources(self):
        """Sample current resource usage."""
        self.resource_monitor.sample()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and get results."""
        return self.resource_monitor.stop()
    
    def start_timer(self, name: str) -> PerformanceTimer:
        """Start a named timer."""
        timer = PerformanceTimer(name)
        self.timers[name] = timer
        timer.__enter__()
        return timer
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name in self.timers:
            timer = self.timers[name]
            timer.__exit__(None, None, None)
            return timer.get_elapsed()
        return 0.0
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code metrics."""
        return CodeMetrics.calculate_complexity(code)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        resource_stats = self.resource_monitor.stop()
        
        timer_stats = {
            name: timer.get_elapsed()
            for name, timer in self.timers.items()
        }
        
        return {
            "system_info": self.system_info.to_dict(),
            "resources": resource_stats,
            "timings": timer_stats,
        }


def compare_metrics(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """
    Generate a human-readable comparison of two metric sets.
    
    Args:
        metrics_a: First metrics dictionary
        metrics_b: Second metrics dictionary
        label_a: Label for first metrics
        label_b: Label for second metrics
    
    Returns:
        Formatted comparison string
    """
    comparison = []
    comparison.append(f"\n{'='*60}")
    comparison.append(f"Comparison: {label_a} vs {label_b}")
    comparison.append(f"{'='*60}\n")
    
    def compare_metric(name: str, value_a: float, value_b: float, lower_is_better: bool = True):
        """Compare a single metric."""
        diff = value_b - value_a
        pct_diff = (diff / value_a * 100) if value_a != 0 else 0
        
        if lower_is_better:
            winner = label_a if value_a < value_b else label_b
            better = value_a < value_b
        else:
            winner = label_a if value_a > value_b else label_b
            better = value_a > value_b
        
        symbol = "✓" if better else "✗"
        
        comparison.append(f"{name}:")
        comparison.append(f"  {label_a}: {value_a:.2f}")
        comparison.append(f"  {label_b}: {value_b:.2f}")
        comparison.append(f"  {symbol} Winner: {winner} ({abs(pct_diff):.1f}% {'better' if better else 'worse'})")
        comparison.append("")
    
    if "lines_of_code" in metrics_a and "lines_of_code" in metrics_b:
        compare_metric("Lines of Code", 
                      metrics_a["lines_of_code"], 
                      metrics_b["lines_of_code"],
                      lower_is_better=True)
    
    if "training_time_seconds" in metrics_a and "training_time_seconds" in metrics_b:
        compare_metric("Training Time",
                      metrics_a["training_time_seconds"],
                      metrics_b["training_time_seconds"],
                      lower_is_better=True)
    
    if "model_accuracy" in metrics_a and "model_accuracy" in metrics_b:
        compare_metric("Model Accuracy",
                      metrics_a["model_accuracy"],
                      metrics_b["model_accuracy"],
                      lower_is_better=False)
    
    comparison.append("="*60)
    
    return "\n".join(comparison)
