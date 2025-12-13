"""
Prometheus metrics exporter for production monitoring.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = CollectorRegistry = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"


class MetricsRegistry:
    """Registry for Prometheus metrics."""
    
    def __init__(self):
        """Initialize metrics registry."""
        if not PROMETHEUS_AVAILABLE:
            self.registry = None
            self.metrics: Dict[str, Any] = {}
            return
        
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        # Initialize standard metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Prediction metrics
        self.metrics['predictions_total'] = Counter(
            'neural_predictions_total',
            'Total number of predictions',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['prediction_latency'] = Histogram(
            'neural_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model', 'version'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        # Performance metrics
        self.metrics['model_accuracy'] = Gauge(
            'neural_model_accuracy',
            'Model accuracy',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['model_precision'] = Gauge(
            'neural_model_precision',
            'Model precision',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['model_recall'] = Gauge(
            'neural_model_recall',
            'Model recall',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['model_f1'] = Gauge(
            'neural_model_f1',
            'Model F1 score',
            ['model', 'version'],
            registry=self.registry
        )
        
        # Drift metrics
        self.metrics['drift_score'] = Gauge(
            'neural_drift_score',
            'Model drift score',
            ['model', 'version', 'drift_type'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.metrics['data_quality_score'] = Gauge(
            'neural_data_quality_score',
            'Data quality score',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['missing_values_rate'] = Gauge(
            'neural_missing_values_rate',
            'Missing values rate',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.metrics['outlier_rate'] = Gauge(
            'neural_outlier_rate',
            'Outlier rate',
            ['model', 'version'],
            registry=self.registry
        )
        
        # System metrics
        self.metrics['errors_total'] = Counter(
            'neural_errors_total',
            'Total number of errors',
            ['model', 'version', 'error_type'],
            registry=self.registry
        )
        
        self.metrics['active_models'] = Gauge(
            'neural_active_models',
            'Number of active models',
            registry=self.registry
        )
    
    def record_prediction(
        self,
        model: str = "default",
        version: str = "1.0",
        latency_seconds: float = 0.0
    ):
        """Record a prediction."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['predictions_total'].labels(
            model=model,
            version=version
        ).inc()
        
        if latency_seconds > 0:
            self.metrics['prediction_latency'].labels(
                model=model,
                version=version
            ).observe(latency_seconds)
    
    def update_performance_metrics(
        self,
        model: str,
        version: str,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None
    ):
        """Update performance metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if accuracy is not None:
            self.metrics['model_accuracy'].labels(
                model=model,
                version=version
            ).set(accuracy)
        
        if precision is not None:
            self.metrics['model_precision'].labels(
                model=model,
                version=version
            ).set(precision)
        
        if recall is not None:
            self.metrics['model_recall'].labels(
                model=model,
                version=version
            ).set(recall)
        
        if f1 is not None:
            self.metrics['model_f1'].labels(
                model=model,
                version=version
            ).set(f1)
    
    def update_drift_metrics(
        self,
        model: str,
        version: str,
        drift_scores: Dict[str, float]
    ):
        """Update drift metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        for drift_type, score in drift_scores.items():
            self.metrics['drift_score'].labels(
                model=model,
                version=version,
                drift_type=drift_type
            ).set(score)
    
    def update_quality_metrics(
        self,
        model: str,
        version: str,
        quality_score: float,
        missing_rate: float = 0.0,
        outlier_rate: float = 0.0
    ):
        """Update data quality metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['data_quality_score'].labels(
            model=model,
            version=version
        ).set(quality_score)
        
        self.metrics['missing_values_rate'].labels(
            model=model,
            version=version
        ).set(missing_rate)
        
        self.metrics['outlier_rate'].labels(
            model=model,
            version=version
        ).set(outlier_rate)
    
    def record_error(
        self,
        model: str,
        version: str,
        error_type: str
    ):
        """Record an error."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['errors_total'].labels(
            model=model,
            version=version,
            error_type=error_type
        ).inc()
    
    def set_active_models(self, count: int):
        """Set number of active models."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.metrics['active_models'].set(count)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus client not available\n"
        
        return generate_latest(self.registry)


class PrometheusExporter:
    """Exporter for Prometheus metrics."""
    
    def __init__(
        self,
        port: int = 9090,
        registry: Optional[MetricsRegistry] = None
    ):
        """
        Initialize Prometheus exporter.
        
        Parameters
        ----------
        port : int
            Port for metrics endpoint
        registry : MetricsRegistry, optional
            Metrics registry to use
        """
        self.port = port
        self.registry = registry or MetricsRegistry()
        self.server = None
        
    def start_server(self):
        """Start metrics server."""
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not available")
        
        from prometheus_client import start_http_server
        
        try:
            start_http_server(self.port, registry=self.registry.registry)
            self.server = True
        except Exception as e:
            raise RuntimeError(f"Failed to start metrics server: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics as text."""
        return self.registry.get_metrics().decode('utf-8')
    
    def create_flask_endpoint(self):
        """Create Flask endpoint for metrics."""
        def metrics_endpoint():
            return self.registry.get_metrics(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        return metrics_endpoint


class SimpleMetricsCollector:
    """Simple metrics collector when Prometheus is not available."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] += value
    
    def set_gauge(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
    
    def observe_histogram(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0):
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Make metric key."""
        if labels:
            label_str = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        import numpy as np
        
        summary = {
            'uptime_seconds': time.time() - self.start_time,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {}
        }
        
        for key, values in self.histograms.items():
            if values:
                summary['histograms'][key] = {
                    'count': len(values),
                    'sum': sum(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'min': min(values),
                    'max': max(values),
                }
        
        return summary
    
    def export_to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # Histograms (simplified)
        for key, values in self.histograms.items():
            if values:
                metric_name = key.split('{')[0]
                lines.append(f"# TYPE {metric_name} histogram")
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
        
        return '\n'.join(lines) + '\n'
