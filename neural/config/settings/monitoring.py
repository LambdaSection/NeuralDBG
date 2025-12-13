"""
Monitoring and metrics configuration settings.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from neural.config.base import BaseConfig


class MonitoringSettings(BaseConfig):
    """Monitoring and metrics configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_MONITORING_"}
    
    # General settings
    enabled: bool = Field(
        default=True,
        description="Enable monitoring"
    )
    collection_interval_seconds: int = Field(
        default=60,
        gt=0,
        description="Metrics collection interval in seconds"
    )
    
    # Prometheus settings
    prometheus_enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics export"
    )
    prometheus_port: int = Field(
        default=9090,
        gt=0,
        lt=65536,
        description="Prometheus metrics port"
    )
    prometheus_path: str = Field(
        default="/metrics",
        description="Prometheus metrics endpoint path"
    )
    
    # System metrics
    collect_cpu_metrics: bool = Field(
        default=True,
        description="Collect CPU metrics"
    )
    collect_memory_metrics: bool = Field(
        default=True,
        description="Collect memory metrics"
    )
    collect_disk_metrics: bool = Field(
        default=True,
        description="Collect disk metrics"
    )
    collect_network_metrics: bool = Field(
        default=True,
        description="Collect network metrics"
    )
    collect_gpu_metrics: bool = Field(
        default=True,
        description="Collect GPU metrics if available"
    )
    
    # Application metrics
    collect_request_metrics: bool = Field(
        default=True,
        description="Collect request metrics"
    )
    collect_error_metrics: bool = Field(
        default=True,
        description="Collect error metrics"
    )
    collect_latency_metrics: bool = Field(
        default=True,
        description="Collect latency metrics"
    )
    
    # Model metrics
    collect_model_metrics: bool = Field(
        default=True,
        description="Collect model training/inference metrics"
    )
    collect_accuracy_metrics: bool = Field(
        default=True,
        description="Collect accuracy metrics"
    )
    collect_loss_metrics: bool = Field(
        default=True,
        description="Collect loss metrics"
    )
    
    # Alerting
    alerting_enabled: bool = Field(
        default=False,
        description="Enable alerting"
    )
    alert_email_recipients: List[str] = Field(
        default=[],
        description="Email recipients for alerts"
    )
    alert_webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for alerts"
    )
    
    # Alert thresholds
    cpu_usage_threshold_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="CPU usage alert threshold"
    )
    memory_usage_threshold_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Memory usage alert threshold"
    )
    disk_usage_threshold_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Disk usage alert threshold"
    )
    error_rate_threshold_percent: float = Field(
        default=5.0,
        ge=0,
        le=100,
        description="Error rate alert threshold"
    )
    
    # Logging integration
    log_metrics: bool = Field(
        default=True,
        description="Log metrics to standard logging"
    )
    log_interval_seconds: int = Field(
        default=300,
        gt=0,
        description="Metrics logging interval in seconds"
    )
    
    # Data retention
    metrics_retention_days: int = Field(
        default=30,
        gt=0,
        description="Metrics data retention in days"
    )
    
    # Health checks
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_check_interval_seconds: int = Field(
        default=30,
        gt=0,
        description="Health check interval in seconds"
    )
    health_check_timeout_seconds: int = Field(
        default=5,
        gt=0,
        description="Health check timeout in seconds"
    )
    
    # Tracing
    tracing_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    tracing_sample_rate: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Tracing sample rate (0-1)"
    )
    jaeger_agent_host: Optional[str] = Field(
        default=None,
        description="Jaeger agent host for tracing"
    )
    jaeger_agent_port: int = Field(
        default=6831,
        gt=0,
        lt=65536,
        description="Jaeger agent port"
    )
