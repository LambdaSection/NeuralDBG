"""
Dashboard (NeuralDbg) configuration settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from neural.config.base import BaseConfig


class DashboardSettings(BaseConfig):
    """Dashboard (NeuralDbg) configuration settings."""
    
    model_config = {"env_prefix": "NEURAL_DASHBOARD_"}
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Dashboard host")
    port: int = Field(
        default=8050,
        gt=0,
        lt=65536,
        description="Dashboard port"
    )
    debug: bool = Field(default=False, description="Debug mode")
    
    # WebSocket settings
    websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket support"
    )
    websocket_interval: int = Field(
        default=1000,
        gt=0,
        description="WebSocket update interval in milliseconds"
    )
    websocket_cors_origins: list[str] = Field(
        default=["http://localhost:8050"],
        description="Allowed WebSocket CORS origins"
    )
    
    # Visualization settings
    update_interval_ms: int = Field(
        default=1000,
        gt=0,
        description="Visualization update interval in milliseconds"
    )
    max_trace_points: int = Field(
        default=1000,
        gt=0,
        description="Maximum trace data points to display"
    )
    theme: str = Field(
        default="darkly",
        description="Dashboard theme (e.g., darkly, flatly, etc.)"
    )
    
    # Performance monitoring
    profiling_enabled: bool = Field(
        default=True,
        description="Enable performance profiling"
    )
    memory_tracking_enabled: bool = Field(
        default=True,
        description="Enable memory tracking"
    )
    gpu_tracking_enabled: bool = Field(
        default=True,
        description="Enable GPU tracking"
    )
    
    # Data retention
    max_execution_history: int = Field(
        default=100,
        gt=0,
        description="Maximum execution history entries"
    )
    trace_data_retention_hours: int = Field(
        default=24,
        gt=0,
        description="Trace data retention in hours"
    )
    
    # Authentication
    auth_enabled: bool = Field(
        default=False,
        description="Enable authentication"
    )
    auth_username: Optional[str] = Field(
        default=None,
        description="Authentication username"
    )
    auth_password: Optional[str] = Field(
        default=None,
        description="Authentication password"
    )
    
    # Features
    enable_layer_profiling: bool = Field(
        default=True,
        description="Enable layer profiling view"
    )
    enable_memory_profiling: bool = Field(
        default=True,
        description="Enable memory profiling view"
    )
    enable_bottleneck_detection: bool = Field(
        default=True,
        description="Enable bottleneck detection"
    )
    enable_recommendations: bool = Field(
        default=True,
        description="Enable performance recommendations"
    )
    enable_distributed_view: bool = Field(
        default=False,
        description="Enable distributed execution view"
    )
    
    # Export settings
    enable_export: bool = Field(
        default=True,
        description="Enable data export functionality"
    )
    export_formats: list[str] = Field(
        default=["json", "csv", "html"],
        description="Supported export formats"
    )
